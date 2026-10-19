import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import inspect
import json
import time
import unittest
from dataclasses import replace
import numpy as np
import pandas as pd
from task4_core.sources import PROJECT, RELEASE, BASE, TASK2, read, sha
from task4_core.router import PreDirectRouter
from task4_core.window import WindowedFixed, WindowedGreedy, WindowedSpatial, ENGINE, SPATIAL, SOURCE_IDENTITIES
from task4_core.accounting import evaluate_window
from task4_core.resources import pools_from_hosts
from task4_core.metrics import quality_harms, fairness, failure_decomposition
from task4_core.risk import cvar, mixture_weights, paired_mixture_cvar
from task4_core.registry import REQUIRED, execution_signature, reusable
from task4_core.qualification import LIMITS, qualify_offline
from src.domain import Request, load_inputs
from src.carbon import Carbon, ForecastView
from src.accounting import evaluate as frozen_evaluate
import queue_milp

OUT = None


def save(name, value):
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')


class Availability:
    def __init__(self, servers, changes=()):
        self.events = [(0., r, m, int(p['capacity'])) for (r, m), p in servers.items()] + list(changes)

    def at(self, r, m, t):
        return [c for when, a, b, c in self.events if a == r and b == m and when <= t][-1]


def fixtures():
    hosts = [{'host_id': r + m, 'region': r, 'model': m, 'gpus': 8, 'slots': 1,
              'shared_w': 1500., 'gpu_idle_w': 175., 'gpu_busy_w': 760., 'pue': 1.2, 'server_max_w': 14300.}
             for r in ['DE', 'FR', 'ES'] for m in ['S', 'L']]
    servers = pools_from_hosts(hosts)
    net = {(a, b): {'delay': 0., 'energy': 1e-6, 'allowed': True} for a in ['DE', 'FR', 'ES'] for b in ['DE', 'FR', 'ES']}
    anchor = pd.Timestamp('2025-01-02T00:00:00Z')
    frame = pd.DataFrame([{'datetime_utc': anchor + pd.Timedelta(hours=h), 'region': r,
                           'carbon_intensity_gco2e_kwh': c}
                          for h in range(-24, 14) for r, c in [('DE', 300.), ('FR', 30.), ('ES', 150.)]])
    carbon = Carbon(frame, anchor)
    cfg = {**read(BASE / 'configs/core.json')['model'], 'router_seconds': 0., 'solver_time_limit': 2.}
    jobs = [Request('a', 'L', 'DE', 'General', 'bounded_delay', 0., 100., 40., 2., 2),
            Request('b', 'L', 'DE', 'General', 'bounded_delay', 0., 100., 40., 2., 2)]
    return jobs, servers, net, carbon, cfg, hosts


def check_equal(old, new, old_eval, new_eval):
    pd.testing.assert_frame_equal(pd.DataFrame(old.log), pd.DataFrame(new.log), check_exact=True)
    om, ore, ol = old_eval
    nm, nre, nl = new_eval
    pd.testing.assert_frame_equal(ore, nre[ore.columns], check_exact=False, rtol=0, atol=1e-8)
    pd.testing.assert_frame_equal(ol, nl[ol.columns], check_exact=False, rtol=0, atol=1e-8)
    ignored = ['solver_wall_seconds']
    for key, expected in om.items():
        if key in ignored:
            continue
        actual = nm[key]
        if isinstance(expected, (float, int)) and not isinstance(expected, bool):
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8, err_msg=key)
        else:
            assert actual == expected, (key, actual, expected)
    assert old.controls == new.controls


class RouterContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.router = PreDirectRouter()

    def test_full_7335_id_decisions_and_pre_predictions(self):
        splits = pd.read_csv(TASK2 / 'runs/experiment/splits.csv', dtype={'request_id': str})
        ids = splits.loc[splits.phase.eq('test'), 'request_id'].tolist()
        self.assertEqual(len(ids), 7335)
        wanted = set(ids)
        with (TASK2 / 'inputs/paired/requests.jsonl').open(encoding='utf-8') as stream:
            minimal = {str(r['request_id']): {k: r[k] for k in ['request_id', 'prompt', 'eval_name']}
                       for r in map(json.loads, stream) if str(r['request_id']) in wanted}
        frame = pd.DataFrame([minimal[rid] for rid in ids])
        began = time.perf_counter()
        predicted = self.router.route_frame(frame)

        paths = pd.read_csv(TASK2 / 'runs/experiment/decisions.csv.gz', dtype={'request_id': str},
                            usecols=['request_id', 'strategy', 'phase', 'rep', 'initial_l', 'final_l', 'calls_s', 'calls_l'])
        expected = paths[paths.strategy.eq('Selected_finite_sample_guard') & paths.phase.eq('test') & paths.rep.eq(0)].set_index('request_id')
        canonical = pd.read_csv(BASE / 'results/data/requests.csv', dtype={'request_id': str}).set_index('request_id')
        self.assertEqual(set(expected.index), set(ids))
        self.assertEqual(set(canonical.index), set(ids))
        predicted = predicted.set_index('request_id').loc[ids]
        trace = predicted.copy()
        trace['task2_expected'] = np.where(expected.loc[ids].initial_l, 'L', 'S')
        trace['task3_expected'] = canonical.loc[ids].selected_model
        trace['matches_task2'] = trace.selected_model.eq(trace.task2_expected)
        trace['matches_task3'] = trace.selected_model.eq(trace.task3_expected)
        trace.to_csv(OUT / 'zero_perturbation_all_7335.csv', index=True)
        detail = {'status': 'PASS' if trace.matches_task2.all() and trace.matches_task3.all() else 'FAIL',
                  'requests': len(ids), 'matched_task2_ids': int(trace.matches_task2.sum()),
                  'matched_task3_ids': int(trace.matches_task3.sum()),
                  'route_counts': trace.selected_model.value_counts().to_dict(),
                  'inference_wall_s': time.perf_counter() - began, 'model_hashes': self.router.hashes,
                  'reference_used_only_after_inference': True, 'training_executed': False}
        save('zero_perturbation.json', detail)
        self.assertTrue(trace.matches_task2.all() and trace.matches_task3.all())
        np.testing.assert_array_equal(predicted.calls_l, expected.loc[ids].calls_l)
        np.testing.assert_array_equal(predicted.calls_s, expected.loc[ids].calls_s)
        np.testing.assert_array_equal(expected.loc[ids].initial_l, expected.loc[ids].final_l)
        np.testing.assert_allclose(predicted.pre_S, canonical.loc[ids].q_hat_S, rtol=0, atol=1e-9)
        np.testing.assert_allclose(predicted.pre_L, canonical.loc[ids].q_hat_L, rtol=0, atol=1e-9)

    def test_whitelist_truth_poison_fallback_and_invalid_error(self):
        sample = pd.DataFrame({'request_id': ['unknown', 'math', 'code', 'general'],
                               'prompt': ['A valid question'] * 4,
                               'eval_name': ['unregistered_task', 'gsm8k', 'mbpp', 'hellaswag']})
        first = self.router.route_frame(sample)
        sample['q_s'], sample['q_l'], sample['output_tokens'] = 1., 0., 10**9
        second = self.router.route_frame(sample)
        pd.testing.assert_frame_equal(first, second)
        shifted = self.router.route_frame(sample, error_s=.5, error_l=-.5)
        self.assertTrue(shifted.iloc[:3].selected_model.eq('L').all())
        self.assertFalse(shifted.unchanged_prediction_rule.any())
        with self.assertRaises(ValueError):
            self.router.route_frame(sample, error_s=np.nan)


class ObserverContracts(unittest.TestCase):
    def test_final_optimizer_identity(self):
        self.assertIs(WindowedSpatial.optimize, SPATIAL.SpatialSimulation.optimize)
        self.assertIn('final_release', SOURCE_IDENTITIES['engine_file'])
        save('source_identities.json', SOURCE_IDENTITIES)

    def test_hand_no_censor_final_engine_and_ledger_equivalence(self):
        jobs, servers, net, carbon, cfg, _ = fixtures()
        shocks = {'a': 1.1, 'b': .9}
        for old_type, new_type, strategy in [(ENGINE.Simulation, WindowedGreedy, 'Greedy'),
                                              (SPATIAL.SpatialSimulation, WindowedSpatial, 'MPC')]:
            old = old_type(jobs, servers, net, Availability(servers), carbon, shocks, strategy, cfg).run()
            new = new_type(jobs, servers, net, Availability(servers), carbon, shocks, cfg=cfg).run()
            check_equal(old, new, frozen_evaluate(old, 43200), evaluate_window(new))

    def _check_original_cohort(self, n):

        jobs, servers, net, anchor = load_inputs(n=n)
        frame = pd.read_csv(BASE / 'results/data/carbon_lifecycle.csv')
        times = pd.to_datetime(frame.datetime_utc, utc=True)
        frame = frame[(times >= anchor - pd.Timedelta(hours=24)) & (times < anchor + pd.Timedelta(hours=13))]
        carbon = Carbon(frame, anchor, read(BASE / 'results/data/carbon_lifecycle.metadata.json'))
        cfg = {**read(BASE / 'configs/core.json')['model'],
               **read(RELEASE.parent / 'task_a2/configs/solver_protocol_frozen.json')['model_overrides']}
        all_shocks = read(RELEASE.parent / 'task_a2/extensions/common_shocks/seed_2027101800.json')
        shocks = {q.request_id: all_shocks[q.request_id] for q in jobs}
        quality = pd.read_csv(BASE / 'results/data/evaluation_only_historical_quality.csv').set_index('request_id').q.to_dict()
        checks = []
        for old_type, new_type, strategy in [(ENGINE.Simulation, WindowedGreedy, 'Greedy'),
                                              (SPATIAL.SpatialSimulation, WindowedSpatial, 'MPC')]:
            began = time.perf_counter()
            old = old_type(jobs, servers, net, Availability(servers), carbon, shocks, strategy, cfg).run()
            new = new_type(jobs, servers, net, Availability(servers), carbon, shocks, cfg=cfg).run()
            old_result, new_result = frozen_evaluate(old, 43200, quality), evaluate_window(new, quality=quality)
            check_equal(old, new, old_result, new_result)
            self.assertEqual(new_result[0]['censored'], 0)
            checks.append({'policy': new.strategy, 'n': n, 'seed': 2027101800,
                           'status': 'PASS', 'runtime_s': time.perf_counter() - began,
                           'events': len(new.log), 'solver_calls': len(new.solver_log),
                           'comparison': 'exact events/controls; shared record, ledger and non-wall metric fields atol=1e-8'})
        save('original_world_equivalence.json' if n == 1000 else 'original_full_7335_equivalence.json', checks)

    def test_original_1000_no_censor_equivalence(self):
        self._check_original_cohort(1000)

    def test_original_7335_no_censor_equivalence(self):
        self._check_original_cohort(7335)

    def test_partial_running_queued_and_remaining_work(self):
        jobs, servers, net, carbon, cfg, _ = fixtures()
        jobs = [replace(jobs[0], arrival=43198., deadline=43300., service=1.),
                replace(jobs[1], arrival=43199., deadline=43300., service=1.)]
        sim = WindowedFixed(jobs, servers, net, Availability(servers), carbon, {'a': 10., 'b': 1.}, cfg=cfg).run()
        metrics, records, ledger = evaluate_window(sim)
        self.assertEqual(metrics['censored'], 2)
        self.assertEqual(metrics['running_at_H'], 1)
        self.assertEqual(metrics['queued_at_H'], 1)
        self.assertEqual(metrics['remaining_work_nominal_s'], 1.)
        self.assertEqual(metrics['remaining_running_actual_s_evaluation_only'], 8.)
        self.assertAlmostEqual(metrics['active_increment_kwh'], 2 * 585 * 1.2 / 3.6e6)
        self.assertAlmostEqual(metrics['network_kwh'], 2e-6)
        self.assertTrue(records.service_constraint_violation.all())
        self.assertTrue((ledger.end_s <= 43200).all())

    def test_router_and_solver_control_clipped_different_pue(self):
        jobs, servers, net, carbon, cfg, _ = fixtures()
        cfg.update(router_seconds=.01, controller_pue={'DE': 1.6, 'FR': 1.2, 'ES': 1.2})
        q = replace(jobs[0], arrival=43199.995, deadline=43300.)
        sim = WindowedFixed([q], servers, net, Availability(servers), carbon, {'a': 1.}, cfg=cfg).run()
        m, _, _ = evaluate_window(sim)
        self.assertAlmostEqual(m['controller_kwh'], .005 * 10 * 1.6 / 3.6e6, places=12)
        self.assertEqual(m['unfinished_stages'], {'routing': 1})
        cfg.update(router_seconds=0., arrival_batch_seconds=0., solve_delay_seconds=2.)
        q = replace(q, arrival=43199.75)
        sim = WindowedSpatial([q], servers, net, Availability(servers), carbon, {'a': 1.}, cfg=cfg).run()
        m, _, _ = evaluate_window(sim)
        self.assertAlmostEqual(m['controller_kwh'], .25 * 10 * 1.6 / 3.6e6)
        self.assertEqual(m['active_increment_kwh'], 0.)
        self.assertEqual(m['unfinished_stages'], {'pending': 1})

    def test_completion_at_H_and_observer_single_use(self):
        jobs, servers, net, carbon, cfg, _ = fixtures()
        q = replace(jobs[0], arrival=43198., deadline=43200., service=1.)
        sim = WindowedFixed([q], servers, net, Availability(servers), carbon, {'a': 2.}, cfg=cfg).run()
        m, records, _ = evaluate_window(sim)
        self.assertEqual(m['completed'], 1)
        self.assertEqual(m['censored'], 0)
        self.assertFalse(records.service_constraint_violation.any())
        with self.assertRaises(RuntimeError):
            sim.run()
        with self.assertRaises(ValueError):
            evaluate_window(sim, horizon=43000)

    def test_shocks_private_until_admission_and_unstarted_never_read(self):
        jobs, servers, net, carbon, cfg, _ = fixtures()
        class Hidden(dict):
            def __getitem__(self, key):
                if inspect.currentframe().f_back.f_code.co_name != '_admit':
                    raise AssertionError('Unrevealed service accessed')
                return super().__getitem__(key)
        sim = WindowedSpatial(jobs, servers, net, Availability(servers), carbon, Hidden(a=1., b=1.), cfg=cfg).run()
        evaluate_window(sim)
        jobs = [replace(jobs[0], arrival=43198., deadline=43300., service=1.),
                replace(jobs[1], arrival=43199., deadline=43300., service=1.)]
        sim = WindowedFixed(jobs, servers, net, Availability(servers), carbon, Hidden(a=10.), cfg=cfg).run()
        self.assertEqual(sim.window_summary['unstarted_remaining_actual_unknown_count'], 1)

    def test_future_carbon_capacity_and_arrival_prefix(self):
        jobs, servers, net, carbon, cfg, _ = fixtures()
        first = WindowedSpatial(jobs, servers, net, Availability(servers), carbon, {'a': 1., 'b': 1.}, cfg=cfg).run()
        carbon.values[carbon.index(3600.):] *= 5
        later = replace(jobs[1], request_id='future', arrival=3600., deadline=3700.)
        second = WindowedSpatial(jobs + [later], servers, net,
                                 Availability(servers, [(3600., 'FR', 'L', 0)]), carbon,
                                 {'a': 1., 'b': 1., 'future': 1.}, cfg=cfg).run()
        a = pd.DataFrame(first.log)
        b = pd.DataFrame(second.log)
        pd.testing.assert_frame_equal(a[a.time_s < 100].reset_index(drop=True),
                                      b[b.time_s < 100][a.columns].reset_index(drop=True), check_exact=True)
        self.assertTrue(all(log['observed_carbon_end_s'] <= log['decision_s'] for log in second.solver_log))

    def test_same_time_capacity_barrier_no_power_off(self):
        jobs, servers, net, carbon, cfg, _ = fixtures()
        jobs[1] = replace(jobs[1], arrival=2.)
        sim = WindowedFixed(jobs, servers, net, Availability(servers, [(2., 'DE', 'L', 0), (5., 'DE', 'L', 1)]),
                             carbon, {'a': 1., 'b': 1.}, cfg=cfg).run()
        self.assertGreaterEqual(sim.rec['b']['service_start_s'], 5.)
        self.assertAlmostEqual(evaluate_window(sim)[0]['idle_kwh'], 250.56)

    def test_multiple_hosts_slots_and_region_pue(self):
        jobs, _, net, carbon, cfg, hosts = fixtures()
        hosts.append({**hosts[1], 'host_id': 'DEL_second'})
        for host in hosts:
            host['pue'] = {'DE': 1.1, 'FR': 1.3, 'ES': 1.6}[host['region']]
        servers = pools_from_hosts(hosts)
        sim = WindowedFixed(jobs, servers, net, Availability(servers), carbon, {'a': 1., 'b': 1.}, cfg=cfg).run()
        m, records, ledger = evaluate_window(sim)
        expected = sum(2900 * h['pue'] * 43200 / 3.6e6 for h in hosts)
        self.assertAlmostEqual(m['idle_kwh'], expected)
        self.assertEqual(ledger[ledger.component.eq('idle')].host_id.nunique(), 7)
        self.assertTrue(records.service_start_s.eq(0).all())
        self.assertAlmostEqual(m['active_increment_kwh'], 4 * 585 * 1.1 / 3.6e6)
        self.assertEqual(ledger[ledger.component.eq('active_increment')].host_id.nunique(), 2)
        self.assertAlmostEqual(m['network_kwh'], 2e-6)
        hosts[-1]['pue'] = 1.5
        with self.assertRaises(ValueError):
            pools_from_hosts(hosts)

    def test_invalid_physical_hosts_and_window_inputs(self):
        jobs, servers, net, carbon, cfg, hosts = fixtures()
        with self.assertRaises(ValueError):
            pools_from_hosts(hosts + [hosts[0]])
        hosts[0]['slots'] = 9
        with self.assertRaises(ValueError):
            pools_from_hosts(hosts)
        with self.assertRaises(ValueError):
            WindowedFixed([replace(jobs[0], arrival=43200., deadline=43300.)], servers, net, Availability(servers), carbon, {'a': 1.}, cfg=cfg)
        with self.assertRaises(ValueError):
            WindowedSpatial(jobs, servers, net, Availability(servers), carbon, {'a': 1., 'b': 1.}, cfg={**cfg, 'measured_wall_fraction': 1})

    def test_slots_without_hosts_and_inflight_window(self):
        jobs, _, net, carbon, cfg, hosts = fixtures()
        hosts[1]['slots'] = 2
        servers = pools_from_hosts(hosts)
        sim = WindowedFixed(jobs, servers, net, Availability(servers), carbon, {'a': 1., 'b': 1.}, cfg=cfg).run()
        m, records, _ = evaluate_window(sim)
        self.assertAlmostEqual(m['idle_kwh'], 250.56)
        self.assertTrue(records.service_start_s.eq(0.).all())
        net[('DE', 'DE')]['delay'] = 10.
        q = replace(jobs[0], arrival=43199., deadline=43300.)
        sim = WindowedFixed([q], servers, net, Availability(servers), carbon, {}, cfg=cfg).run()
        m, _, _ = evaluate_window(sim)
        self.assertEqual(m['unfinished_stages'], {'inflight': 1})
        self.assertEqual(m['active_increment_kwh'], 0.)
        self.assertAlmostEqual(m['network_kwh'], 1e-6)

    def test_capacity_queue_and_execution_failure_tags(self):
        jobs, servers, net, carbon, cfg, _ = fixtures()
        availability = Availability(servers)
        availability.events = [(t, r, m, 0 if (r,m)==('DE','L') else c) for t,r,m,c in availability.events]
        sim = WindowedFixed([jobs[0]], servers, net, availability, carbon, {}, cfg=cfg).run()
        _, records, _ = evaluate_window(sim)
        tags = failure_decomposition(sim, records)
        self.assertTrue(tags.server_capacity_exposure.all())
        self.assertTrue(tags.queue_expiration.all())
        q = replace(jobs[0], service=1., deadline=3.)
        sim = WindowedFixed([q], servers, net, Availability(servers), carbon, {'a': 5.}, cfg=cfg).run()
        _, records, _ = evaluate_window(sim)
        self.assertTrue(failure_decomposition(sim, records).execution_timeout.all())

    def test_numerical_reject_real_fallback_lifecycle(self):
        from unittest.mock import patch
        jobs, servers, net, carbon, cfg, _ = fixtures()
        log = {'status': 4, 'wall_seconds': 0., 'engine': 'contract_injected_rejection',
               'recovery': {'unrecoverable': True}}
        with patch.object(SPATIAL, 'spatial_solve', return_value=([], log)):
            sim = WindowedSpatial(jobs, servers, net, Availability(servers), carbon, {'a': 1., 'b': 1.}, cfg=cfg).run()
        m, records, _ = evaluate_window(sim)
        tags = failure_decomposition(sim, records)
        self.assertTrue(tags.milp_numerical_rejection.all())
        self.assertTrue(tags.fallback_used.all())
        self.assertEqual(m['completed'], 2)
        self.assertEqual(m['fallback_count'], 2)

    def test_cross_hour_and_independent_carbon_ledger(self):
        jobs, servers, net, carbon, cfg, _ = fixtures()
        carbon.values[carbon.index(0.), 0] = 100
        carbon.values[carbon.index(3600.), 0] = 200
        q = replace(jobs[0], arrival=3590., deadline=3700., service=20.)
        sim = WindowedFixed([q], servers, net, Availability(servers), carbon, {'a': 1.}, cfg=cfg).run()
        m, _, ledger = evaluate_window(sim)
        self.assertAlmostEqual(m['active_increment_co2e_g'], 585 * 1.2 * (10 * 100 + 10 * 200) / 3.6e6)
        independent = 0.
        for rec in ledger.itertuples():
            if rec.end_s == rec.start_s:
                independent += rec.energy_kwh * carbon.at(rec.start_s, rec.region)
            else:
                for h in range(12):
                    overlap = max(0., min(rec.end_s, (h+1)*3600) - max(rec.start_s, h*3600))
                    independent += rec.energy_kwh * overlap / (rec.end_s-rec.start_s) * carbon.at(h*3600, rec.region)
        self.assertAlmostEqual(m['co2e_g'], independent, places=8)


class MetricContracts(unittest.TestCase):
    def test_quality_damage_additivity_including_better_than_L(self):
        records = pd.DataFrame({'request_id': ['a', 'b', 'c'], 'selected_model': ['S']*3,
                                'status': ['completed', 'failed', 'failed'], 'service_constraint_violation': [False, True, True]})
        quality = pd.DataFrame({'request_id': ['a', 'b', 'c'], 'q_s': [.2, .2, 1.], 'q_l': [.8, .8, .5]})
        result = quality_harms(records, quality)
        np.testing.assert_allclose(result.model_selection_harm, [.6, .6, 0.])
        np.testing.assert_allclose(result.delivery_incremental_harm, [0., .2, .5])
        np.testing.assert_allclose(result.end_to_end_harm, result.model_selection_harm + result.delivery_incremental_harm)
        np.testing.assert_allclose(result.end_to_end_signed_loss, result.model_selection_signed_loss + result.delivery_quality_lost)

    def test_fairness_failures_zero_W_and_missing_standardization(self):
        frame = pd.DataFrame({'request_id': ['a', 'b', 'c'], 'group': ['g', 'g', 'other'], 'task_type': ['Code', 'Code', 'Math'],
                              'service_class': ['realtime', 'batch', 'realtime'], 'status': ['failed', 'censored_at_H', 'completed'],
                              'selected_model': ['S']*3, 'service_constraint_violation': [True, True, False],
                              'response_s': [np.nan, np.nan, 1.], 'max_wait_s': [0., 2., 0.],
                              'intentional_delay_s': [np.nan, np.nan, 0.], 'queue_wait_s': [np.nan, np.nan, 0.]})
        frame = quality_harms(frame, pd.DataFrame({'request_id': ['a', 'b', 'c'], 'q_s': [.5]*3, 'q_l': [.8]*3}))
        result = fairness(frame, ['task_type'], {'realtime': .5, 'batch': .5}).set_index('task_type')
        self.assertEqual(result.loc['Code', 'violation_rate'], 1.)
        self.assertEqual(result.loc['Code', 'unique_prompt_groups'], 1)
        self.assertEqual(result.loc['Code', 'censored'], 1)
        self.assertTrue(pd.isna(result.loc['Math', 'standardized_violation_rate']))
        self.assertTrue(pd.isna(result.loc['Math', 'completed_mean_wait_budget_ratio']))

    def test_failure_tags_not_exclusive_causal_proof(self):
        jobs, servers, net, carbon, cfg, _ = fixtures()
        q = replace(jobs[0], arrival=43198., deadline=43300., service=1.)
        sim = WindowedFixed([q], servers, net, Availability(servers), carbon, {'a': 10.}, cfg=cfg).run()
        _, records, _ = evaluate_window(sim)
        records.loc[:, 'fallback_reason'] = 'solver_status_4'
        sim.solver_log = [{'status': 4, 'request_ids': ['a'], 'recovery': {'unrecoverable': True}}]
        result = failure_decomposition(sim, records)
        self.assertTrue(result.window_censoring.all())
        self.assertTrue(result.milp_numerical_rejection.all())
        self.assertTrue(result.fallback_used.all())
        self.assertFalse(result.execution_timeout.any())
        self.assertEqual(result.primary_service_outcome.tolist(), ['window_censoring'])

    def test_discrete_cvar_fractional_ties_and_zero_probability(self):
        self.assertAlmostEqual(cvar([0, 1, 2, 3], .625)['cvar'], 8/3)
        self.assertEqual(cvar([0, 10], .5, [.9, .1])['cvar'], 2.)
        constant = cvar([7]*100, .95)
        self.assertAlmostEqual(constant['cvar'], 7.)
        self.assertEqual(constant['distinct_tail_losses'], 1)
        self.assertAlmostEqual(cvar([0, 5, 9], .5, [.5, .5, 0.])['cvar'], 5.)
        self.assertAlmostEqual(cvar([1, 2, 3], 0.)['cvar'], 2.)
        for losses, weights in [([np.nan], None), ([], None), ([0, 1], [1., 1.]), ([0, 1], [1., -0.1])]:
            with self.assertRaises(ValueError):
                cvar(losses, weights=weights)

    def test_cvar_paired_bootstrap_and_probability_sensitivity(self):
        regimes = np.r_[np.zeros(50, int), np.ones(50, int)]
        base = np.r_[np.arange(50)/100, 1 + np.arange(50)/100]
        first = paired_mixture_cvar(base + 2, base, regimes, bootstrap=100, seed=42)
        second = paired_mixture_cvar(base + 2, base, regimes, bootstrap=100, seed=42)
        self.assertEqual(first, second)
        np.testing.assert_allclose(first['ci95_difference'], [2., 2.])
        pure = cvar(base, weights=mixture_weights(regimes, 0))['cvar']
        mixed = cvar(base, weights=mixture_weights(regimes, .4))['cvar']
        self.assertGreater(mixed, pure)

        regimes = np.r_[np.zeros(1000, int), np.ones(1000, int)]
        losses = np.arange(2000.)
        for probability in [0, .1, .2, .4]:
            self.assertGreaterEqual(cvar(losses, weights=mixture_weights(regimes, probability))['tail_ess'], 50 - 1e-8)

    def test_cvar_against_convex_definition(self):
        rng = np.random.default_rng(11)
        for _ in range(20):
            values = rng.integers(0, 5, 10).astype(float)
            weights = rng.random(10)
            weights /= weights.sum()
            alpha = float(rng.uniform(0, .99))
            oracle = min(t + weights @ np.maximum(values-t, 0)/(1-alpha) for t in values)
            self.assertAlmostEqual(cvar(values, alpha, weights)['cvar'], oracle)

    def test_fairness_simpson_standardization(self):
        rows = []

        for group, service_class, n, violations in [('A','easy',10,1),('A','hard',90,45),
                                                   ('B','easy',90,18),('B','hard',10,6)]:
            for i in range(n):
                rows.append({'task_type': group, 'service_class': service_class, 'group': f'{group}|{service_class}|{i}',
                             'status': 'completed', 'service_constraint_violation': i < violations,
                             'model_selection_harm': 0., 'delivery_incremental_harm': 0., 'end_to_end_harm': 0., 'max_wait_s': 1.})
        result = fairness(pd.DataFrame(rows), ['task_type'], {'easy': .5, 'hard': .5}).set_index('task_type')
        self.assertGreater(result.loc['A','violation_rate'], result.loc['B','violation_rate'])
        self.assertLess(result.loc['A','standardized_violation_rate'], result.loc['B','standardized_violation_rate'])

    def test_strict_reuse_rejects_any_configuration_or_artifact_change(self):
        config = dict.fromkeys(REQUIRED, 'contract_value')
        path = OUT / 'toy_artifact.txt'
        path.write_text('contract artifact only', encoding='utf-8')
        record = {'status': 'COMPLETE_VALIDATED', 'signature': execution_signature(config),
                  'artifact_hashes': {str(path): sha(path)}}
        self.assertTrue(reusable(record, config))
        self.assertFalse(reusable(record, {**config, 'horizon_s': 86400}))
        self.assertFalse(reusable({**record, 'status': 'INCOMPLETE'}, config))
        path.write_text('changed toy artifact', encoding='utf-8')
        self.assertFalse(reusable(record, config))

    def test_offline_qualification_does_not_claim_production_validity(self):
        evidence = {name: value/2 for name, value in LIMITS.items()}
        evidence.update(total_carbon_saving_lower=1., minimum_tail_ess=50., contracts_passed=True, worlds_complete=True)
        self.assertEqual(qualify_offline(evidence)['status'], 'CONDITIONAL_SIMULATION_CANDIDATE_ONLY')
        self.assertEqual(qualify_offline({**evidence, 'minimum_tail_ess': 49})['status'], 'INSUFFICIENT_EVIDENCE')
        self.assertEqual(qualify_offline({**evidence, 'total_carbon_saving_lower': -1})['status'], 'NOT_QUALIFIED_IN_TESTED_SCOPE')
        self.assertEqual(qualify_offline({})['status'], 'INSUFFICIENT_EVIDENCE')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    OUT = args.out.resolve()
    if ROOT.resolve() not in OUT.parents or OUT.exists():
        raise ValueError('Use a NEW attempt directory inside TASK 4')
    OUT.mkdir(parents=True)
    queue_milp.ROOT = OUT / 'scratch'
    used = [p for p in RELEASE.glob('*.py')] + list((RELEASE / 'v2').glob('*.py'))
    used += list((BASE / 'src').glob('*.py')) + list((BASE / 'results/data').glob('*.csv')) + list((BASE / 'results/data').glob('*.json'))
    used += [TASK2 / 'runs/experiment' / name for name in ['pre_full_train.joblib', 'selected_parameters.json',
             'final_model_choice.json', 'final_safety_calibration.json', 'cost_predictor.json', 'decisions.csv.gz']]
    before = {str(p): sha(p) for p in used}
    save('used_source_hashes_before.json', before)
    began = time.perf_counter()
    suite = unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c)
                               for c in [RouterContracts, ObserverContracts, MetricContracts])
    with (OUT / 'contracts.log').open('w', encoding='utf-8') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    changed = [p for p, h in before.items() if sha(p) != h]
    report = {'status': 'PASS' if result.wasSuccessful() and not changed else 'FAIL',
              'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors),
              'runtime_seconds': time.perf_counter() - began, 'upstream_used_files_checked': len(before),
              'changed_upstream_files': changed, 'formal_M1_M8_experiments_executed': 0, 'training_executed': False,
              'scope': 'full 7335 inference identity; toy contracts; one original world at N=1000 and N=7335 for two policies/two engines; no stress/performance grid'}
    save('contract_results.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if report['status'] == 'PASS' else 1)
