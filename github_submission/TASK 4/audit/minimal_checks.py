import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CANONICAL = Path('C:/Users/吴连杰/.codex/worktrees/47e1/EuroMCM-project')
RELEASE = CANONICAL / 'TASK 3/task3_model/results/robustness_v2/final_release'
BASE = RELEASE.parents[2]
sys.path[:0] = [str(RELEASE), str(BASE), str(PROJECT / 'task2')]

import inspect
import json
import time
import unittest
from dataclasses import replace
import joblib
import numpy as np
import pandas as pd
from scipy.stats import qmc
from src.domain import Request
from src.carbon import Carbon, ForecastView
from src.accounting import evaluate
from v2.des import Simulation
from spatial import SpatialSimulation
import queue_milp
queue_milp.ROOT = HERE / 'scratch'
from routing import apply, slice_prediction
from risk_bounds import binomial_upper


def empirical_cvar(losses, alpha, weights=None):

    values = np.asarray(losses, dtype=float)
    if not len(values) or not np.isfinite(values).all() or not 0 <= alpha < 1:
        raise ValueError('Invalid loss distribution')
    w = np.ones(len(values)) / len(values) if weights is None else np.asarray(weights, dtype=float)
    if (w < 0).any() or not np.isclose(w.sum(), 1.):
        raise ValueError('Invalid scenario probabilities')
    return min(float(t + w @ np.maximum(values - t, 0.) / (1 - alpha)) for t in values)


class Available:
    def __init__(self, servers, extra=()):
        self.servers = servers
        self.events = [(0., r, m, p['capacity']) for (r, m), p in servers.items()] + list(extra)

    def at(self, r, m, t):
        return [c for s, a, b, c in self.events if a == r and b == m and s <= t][-1]


def fixture():
    anchor = pd.Timestamp('2025-01-02T00:00:00Z')
    rows = [{'datetime_utc': anchor + pd.Timedelta(hours=h), 'region': r,
             'carbon_intensity_gco2e_kwh': c}
            for h in range(-24, 14) for r, c in [('DE', 300.), ('FR', 30.), ('ES', 150.)]]
    carbon = Carbon(pd.DataFrame(rows), anchor)
    servers = {(r, m): {'capacity': 1, 'idle': 2900., 'busy': 3485., 'pue': 1.2,
                         'idle_base_w': 2900., 'installed_gpus': 8, 'host_id': r + m}
               for r in ['DE', 'FR', 'ES'] for m in ['S', 'L']}
    network = {(a, b): {'delay': 0., 'energy': 1e-6, 'allowed': True}
               for a in ['DE', 'FR', 'ES'] for b in ['DE', 'FR', 'ES']}
    cfg = {'observation_horizon_seconds': 43200, 'router_seconds': 0., 'solve_delay_seconds': .01,
           'solver_time_limit': 2., 'service_guard': 1.35, 'max_batch': 8, 'record_events': True}
    jobs = [Request('toy_a', 'L', 'DE', 'General', 'bounded_delay', 0., 100., 40., 2., 2),
            Request('toy_b', 'L', 'DE', 'General', 'bounded_delay', 2., 100., 40., 2., 2)]
    return jobs, servers, network, carbon, cfg


class Checks(unittest.TestCase):
    def test_01_saved_router_inference_six_examples(self):
        run = PROJECT / 'task2/runs/experiment'
        from q1_adapter import BASE as FEATURES
        import importlib.util
        spec = importlib.util.spec_from_file_location('task4_readonly_common', PROJECT / 'TASK 3/data_preparation/common.py')
        common = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(common)
        choice = json.loads((run / 'final_model_choice.json').read_text())
        self.assertEqual(choice['family'], 'Pre_direct')
        self.assertEqual(set(choice['fallback_tasks']), {'Code', 'Math'})
        pre = joblib.load(run / 'pre_full_train.joblib')
        params = json.loads((run / 'selected_parameters.json').read_text())['Pre_direct']['General']['parameters']
        cost = json.loads((run / 'cost_predictor.json').read_text())
        requests = pd.read_csv(BASE / 'results/data/requests.csv').set_index('request_id')
        chosen = []
        for model in ['S', 'L']:
            chosen.extend(requests[requests.selected_model.eq(model) & requests.task_type.eq('General')].index[:2])
        for task in ['Code', 'Math']:
            chosen.extend(requests[requests.task_type.eq(task)].index[:1])
        with (PROJECT / 'task2/inputs/paired/requests.jsonl').open(encoding='utf-8') as stream:
            frame = {r['request_id']: r for r in map(json.loads, stream) if r['request_id'] in chosen}
        for rid in chosen:
            row = frame[rid]
            data = pd.DataFrame([{'token_length': len(row['prompt']), 'eval_name': row['eval_name'], 'model_id': m}
                                 for m in ['mistralai/mixtral-8x7b-chat', 'gpt-4-1106-preview']])
            prediction = np.clip(pre.models['A_Baseline'].predict(pre.basic(data)[FEATURES]), 0, 1)
            np.testing.assert_allclose(prediction, requests.loc[rid, ['q_hat_S', 'q_hat_L']].to_numpy(float), atol=1e-9)
            action = common.frozen_route(requests.at[rid, 'task_type'], row['eval_name'], *prediction,
                                        len(row['prompt']), pre.benchmark_names, cost, params, choice['fallback_tasks'])
            self.assertEqual(action, requests.at[rid, 'selected_model'])
        self.assertEqual(len(chosen), 6)
        self.assertEqual(common.frozen_route('General', 'unregistered_task', .9, .1, 10,
                                            pre.benchmark_names, cost, params, choice['fallback_tasks']), 'L')

    def test_02_truth_poisoning_cannot_change_frozen_route(self):
        run = PROJECT / 'task2/runs/experiment'

        frame = pd.read_csv(BASE / 'results/data/requests.csv', nrows=12)
        cost = json.loads((run / 'cost_predictor.json').read_text())
        for action in ['s', 'l']:
            frame['cost_' + action] = np.column_stack([np.ones(len(frame)), frame.prompt_chars / 1000]) @ cost['coefficients'][action]
        frame['cost_delta'] = (frame.cost_l - frame.cost_s) / cost['normalization_c0']
        frame['q_s'], frame['q_l'] = .2, .8
        mu = (frame.q_hat_L - frame.q_hat_S).to_numpy()
        pred = {'scores': mu[None], 'mu': mu, 'std': np.zeros(len(frame))}
        params = json.loads((run / 'selected_parameters.json').read_text())['Pre_direct']
        fallback = json.loads((run / 'final_model_choice.json').read_text())['fallback_tasks']
        before = apply(frame, pred, params, fallback)
        np.testing.assert_array_equal(before['initial_l'][0], frame.selected_model.eq('L'))
        poisoned = frame.copy()
        poisoned['q_s'] = 1. - poisoned.q_s
        poisoned['q_l'] = 1. - poisoned.q_l
        after = apply(poisoned, pred, params, fallback)
        for name in ['initial_l', 'final_l', 'calls_s', 'calls_l']:
            np.testing.assert_array_equal(before[name], after[name])

    def test_03_zero_events_have_nonzero_risk_bound(self):
        alpha = .05 / 60
        self.assertAlmostEqual(binomial_upper(0, 43, alpha), 1 - alpha ** (1 / 43))
        self.assertGreater(binomial_upper(0, 43, alpha), .02)

    def test_04_future_carbon_cannot_change_issued_forecast(self):
        _, _, _, carbon, _ = fixture()
        old = ForecastView(carbon, 100., 10800.)
        carbon.values[carbon.index(0.):] *= 5.
        new = ForecastView(carbon, 100., 10800.)
        self.assertEqual(old.table, new.table)
        self.assertLessEqual(new.cutoff, new.issued)

    def test_05_same_time_capacity_before_admission_and_no_idle_discount(self):
        jobs, servers, net, carbon, cfg = fixture()
        availability = Available(servers, [(2., 'DE', 'L', 0), (5., 'DE', 'L', 1)])
        sim = Simulation(jobs, servers, net, availability, carbon, {'toy_a': 1., 'toy_b': 1.}, 'Fixed', cfg).run()
        self.assertEqual(sim.rec['toy_a']['completion_s'], 2.)
        self.assertGreaterEqual(sim.rec['toy_b']['service_start_s'], 5.)
        metrics, records, ledger = evaluate(sim, 43200)
        self.assertAlmostEqual(metrics['idle_kwh'], 250.56)
        self.assertEqual(len(records), 2)
        self.assertAlmostEqual(ledger.energy_kwh.sum(), metrics['energy_kwh'])

    def test_06_spatial_uses_shocks_only_at_admission_and_retains_pairing(self):
        jobs, servers, net, carbon, cfg = fixture()
        class Guarded(dict):
            def __getitem__(self, key):
                assert inspect.currentframe().f_back.f_code.co_name == '_admit'
                return super().__getitem__(key)
        shocks = Guarded(toy_a=1.1, toy_b=.9)
        sims = [kind(jobs, servers, net, Available(servers), carbon, shocks, strategy, cfg).run()
                for kind, strategy in [(Simulation, 'Greedy'), (SpatialSimulation, 'MPC')]]
        for sim in sims:
            for q in jobs:
                self.assertAlmostEqual(sim.rec[q.request_id]['actual_service_s'], q.service * dict(shocks)[q.request_id])
            self.assertTrue(all(log['max_request_arrival_s'] <= log['decision_s'] for log in sim.solver_log))

    def test_07_slot_increase_never_multiplies_host_idle(self):
        jobs, servers, net, carbon, cfg = fixture()
        servers[('DE', 'L')]['capacity'] = 2
        jobs[1] = replace(jobs[1], arrival=0.)
        sim = Simulation(jobs, servers, net, Available(servers), carbon, {'toy_a': 1., 'toy_b': 1.}, 'Fixed', cfg).run()
        metrics, records, _ = evaluate(sim, 43200)
        self.assertTrue(records.service_start_s.eq(0.).all())
        self.assertAlmostEqual(metrics['idle_kwh'], 250.56)
        self.assertAlmostEqual(metrics['active_increment_kwh'], 4 * 585 * 1.2 / 3.6e6)
        self.assertAlmostEqual(metrics['network_kwh'], 2e-6)

    def test_08_cross_hour_integral(self):
        _, _, _, carbon, _ = fixture()
        carbon.values[carbon.index(0.), 0] = 100.
        carbon.values[carbon.index(3600.), 0] = 200.
        self.assertAlmostEqual(carbon.integral(3590., 3610., 'DE'), 3000.)

    def test_09_frozen_engine_refuses_window_overflow(self):
        jobs, servers, net, carbon, cfg = fixture()
        q = replace(jobs[0], deadline=20., service=2.)
        cfg['observation_horizon_seconds'] = 30.
        with self.assertRaisesRegex(RuntimeError, 'Observation window'):
            Simulation([q], servers, net, Available(servers), carbon, {'toy_a': 100.}, 'Fixed', cfg).run()

    def test_10_cvar_fractional_tail_and_degenerate_distribution(self):
        self.assertAlmostEqual(empirical_cvar([0, 1, 2, 3], .625), 8 / 3)
        self.assertAlmostEqual(empirical_cvar([0, 1, 2, 3], .5), 2.5)
        self.assertEqual(empirical_cvar([7, 7, 7], .95), 7.)
        self.assertAlmostEqual(empirical_cvar([0, 10], .5, [.9, .1]), 2.)

    def test_11_lhs_strata_and_independent_seed_streams(self):
        draws = qmc.LatinHypercube(d=4, seed=20261018).random(16)
        for col in draws.T:
            self.assertEqual(sorted(np.floor(col * 16).astype(int)), list(range(16)))
        import hashlib
        def token(stream, rid):
            return hashlib.sha256(f'20261018|{stream}|{rid}'.encode()).hexdigest()
        self.assertEqual(token('service', 'a'), token('service', 'a'))
        self.assertNotEqual(token('service', 'a'), token('forecast', 'a'))


if __name__ == '__main__':
    began = time.perf_counter()

    from integrated_api import IntegratedRouter
    try:
        IntegratedRouter(PROJECT / 'task2/runs/experiment')
        api_diagnostic = {'status': 'LOAD_PASS', 'scope': 'constructor only'}
    except Exception as exc:
        api_diagnostic = {'status': 'BLOCKED_ENVIRONMENT', 'error': repr(exc),
                          'scope': 'full frozen IntegratedRouter constructor; no compatibility patch or retraining attempted'}
    (HERE / 'integrated_api_diagnostic.json').write_text(json.dumps(api_diagnostic, indent=2), encoding='utf-8')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Checks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'status': 'PASS' if result.wasSuccessful() else 'FAIL', 'tests_run': result.testsRun,
              'errors': len(result.errors), 'failures': len(result.failures),
              'runtime_seconds': time.perf_counter() - began,
              'scope': 'six saved Pre-model predictions and frozen_route examples; twelve-row apply replay with toy labels; toy DES and formula checks',
              'full_integrated_api_status': api_diagnostic['status'],
              'task4_performance_experiments_executed': 0, 'model_training_executed': 0,
              'future_stress_adapters_validated': False}
    (HERE / 'minimal_checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
