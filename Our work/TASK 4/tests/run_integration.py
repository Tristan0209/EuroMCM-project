import argparse
from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(Path(__file__).resolve().parent)]
import numpy as np
import pandas as pd
from task4_core.sources import BASE, read, sha
from task4_core.serialization import digest, save
from task4_core.world import Scenario, ScenarioBuilder, ExogenousWorld, FieldForecast
from task4_core.router import PreDirectRouter
from task4_core.window import WindowedGreedy, WindowedSpatial, ENGINE, SPATIAL
from task4_core.runner import Task4Runner, frozen_config
from task4_core.validation import validate_execution
from task4_core.protocol import core_plan, conditional_mc, lhs_points, joint
from task4_core.resources import pools_from_hosts
from task4_core.analysis import paired_rows, formal_mixture_risk
from src.domain import load_inputs
from src.accounting import evaluate as frozen_evaluate
from task4_core.accounting import evaluate_window
from run_contracts import check_equal

OUT = None


def run_world(world, router, policy):
    jobs, route = world.routed_requests(router)
    servers,net,availability,carbon,shocks,forecast = world.environment()
    cls = WindowedSpatial if policy=='Spatial_MILP' else WindowedGreedy
    sim = cls(jobs,servers,net,availability,carbon,shocks,cfg=frozen_config(world),forecast_factory=forecast).run()
    return sim,evaluate_window(sim),route


class IntegrationContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = ScenarioBuilder()
        cls.router = PreDirectRouter()
        cls.before = cls.builder.source_hashes
        save(OUT/'source_hashes_before.json',cls.before)

    @classmethod
    def tearDownClass(cls):
        changed = [p for p,h in cls.before.items() if sha(p)!=h]
        save(OUT/'source_protection.json',{'status':'PASS' if not changed else 'FAIL','files':len(cls.before),'changed':changed})
        if changed:
            raise AssertionError(changed)

    def test_A_full_7335_through_new_world_pipeline(self):
        world = self.builder.build(Scenario(frozen_shock_seed=2027101800),2027101800)
        jobs,route = world.routed_requests(self.router)
        original,servers,net,anchor = load_inputs()
        expected = {q.request_id:q for q in original}
        for q in jobs:
            self.assertEqual(q.model,expected[q.request_id].model)
            self.assertAlmostEqual(q.service,expected[q.request_id].service,places=9)
        self.assertEqual(route.selected_model.value_counts().to_dict(),{'L':6904,'S':431})
        route.to_csv(OUT/'full_7335_pipeline_routing.csv.gz',index=False,compression='gzip')
        details = []
        for policy,cls in [('Greedy',ENGINE.Simulation),('Spatial_MILP',SPATIAL.SpatialSimulation)]:
            _,_,availability,carbon,shocks,_ = world.environment()
            old = cls(original,servers,net,availability,carbon,shocks,'Greedy' if policy=='Greedy' else 'MPC',cfg=frozen_config(world)).run()
            new,result,_ = run_world(world,self.router,policy)

            check_equal(old,new,frozen_evaluate(old,43200),result)
            details.append({'policy':policy,'requests':7335,'events':len(new.log),'status':'PASS'})
        save(OUT/'full_7335_pipeline_equivalence.json',{'status':'PASS','seed':2027101800,'comparisons':details})

    def test_B_pre_changes_fallback_service_energy_quality_and_truth_boundary(self):
        normal = self.builder.build(Scenario(n=250),2028101990)
        shifted = self.builder.build(Scenario(n=250,gain_bias=-1.,sigma_q=.1),2028101990)
        jobs,a = normal.routed_requests(self.router)
        new,b = shifted.routed_requests(self.router)
        self.assertGreater(int(a.selected_model.ne(b.selected_model).sum()),0)
        self.assertTrue(b.loc[b.task_type.isin(['Code','Math']),'selected_model'].eq('L').all())
        mapping = shifted.instances.set_index('instance_id')
        for q in new:
            self.assertEqual(q.service,mapping.loc[q.request_id,'service_'+q.model.lower()])
        runner = Task4Runner(OUT/'pre_runner',self.router)
        record = runner.execute(shifted,'Greedy')
        self.assertEqual(record['status'],'COMPLETE_VALIDATED')
        validate_execution(Path(record['folder']),shifted)
        frame = pd.read_csv(Path(record['folder'])/'quality_harms.csv.gz')
        quality = runner.quality_after_simulation(shifted).set_index('request_id')
        self.assertTrue(frame.q_s.eq(frame.request_id.map(quality.q_s)).all())
        self.assertTrue(frame.q_l.eq(frame.request_id.map(quality.q_l)).all())
        poisoned = shifted.instances.rename(columns={'instance_id':'request_id'}).copy()
        errors = shifted.pre_errors
        first = self.router.route_frame(poisoned,errors.error_s,errors.error_l)
        poisoned['quality_true'],poisoned['response'],poisoned['output_tokens_proxy'] = -999,'truth poison',10**9
        second = self.router.route_frame(poisoned,errors.error_s,errors.error_l)
        pd.testing.assert_frame_equal(first,second)

    def test_C_four_joint_worlds_persisted_conservation(self):
        cases = [Scenario(n=160,compression=2,outage='FR_L'),
                 Scenario(n=160,hard_fraction=.6,gain_bias=-.1,sigma_q=.1),
                 Scenario(n=160,sigma_carbon=.4,carbon_bias_fr=.4,carbon_bias_de=-.4,
                          l_hosts=2,outage='FR_L',reduced_fraction=.5,outage_duration=10800),
                 Scenario(n=160,outage='all_L',sigma_service=.5)]
        runner = Task4Runner(OUT/'joint_runner',self.router)
        report = []
        for j,s in enumerate(cases):
            world = self.builder.build(s,2028101991+j)
            shared = world.world_id
            for policy in ['Greedy','Spatial_MILP']:
                record = runner.execute(world,policy)
                self.assertEqual(record['status'],'COMPLETE_VALIDATED')
                self.assertEqual(shared,record['world_id'])
                report.append({'case':j,'policy':policy,**validate_execution(Path(record['folder']),world)})
        save(OUT/'joint_contracts.json',report)

    def test_D_world_roundtrip_pairing_policy_order_and_reproducibility(self):
        params = Scenario(n=90,hard_fraction=.6,sigma_q=.1,sigma_carbon=.4,compression=2,outage='all_L')
        a = self.builder.build(params,2028101995)
        b = self.builder.build(params,2028101995)
        c = self.builder.build(params,2028101996)
        self.assertEqual(a.world_id,b.world_id)
        self.assertNotEqual(a.world_id,c.world_id)
        a.save(OUT/'roundtrip_world')
        restored = ExogenousWorld.load(OUT/'roundtrip_world')
        self.assertEqual(restored.world_id,a.world_id)
        first,second = {},{}
        for policy in ['Greedy','Spatial_MILP']:
            sim,result,_ = run_world(a,self.router,policy)
            first[policy] = (sim.log,result[1])
        for policy in ['Spatial_MILP','Greedy']:
            sim,result,_ = run_world(b,self.router,policy)
            second[policy] = (sim.log,result[1])
        for policy in first:
            self.assertEqual(first[policy][0],second[policy][0])
            pd.testing.assert_frame_equal(first[policy][1],second[policy][1],check_exact=True)
        a.verify()

    def test_D_past_only_grid_field_future_prefix_and_hidden_shocks(self):
        world = self.builder.build(Scenario(n=40,sigma_carbon=.4,outage='all_L',outage_start=10000),2028101997)
        pools,net,av,carbon,shocks,factory = world.environment()
        one,two = factory(carbon,1.,10800),factory(carbon,100.,10800)
        self.assertEqual(one.grid_hour,two.grid_hour)
        self.assertEqual(one.at(200.,'FR'),two.at(200.,'FR'))
        prefix = world.instances[world.instances.arrival_s<3600].instance_id.tolist()
        if not prefix:
            self.fail('Prefix fixture empty')
        base,_,_ = run_world(world,self.router,'Spatial_MILP')
        carbon.values[carbon.index(3600.):] *= 10
        jobs,route = world.routed_requests(self.router)
        av.events += [(20000.,'DE','L',0)]
        class Hidden(dict):
            def __getitem__(self,key):
                if inspect.currentframe().f_back.f_code.co_name != '_admit':
                    raise AssertionError('Hidden service read outside admission')
                return super().__getitem__(key)
        second = WindowedSpatial(jobs,pools,net,av,carbon,Hidden(shocks),cfg=frozen_config(world),forecast_factory=factory).run()
        logs = lambda sim:[r for r in sim.log if r['time_s']<3600]
        self.assertEqual(logs(base),logs(second))
        self.assertTrue(all(x['observed_carbon_end_s']<=x['decision_s'] for x in second.solver_log))

    def test_E_growth_nested_and_slots_vs_physical_hosts(self):
        original = self.builder.build(Scenario(n=100),2028101998)
        a = self.builder.build(Scenario(n=100,growth=.5),2028101998)
        b = self.builder.build(Scenario(n=100,growth=1),2028101998)
        self.assertEqual((len(a.instances),len(b.instances)),(150,200))
        for world in [a,b]:
            original_part = world.instances.set_index('instance_id').loc[original.instances.instance_id]
            pd.testing.assert_frame_equal(original_part.reset_index(),original.instances,check_exact=True)
        ids = set(a.instances.instance_id)
        self.assertTrue(ids <= set(b.instances.instance_id))
        self.assertEqual({i:a.service_shocks[i] for i in ids},{i:b.service_shocks[i] for i in ids})
        slots = self.builder.build(Scenario(n=100,slots_per_host=2),2028101998)
        hosts = self.builder.build(Scenario(n=100,l_hosts=2),2028101998)
        idle = lambda w:sum((h['shared_w']+h['gpus']*h['gpu_idle_w'])*h['pue']*43200/3.6e6 for h in w.hosts)
        self.assertEqual(idle(slots),idle(original))
        self.assertAlmostEqual(idle(hosts),idle(original)*1.5)
        self.assertEqual(sum(p['capacity'] for p in pools_from_hosts(hosts.hosts).values()),9)

    def test_E_censored_has_partial_energy_and_nonzero_quality_loss(self):
        world = self.builder.build(Scenario(n=1),2028101999)
        world.instances.loc[:,'arrival_s'] = 43199.
        world.instances.loc[:,'service_s'] = 1.
        world.instances.loc[:,'service_l'] = 1.
        world.instances.loc[:,'budget_s'] = 100.
        world.service_shocks = {i:10. for i in world.instances.instance_id}
        world.world_id = digest(world.payload())
        runner = Task4Runner(OUT/'censor_runner',self.router)
        record = runner.execute(world,'Greedy')
        self.assertEqual(record['status'],'COMPLETE_VALIDATED')
        summary = read(Path(record['folder'])/'metrics.json')
        self.assertEqual(summary['censored'],1)
        self.assertEqual(summary['service_violation_rate'],1)
        self.assertGreater(summary['active_increment_kwh'],0)
        self.assertGreater(summary['remaining_running_actual_s_evaluation_only'],0)
        self.assertAlmostEqual(summary['idle_kwh'],250.56)

    def test_F_reuse_failed_attempt_resume_and_artifact_mutation(self):
        world = self.builder.build(Scenario(n=40),2028102000)
        runner = Task4Runner(OUT/'reuse_runner',self.router)
        with patch('task4_core.runner.evaluate_window',side_effect=RuntimeError('intentional contract failure')):
            failed = runner.execute(world,'Spatial_MILP')
        self.assertEqual(failed['status'],'FAILED')
        self.assertTrue((Path(failed['folder'])/'exception.json').is_file())
        good = runner.execute(world,'Spatial_MILP')
        self.assertEqual(good['status'],'COMPLETE_VALIDATED')
        reused = runner.execute(world,'Spatial_MILP')
        self.assertTrue(reused['reused'])
        self.assertEqual(good['folder'],reused['folder'])
        path = Path(good['folder'])/'metrics.json'
        path.write_text(path.read_text(encoding='utf-8')+'\n',encoding='utf-8')
        fresh = runner.execute(world,'Spatial_MILP')
        self.assertFalse(fresh['reused'])
        self.assertNotEqual(fresh['folder'],good['folder'])
        self.assertTrue((Path(failed['folder'])/'exception.json').exists())

    def test_G_protocol_counts_mc_ranges_and_lhs_stratification(self):
        plan = core_plan()
        counts = pd.Series([r['module'] for r in plan]).value_counts().to_dict()
        self.assertEqual(counts,{'C3':4000,'C2':192,'C6':192,'C4':120,'C1':48,'C8':6})
        for regime in [0,1]:
            draws = [conditional_mc(regime,2070000000+j) for j in range(500)]
            self.assertTrue(all((.8<=d.compression<=1.2 if regime==0 else 1.2<=d.compression<=2) for d in draws))
            self.assertTrue(all((1796/7335<=d.hard_fraction<=.4 if regime==0 else .4<=d.hard_fraction<=.6) for d in draws))
        unit,points = lhs_points()
        for j in range(15):
            self.assertEqual(sorted(np.floor(unit[:,j]*16).astype(int)),list(range(16)))
        for s in points:
            pools_from_hosts(self.builder.build(replace(s,n=5),2080000000).hosts)
        with self.assertRaises(ValueError):
            formal_mixture_risk(pd.DataFrame(),{})
        with self.assertRaises(PermissionError):
            Task4Runner(OUT/'gate_runner',self.router).execute(self.builder.build(Scenario(n=1001),2080000001),'Greedy')

    def test_H_arrival_count_support_burst_and_group_errors(self):
        normal = self.builder.build(Scenario(n=300),2028102001)
        compressed = self.builder.build(Scenario(n=300,compression=2),2028102001)
        burst = self.builder.build(Scenario(n=300,burst_factor=5),2028102001)
        self.assertEqual(len(normal.instances),len(compressed.instances))
        np.testing.assert_allclose(normal.instances.arrival_s,compressed.instances.arrival_s*2)
        self.assertEqual(len(normal.instances),len(burst.instances))
        self.assertGreater(sum(burst.instances.arrival_s.between(3600,4500)),sum(normal.instances.arrival_s.between(3600,4500)))
        duplicate = self.builder.build(Scenario(n=200,growth=1,sigma_q=.1),2028102001)
        self.assertTrue((duplicate.pre_errors.groupby('prompt_group')[['error_s','error_l']].nunique() == 1).all().all())
        z = np.array([float(np.exp(-.5*.5**2+.5*np.random.default_rng(j).normal())) for j in range(20000)])
        self.assertAlmostEqual(float(z.mean()),1.,delta=.015)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    OUT = args.out.resolve()
    if ROOT not in OUT.parents or OUT.exists():
        raise ValueError('New integration attempt directory inside TASK 4 required')
    OUT.mkdir(parents=True)
    import queue_milp
    queue_milp.ROOT = OUT/'solver_evidence'
    start = time.perf_counter()
    with (OUT/'integration.log').open('w',encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(IntegrationContracts))
    report = {'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
              'failures':len(result.failures),'errors':len(result.errors),'wall_s':time.perf_counter()-start,
              'formal_M1_M8_executions':0,'scope':'engineering contracts; injected failures retained; no performance certification'}
    save(OUT/'integration_results.json',report)
    print(json.dumps(report,indent=2))
    sys.exit(0 if result.wasSuccessful() else 1)
