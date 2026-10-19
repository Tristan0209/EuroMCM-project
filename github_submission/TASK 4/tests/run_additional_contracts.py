import argparse
from pathlib import Path
import sys
import time
import unittest
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from task4_core.world import Scenario,ScenarioBuilder
from task4_core.runner import Task4Runner
from task4_core.serialization import digest,save
from task4_core.sources import read
from task4_core.protocol import conditional_mc

OUT = None


class AdditionalContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.builder = ScenarioBuilder()

    def test_actual_generator_correlations_and_conditional_marginals(self):
        world = self.builder.build(Scenario(sigma_q=.1,sigma_carbon=.4,sigma_service=.5),2090101800)
        groups = world.pre_errors.drop_duplicates('prompt_group')
        pre_rho = float(groups[['error_s','error_l']].corr().iloc[0,1])
        self.assertAlmostEqual(pre_rho,.5,delta=.06)
        self.assertAlmostEqual(float(np.mean(list(world.service_shocks.values()))),1.,delta=.025)
        fields = []
        for seed in range(2090101900,2090101964):
            small = self.builder.build(Scenario(n=1,sigma_carbon=.4),seed)
            fields.append(np.log(small.forecast_field[['DE','FR','ES']].to_numpy()))
        array = np.concatenate(fields)
        carbon_rho = np.corrcoef(array,rowvar=False)
        for j,k in [(0,1),(0,2),(1,2)]:
            self.assertAlmostEqual(carbon_rho[j,k],.5,delta=.08)
        self.assertTrue(np.all(np.abs(np.exp(array).mean(axis=0)-1)<.06))
        parameter_diagnostics = []
        for regime in [0,1]:
            draws = [conditional_mc(regime,2091101800+j) for j in range(2000)]
            fault = [d for d in draws if d.outage!='none']
            rate = len(fault)/len(draws)
            self.assertAlmostEqual(rate,.05 if regime==0 else .5,delta=.025)
            burst = np.mean([d.burst_factor>1 for d in draws])
            self.assertAlmostEqual(float(burst),.1 if regime==0 else .5,delta=.03)
            if regime==1:
                common = np.mean([d.outage=='all_L' for d in fault])
                self.assertAlmostEqual(float(common),.25,delta=.04)
            self.assertTrue(all((0<=d.sigma_q<=.05 if regime==0 else .05<=d.sigma_q<=.1) for d in draws))
            self.assertTrue(all((0<=d.sigma_carbon<=.15 if regime==0 else .15<=d.sigma_carbon<=.4) for d in draws))
            self.assertTrue(all((.1<=d.sigma_service<=.25 if regime==0 else .3<=d.sigma_service<=.5) for d in draws))
            parameter_diagnostics.append({'regime':regime,'parameter_draws_only':len(draws),'outage_rate':rate,'burst_rate':float(burst)})
        save(OUT/'distribution_checks.json',{'status':'PASS','DES_worlds_for_distribution_test':0,
            'pre_group_correlation':pre_rho,'pre_unique_groups':len(groups),'carbon_correlation':carbon_rho,
            'conditional_parameter_checks':parameter_diagnostics,
            'interpretation':'fixed-seed software draw checks, not fitted occurrence probabilities'})

    def test_dispatch_exactly_H_retains_network_and_zero_busy(self):
        world = self.builder.build(Scenario(n=1),2092101800)
        world.instances.loc[:,'arrival_s'] = 43200-.01
        world.instances.loc[:,'budget_s'] = 100.
        world.world_id = digest(world.payload())
        record = Task4Runner(OUT/'endpoint_runner').execute(world,'Greedy')
        self.assertEqual(record['status'],'COMPLETE_VALIDATED')
        folder = Path(record['folder'])
        summary = read(folder/'metrics.json')
        records = pd.read_csv(folder/'requests.csv.gz')
        self.assertEqual(records.dispatch_s.tolist(),[43200.])
        self.assertEqual(summary['censored'],1)
        self.assertEqual(summary['unfinished_stages'],{'inflight':1})
        self.assertEqual(summary['active_increment_kwh'],0.)
        self.assertGreater(summary['network_kwh'],0.)
        self.assertAlmostEqual(summary['controller_consumed_work_s'],.01,places=8)
        save(OUT/'H_endpoint.json',{'status':'PASS','dispatch_s':43200.,'network_kwh':summary['network_kwh'],
             'active_increment_kwh':summary['active_increment_kwh'],'definition':'events at H processed; continuous power intersection [0,H)'})


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    OUT = args.out.resolve()
    if ROOT not in OUT.parents or OUT.exists():
        raise ValueError('New contract output directory required')
    OUT.mkdir(parents=True)
    start = time.perf_counter()
    with (OUT/'additional_contracts.log').open('w',encoding='utf-8') as stream:
        result = unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AdditionalContracts))
    report = {'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
              'failures':len(result.failures),'errors':len(result.errors),'wall_s':time.perf_counter()-start,
              'formal_experiments':0,'scope':'distribution-only draws and one synthetic endpoint DES contract; not new pilot samples'}
    save(OUT/'results.json',report)
    print(report)
    sys.exit(0 if result.wasSuccessful() else 1)
