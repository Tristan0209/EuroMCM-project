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
from dataclasses import asdict
from task4_core.analysis import formal_mixture_risk,lhs_screen,rebound_rows
from task4_core.protocol import lhs_points,LHS_NAMES
from task4_core.serialization import save

OUT = None


class AnalysisContracts(unittest.TestCase):
    def test_full_size_synthetic_cvar_adapter_preserves_translation_and_tail_support(self):
        rows,regimes = [],{}
        for j in range(2000):
            wid = 'SYNTHETIC_NOT_DES_'+str(j)
            regimes[wid] = int(j>=1000)
            for policy,shift in [('Greedy',0.),('Spatial_MILP',.002)]:
                rows.append({'world_id':wid,'policy':policy,'service_violation_rate':j/200000+shift,
                    'model_selection_harm':j/200000+shift,'delivery_incremental_harm':j/200000+shift,
                    'end_to_end_harm':j/100000+shift,'co2e_g':1000+j/100+shift*1000})
        result = formal_mixture_risk(pd.DataFrame(rows),regimes)
        for p,metrics in result['results'].items():
            for field in ['service_violation_rate','model_selection_harm','delivery_incremental_harm','end_to_end_harm']:
                self.assertAlmostEqual(metrics[field]['proposed_minus_baseline'],.002,places=10)
                np.testing.assert_allclose(metrics[field]['ci95_difference'],[.002,.002],atol=1e-10)
                self.assertGreaterEqual(metrics[field]['proposed']['tail_ess'],50-1e-8)
            self.assertAlmostEqual(metrics['co2e_g']['proposed_minus_baseline'],2.)
            self.assertAlmostEqual(sum(metrics['service_violation_rate']['proposed_tail_weights'].values()),1.)
        save(OUT/'synthetic_cvar_summary.json',{'status':'PASS','synthetic_rows':4000,'DES_worlds_executed':0,
            'bootstrap':2000,'probabilities':[0,.1,.2,.4],'known_quality_translation':.002,'known_carbon_translation_g':2.,
            'system_risk_claim':False})

    def test_lhs_screen_recovers_predeclared_dominant_parameter(self):
        rows = []
        for scramble in [0,1]:
            _,points = lhs_points(seed=20311018+scramble)
            for j,s in enumerate(points):
                for seed in [1,2,3]:
                    rows.append({**{p:asdict(s)[p] for p in LHS_NAMES},'scramble':scramble,'parameter_point':j,
                        'seed':seed,'scheduler_carbon_saving_g':s.shared_w+seed*.00001})
        frame = pd.DataFrame(rows)
        table,review = lhs_screen(frame)
        for scramble in [0,1]:
            shared = table[(table.scramble==scramble)&table.parameter.eq('shared_w')]
            self.assertAlmostEqual(shared.spearman_descriptive.iloc[0],1.)
        self.assertFalse(review['PRCC_significance_claim'])
        with self.assertRaises(ValueError):
            lhs_screen(pd.concat([frame,frame.iloc[:1]],ignore_index=True))

    def test_rebound_rejects_mismatched_original_demand(self):
        base = dict(policy='Greedy',growth=0.,l_hosts=1,seed=42,base_n=800,co2e_g=100.,
                    world_id='synthetic_base',not_completed_count=0,effective_delivered_quality_per_input=.8)
        grown = {**base,'policy':'Spatial_MILP','growth':.25,'co2e_g':101.,'world_id':'synthetic_growth'}
        output = rebound_rows(pd.DataFrame([base,grown]))
        self.assertEqual(output.fixed_demand_total_carbon_change_g.tolist(),[1.])
        self.assertIn('exogenous',output.interpretation.iloc[0])
        with self.assertRaises(ValueError):
            rebound_rows(pd.DataFrame([base,{**grown,'base_n':500}]))


if __name__=='__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out',type=Path,required=True)
    args = p.parse_args()
    OUT = args.out.resolve()
    if ROOT not in OUT.parents or OUT.exists():
        raise ValueError('Use a new analysis contract output directory inside TASK 4')
    OUT.mkdir(parents=True)
    start = time.perf_counter()
    with (OUT/'analysis_contracts.log').open('w',encoding='utf-8') as log:
        result = unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AnalysisContracts))
    report = {'status':'PASS' if result.wasSuccessful() else 'FAIL','tests_run':result.testsRun,
              'failures':len(result.failures),'errors':len(result.errors),'wall_s':time.perf_counter()-start,
              'DES_worlds_executed':0,'all_examples_are_synthetic':True,'formal_M1_M8_executions':0}
    save(OUT/'results.json',report)
    print(report)
    sys.exit(0 if result.wasSuccessful() else 1)
