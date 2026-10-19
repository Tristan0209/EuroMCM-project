import unittest

import numpy as np
import pandas as pd

from evaluate import safety_gate
from resource_scenarios import simulate
from routing import baseline, execute, metrics
from risk_bounds import bernstein_upper, binomial_upper


class CompleteBoundaries(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'request_id':['a','b'],'group':['a','b'],'task_type':['General','Math'],
            'task_kind':['multiple_choice','other'],'q_s':[.25,.25],'q_l':[1.,.75],
            'cost_s':[1.,1.],'cost_l':[10.,10.],'cost_delta':[1.,1.],
            'v_s':[.1,np.nan],'v_l':[.8,np.nan], 'vc2_s':[.9,np.nan],'vc2_l':[.8,np.nan],
            'hard_s':[False,False]})

    def params(self,**kwargs):
        return {'lambda':0.,'tau':1.,'gamma':.75,'chooser':'L',**kwargs}

    def test_direct_large_does_not_need_unexecuted_small_or_peer(self):
        frame=self.frame();frame[['v_s','vc2_s','vc2_l']]=np.nan
        v=execute(frame,np.ones((1,2)),self.params(chooser='VC2'))
        np.testing.assert_array_equal(v['calls_s'],[[0,0]])
        np.testing.assert_array_equal(v['q'],[[1.,.75]])

    def test_peer_can_change_answer_only_after_both_calls(self):
        frame=self.frame()
        v=execute(frame,np.zeros((1,2)),self.params(chooser='VC2'))
        np.testing.assert_array_equal(v['calls_l'],[[1,0]])
        np.testing.assert_array_equal(v['calls_s'],[[1,1]])
        np.testing.assert_allclose(v['cost'],[[11.,1.]])

        self.assertEqual(v['q'][0,0],.25)

    def test_no_peer_needed_if_small_passes_check(self):
        frame=self.frame();frame.loc[0,'v_s']=.9;frame[['vc2_s','vc2_l']]=np.nan
        v=execute(frame,np.zeros((1,2)),self.params(chooser='VC2'))
        self.assertFalse(v['upgraded'].any())

    def test_missing_applicable_v_rejected(self):
        frame=self.frame();frame.loc[0,'v_s']=np.nan
        with self.assertRaises(ValueError):execute(frame,np.zeros((1,2)),self.params())

    def test_missing_executed_peer_rejected(self):
        frame=self.frame();frame.loc[0,'vc2_s']=np.nan
        with self.assertRaises(ValueError):execute(frame,np.zeros((1,2)),self.params(chooser='VC2'))

    def test_non_mc_diagnostics_can_trigger_without_fake_quality(self):
        frame=self.frame();frame.loc[1,'hard_s']=True
        v=execute(frame,np.zeros((1,2)),self.params())
        self.assertTrue(v['upgraded'][0,1]);self.assertEqual(v['cost'][0,1],11.)

    def test_safety_gate_and_mean_pass_are_distinct(self):
        frame=self.frame();v=baseline(frame,False)
        self.assertEqual(set(safety_gate(frame,v)),{'General','Math'})
        self.assertFalse(metrics(frame,v)['final_constraints_all_reps_ok'])

    def test_upgrade_joins_large_queue_at_actual_small_completion(self):
        frame=self.frame()
        path=pd.DataFrame({'initial_l':[False,True],'upgraded':[True,False],'calls_s':[1,0],'calls_l':[1,1]})
        result=simulate(frame,path,np.array([0.,1.]),{'S':np.ones(2),'L':np.ones(2)},
            {'S':np.ones(2)*5,'L':np.ones(2)*5},1,0.,0.,set(),set(),0.)
        np.testing.assert_allclose(result['finish'],[11.,6.])
        later=result['jobs'].query("stage == 1").iloc[0]
        self.assertEqual(later.arrival_s,5.);self.assertEqual(later.start_s,6.)

    def test_zero_events_does_not_certify_zero_rare_task_risk(self):
        alpha=.05/60
        upper=binomial_upper(0,43,alpha)
        self.assertGreater(upper,.02)
        self.assertAlmostEqual(upper,1-alpha**(1/43))
        self.assertGreater(bernstein_upper(np.zeros(43),alpha),0.)


if __name__=='__main__':unittest.main()
