import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prepare import save
from routing import SEED, apply, execute, ledger, metrics, slice_prediction
from evaluate import task_metrics

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'runs/experiment')
    args=parser.parse_args();out=args.out.resolve()
    state=joblib.load(out/'routing_state.joblib');frame=state['frame'];winner=state['winner']
    selected=state['parameters'][winner];prediction=state['predictions'][winner];fallback=state['fallback_tasks']
    sm=frame.phase.eq('select').to_numpy();tm=frame.phase.eq('test').to_numpy()
    selection=frame[sm].reset_index(drop=True);test=frame[tm].reset_index(drop=True)
    table=pd.read_csv(out/'quality_cost_comparison.csv')
    if not table.strategy.eq('Final_random_matched').any():
        chosen=apply(selection,slice_prediction(prediction,sm),selected,fallback)
        budget=float(chosen['calls_l'].mean());all_values={}
        for rep in range(10):
            initial=np.random.default_rng(SEED+400+rep).random(len(test))<budget
            values=execute(test,initial.astype(float)[None],{'lambda':0.,'tau':.5,'gamma':None,'chooser':'L'})
            for k,v in values.items():all_values.setdefault(k,[]).append(v)
        values={k:np.concatenate(v,axis=0) for k,v in all_values.items()}
        row={'strategy':'Final_random_matched','phase':'test',**metrics(test,values)}
        pd.concat([table,pd.DataFrame([row])],ignore_index=True).to_csv(out/'quality_cost_comparison.csv',index=False,encoding='utf-8-sig')
        tasks=pd.read_csv(out/'by_task.csv')
        pd.concat([tasks,pd.DataFrame(task_metrics(test,values,'Final_random_matched','test'))],ignore_index=True).to_csv(out/'by_task.csv',index=False,encoding='utf-8-sig')
        paths=pd.read_csv(out/'decisions.csv.gz')
        pd.concat([paths,ledger(test,values,'Final_random_matched','test')],ignore_index=True).to_csv(out/'decisions.csv.gz',index=False,compression='gzip')
        save(out/'final_random_budget.json',{'selection_L_call_budget':budget,'repetitions':10,
             'matched_quantity':'selection L call rate, not exact realized USD','not_best_seed_selection':True})

    for path in out.glob('*.json'):
        save(path,json.loads(path.read_text(encoding='utf-8')))
    save(out/'completion.json',{'completed_stages':['Work1 adapters','nested OOF Pre','OOF V/VC2',
          '10 route families','point gate initial trial','bootstrap diagnostic trial','finite-sample conservative supplement'],
        'final_family':winner,'final_strategy':state['main_strategy'],'fallback_tasks':fallback,
        'initial_point_gate_test_failure_retained':True,'bootstrap_gate_test_failure_retained':True,
        'final_method_is_post_diagnosis_research':True,'new_independent_holdout_still_needed':True,
        'no_synthetic_quality_observations':True,'extra_thinking_quality_effect_not_invented':True,
        'active_online_feedback':False,'actual_net_cost_latency_energy_measured':False})
    print('FINAL ARTIFACTS',winner,flush=True)


if __name__=='__main__':main()
