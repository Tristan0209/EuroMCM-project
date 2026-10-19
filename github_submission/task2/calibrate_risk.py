import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from evaluate import feedback_pilot, paired_ci, task_metrics
from prepare import save
from routing import apply, ledger, metrics, slice_prediction

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'runs/experiment')
    args=parser.parse_args();out=args.out.resolve()
    if (out/'final_model_choice.json').exists():raise FileExistsError('Supplement already exists; do not silently append it twice')
    state=joblib.load(out/'routing_state.joblib');frame=state['frame'];predictions=state['predictions'];parameters=state['parameters']
    snapshot=out/'point_gate_initial';snapshot.mkdir(exist_ok=True)
    for name in ['quality_cost_comparison.csv','by_task.csv','safety_calibration.json','model_choice.json',
                 'selected_confidence_intervals.csv','feedback_comparison.csv']:
        shutil.copy2(out/name,snapshot/name)
    shutil.copy2(out/'routing_state.joblib',out/'routing_state_initial.joblib')
    select_mask=frame.phase.eq('select').to_numpy();select=frame[select_mask].reset_index(drop=True)
    confirm_mask=frame.phase.eq('confirm').to_numpy();confirm=frame[confirm_mask].reset_index(drop=True)
    complexity={'Pre_direct':0,'Ridge':1,'Bayesian_Ridge':2,'CART':3,'Logistic_Regression':4,
                'LightGBM':5,'Random_Forest':6,'MLP':7,'LinUCB':8,'Thompson_Sampling':9}
    rows=[];rules={};intervals={}
    for family,pred in predictions.items():
        raw=apply(confirm,slice_prediction(pred,confirm_mask),parameters[family])
        ci=paired_ci(confirm,raw,repetitions=800)
        fallback=[]
        for record in ci:
            if record['task_type']=='ALL':continue
            mask=confirm.task_type.eq(record['task_type']).to_numpy()
            point=metrics(confirm[mask],{k:v[:,mask] for k,v in raw.items()})
            if (record['quality_loss_high']>.05+1e-12 or record['pass_loss_high']>.02+1e-12
                    or point['quality_loss_worst']>.05+1e-12 or point['pass_loss_worst']>.02+1e-12):
                fallback.append(record['task_type'])
        protected=apply(confirm,slice_prediction(pred,confirm_mask),parameters[family],fallback)
        protected_ci=paired_ci(confirm,protected,repetitions=800)
        overall=next(r for r in protected_ci if r['task_type']=='ALL')
        if overall['quality_loss_high']>.02+1e-12 or overall['pass_loss_high']>.02+1e-12:
            fallback=sorted(frame.task_type.unique())
            protected=apply(confirm,slice_prediction(pred,confirm_mask),parameters[family],fallback)
            protected_ci=paired_ci(confirm,protected,repetitions=800)
        select_values=apply(select,slice_prediction(pred,select_mask),parameters[family],fallback)
        selected_stats=metrics(select,select_values)
        rows.append({'family':family,'fallback_tasks':','.join(fallback),'complexity_tie_order':complexity[family],
                     **selected_stats})
        rules[family]={'fallback_tasks':fallback,'confirm_metrics':metrics(confirm,protected)}
        intervals[family]={'raw':ci,'protected':protected_ci}
        print('RISK CALIBRATE',family,'fallback=',fallback,'selection_save=',selected_stats['cost_saving'],flush=True)
    ranking=pd.DataFrame(rows).sort_values(['model_cost_usd_mean','complexity_tie_order'])
    winner=str(ranking.iloc[0].family);fallback=rules[winner]['fallback_tasks']
    ranking.to_csv(out/'uncertainty_family_comparison.csv',index=False)
    save(out/'final_model_choice.json',{'family':winner,'selected_on':'selection_cost_after_validation_calibration_risk_gate',
        'point_selection_partition':'select','risk_calibration_partition':'confirm',
        'guard_method':'95pct paired prompt-group bootstrap interval upper endpoint; 800 fixed draws',
        'point_and_interval_tolerances':{'overall_mean':.02,'overall_pass':.02,'task_mean':.05,'task_pass':.02},
        'model_weights_or_thresholds_changed':False,'test_used_for_exact_family_or_threshold_selection':False,
        'introduced_after_initial_test_diagnosis':True,'not_an_independent_confirmation_or_simultaneous_coverage_guarantee':True})
    save(out/'final_safety_calibration.json',{'family':winner,'fallback_tasks':fallback,
        'rules_for_all_families':rules,'intervals_for_all_families':intervals,
        'uncertainty_scope':'conditional fixed policies; does not account for model selection, retraining, unseen drift or hardware error'})
    print('FINAL FROZEN WINNER',winner,'fallback=',fallback,flush=True)
    test_mask=frame.phase.eq('test').to_numpy();test=frame[test_mask].reset_index(drop=True)
    active=apply(test,slice_prediction(predictions[winner],test_mask),parameters[winner],fallback)
    row={'strategy':'Selected_uncertainty_guard','phase':'test',**metrics(test,active)}
    comparison=pd.read_csv(out/'quality_cost_comparison.csv');comparison=pd.concat([comparison,pd.DataFrame([row])],ignore_index=True)
    comparison.to_csv(out/'quality_cost_comparison.csv',index=False,encoding='utf-8-sig')
    tasks=pd.read_csv(out/'by_task.csv');tasks=pd.concat([tasks,pd.DataFrame(task_metrics(test,active,row['strategy'],'test'))],ignore_index=True)
    tasks.to_csv(out/'by_task.csv',index=False,encoding='utf-8-sig')
    detail=pd.read_csv(out/'decisions.csv.gz');detail=pd.concat([detail,ledger(test,active,row['strategy'],'test')],ignore_index=True)
    detail.to_csv(out/'decisions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(paired_ci(test,active)).to_csv(out/'final_confidence_intervals.csv',index=False)
    feedback_rows=[];feedback_logs=[];feedback_ledgers=[]
    for rep in range(3):
        for learn in [False,True]:
            values,logs,audit=feedback_pilot(test,slice_prediction(predictions[winner],test_mask),parameters[winner],fallback,learn,rep)
            name='Final_audit_residual_update' if learn else 'Final_audit_frozen_control'
            feedback_rows.append({'strategy':name,'rep':rep,'audit_requests':int(audit.sum()),**metrics(test,values)})
            feedback_logs.extend([{**log,'protocol':'uncertainty_guard_supplement'} for log in logs])
            paths=ledger(test,values,name,'feedback');paths['pilot_rep']=rep;paths['audited']=audit;feedback_ledgers.append(paths)
    old_feedback=pd.read_csv(out/'feedback_comparison.csv')
    pd.concat([old_feedback,pd.DataFrame(feedback_rows)],ignore_index=True).to_csv(out/'feedback_comparison.csv',index=False)
    old_logs=json.loads((out/'feedback_audit.json').read_text(encoding='utf-8'))
    save(out/'feedback_audit.json',old_logs+feedback_logs)
    old_paths=pd.read_csv(out/'feedback_decisions.csv.gz')
    pd.concat([old_paths,*feedback_ledgers],ignore_index=True).to_csv(out/'feedback_decisions.csv.gz',index=False,compression='gzip')
    state.update({'initial_winner':state['winner'],'initial_fallback_tasks':state['fallback_tasks'],
                  'winner':winner,'fallback_tasks':fallback,'main_strategy':'Selected_uncertainty_guard'})
    joblib.dump(state,out/'routing_state.joblib',compress=3)
    print(json.dumps(row,indent=2),flush=True)


if __name__=='__main__':main()
