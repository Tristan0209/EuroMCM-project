import argparse
import json
import shutil
from pathlib import Path

import joblib
import pandas as pd

from evaluate import feedback_pilot, paired_ci, task_metrics
from prepare import save
from risk_bounds import allowed_fallbacks, task_bounds
from routing import apply, ledger, metrics, slice_prediction

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'runs/experiment')
    args=parser.parse_args();out=args.out.resolve()
    if (out/'finite_sample_calibration.json').exists():raise FileExistsError('Final calibration already saved')
    state=joblib.load(out/'routing_state.joblib');frame=state['frame'];predictions=state['predictions'];parameters=state['parameters']
    snapshot=out/'bootstrap_gate_initial';snapshot.mkdir(exist_ok=True)
    for name in ['final_model_choice.json','final_safety_calibration.json','final_confidence_intervals.csv']:
        shutil.copy2(out/name,snapshot/name)
    shutil.copy2(out/'routing_state.joblib',out/'routing_state_bootstrap.joblib')
    select_mask=frame.phase.eq('select').to_numpy();select=frame[select_mask].reset_index(drop=True)
    confirm_mask=frame.phase.eq('confirm').to_numpy();confirm=frame[confirm_mask].reset_index(drop=True)
    tasks=sorted(frame.task_type.unique());mixtures=[];duplicates={t:0. for t in tasks}
    for phase in ['select','confirm','test']:
        sub=frame[frame.phase.eq(phase)];mixtures.append(sub.task_type.value_counts(normalize=True).to_dict())
        for t,part in sub.groupby('task_type'):
            duplicates[t]=max(duplicates[t],1-part.group.nunique()/len(part))

    alpha=.05/(len(predictions)*len(tasks)*2)
    complexity={'Pre_direct':0,'Ridge':1,'Bayesian_Ridge':2,'CART':3,'Logistic_Regression':4,
                'LightGBM':5,'Random_Forest':6,'MLP':7,'LinUCB':8,'Thompson_Sampling':9}
    rows=[];rules={};bounds_by_family={}
    for family,pred in predictions.items():
        raw=apply(confirm,slice_prediction(pred,confirm_mask),parameters[family])
        bounds=task_bounds(confirm,raw,alpha,duplicates);candidates=[]
        for rule in allowed_fallbacks(bounds,mixtures):
            values=apply(select,slice_prediction(pred,select_mask),parameters[family],rule['fallback_tasks'])
            candidates.append({'rule':rule,'metrics':metrics(select,values)})
        candidate=min(candidates,key=lambda r:(r['metrics']['model_cost_usd_mean'],r['metrics']['large_call_rate']))
        rules[family]=candidate['rule'];bounds_by_family[family]=bounds
        rows.append({'family':family,'fallback_tasks':','.join(candidate['rule']['fallback_tasks']),
                     'complexity_tie_order':complexity[family],**candidate['metrics'],
                     'aggregate_mean_upper':candidate['rule']['aggregate_mean_upper'],
                     'aggregate_pass_upper':candidate['rule']['aggregate_pass_upper']})
        print('FINITE RISK',family,candidate['rule'],'select_save=',candidate['metrics']['cost_saving'],flush=True)
    ranking=pd.DataFrame(rows).sort_values(['model_cost_usd_mean','complexity_tie_order'])
    winner=str(ranking.iloc[0].family);fallback=rules[winner]['fallback_tasks']
    ranking.to_csv(out/'finite_sample_family_comparison.csv',index=False)
    metadata={'family':winner,'fallback_tasks':fallback,'overall_alpha':.05,'individual_alpha':alpha,
        'number_fixed_families':len(predictions),'number_tasks':len(tasks),'number_losses':2,
        'bounds':'Clopper-Pearson upper harm probability; empirical Bernstein upper positive Score loss',
        'family_selection':'minimum selection-partition model cost after uniform calibration bounds and explicit all-L fallbacks',
        'zero_event_risk_not_zero':True,'known_task_mixtures':mixtures,'duplicate_mass_limits':duplicates,
        'model_weights_and_thresholds_unchanged':True,'introduced_after_prior_old_test_diagnostics':True,
        'test_scores_used_for_exact_selection':False,
        'assumptions':['fixed learned policies conditional on training/selection',
            'independent exchangeable normalized prompt groups within each task',
            'future task mixture and duplicate mass bounded by declared scenario',
            'original benchmark Score is the evaluation target, not real correctness'],
        'not_a_production_quality_guarantee':True,'earlier_failures_preserved':True}
    save(out/'finite_sample_calibration.json',{**metadata,'rules':rules,'bounds':bounds_by_family})
    save(out/'final_model_choice.json',metadata)
    save(out/'final_safety_calibration.json',{'family':winner,'fallback_tasks':fallback,'method':metadata['bounds'],
         'finite_sample_detail':'finite_sample_calibration.json','not_a_production_quality_guarantee':True})
    print('FINAL FINITE MODEL',winner,'fallback=',fallback,flush=True)
    test_mask=frame.phase.eq('test').to_numpy();test=frame[test_mask].reset_index(drop=True)
    all_rows=[];all_tasks=[];all_paths=[]
    for family,pred in predictions.items():
        values=apply(test,slice_prediction(pred,test_mask),parameters[family],rules[family]['fallback_tasks'])
        label='Safe_'+family
        all_rows.append({'strategy':label,'phase':'test',**metrics(test,values)})
        all_tasks.extend(task_metrics(test,values,label,'test'));all_paths.append(ledger(test,values,label,'test'))
    active=apply(test,slice_prediction(predictions[winner],test_mask),parameters[winner],fallback)
    main_label='Selected_finite_sample_guard'
    all_rows.append({'strategy':main_label,'phase':'test',**metrics(test,active)})
    all_tasks.extend(task_metrics(test,active,main_label,'test'));all_paths.append(ledger(test,active,main_label,'test'))

    import copy
    for label,disable in [('Final_no_Post','post'),('Final_no_VC2','peer')]:
        altered=copy.deepcopy(parameters[winner])
        for setting in altered.values():
            if disable=='post':setting['parameters']['gamma']=None
            else:setting['parameters']['chooser']='L'
        values=apply(test,slice_prediction(predictions[winner],test_mask),altered,fallback)
        all_rows.append({'strategy':label,'phase':'test',**metrics(test,values)})
        all_tasks.extend(task_metrics(test,values,label,'test'));all_paths.append(ledger(test,values,label,'test'))
    table=pd.read_csv(out/'quality_cost_comparison.csv')
    pd.concat([table,pd.DataFrame(all_rows)],ignore_index=True).to_csv(out/'quality_cost_comparison.csv',index=False,encoding='utf-8-sig')
    table=pd.read_csv(out/'by_task.csv')
    pd.concat([table,pd.DataFrame(all_tasks)],ignore_index=True).to_csv(out/'by_task.csv',index=False,encoding='utf-8-sig')
    table=pd.read_csv(out/'decisions.csv.gz')
    pd.concat([table,*all_paths],ignore_index=True).to_csv(out/'decisions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(paired_ci(test,active)).to_csv(out/'final_confidence_intervals.csv',index=False)
    feedback_rows=[];feedback_logs=[];feedback_paths=[]
    for rep in range(3):
        for learn in [False,True]:
            values,logs,audit=feedback_pilot(test,slice_prediction(predictions[winner],test_mask),parameters[winner],fallback,learn,rep)
            name='Finite_audit_residual_update' if learn else 'Finite_audit_frozen_control'
            feedback_rows.append({'strategy':name,'rep':rep,'audit_requests':int(audit.sum()),**metrics(test,values)})
            feedback_logs.extend([{**log,'protocol':'finite_sample_guard'} for log in logs])
            paths=ledger(test,values,name,'feedback');paths['pilot_rep']=rep;paths['audited']=audit;feedback_paths.append(paths)
    table=pd.read_csv(out/'feedback_comparison.csv')
    pd.concat([table,pd.DataFrame(feedback_rows)],ignore_index=True).to_csv(out/'feedback_comparison.csv',index=False)
    logs=json.loads((out/'feedback_audit.json').read_text(encoding='utf-8'));save(out/'feedback_audit.json',logs+feedback_logs)
    table=pd.read_csv(out/'feedback_decisions.csv.gz')
    pd.concat([table,*feedback_paths],ignore_index=True).to_csv(out/'feedback_decisions.csv.gz',index=False,compression='gzip')
    state.update({'bootstrap_winner':state['winner'],'bootstrap_fallback_tasks':state['fallback_tasks'],
                  'winner':winner,'fallback_tasks':fallback,'main_strategy':main_label})
    joblib.dump(state,out/'routing_state.joblib',compress=3)
    print(json.dumps(metrics(test,active),indent=2),flush=True)


if __name__=='__main__':main()
