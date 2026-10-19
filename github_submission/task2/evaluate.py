import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from q1_adapter import digest
from prepare import save
from routing import (SEED, apply, baseline, fit_cost, ledger, metrics,
                     slice_prediction, tune)

HERE=Path(__file__).resolve().parent


def task_metrics(frame, values, strategy, phase):
    rows=[]
    for task in sorted(frame.task_type.unique()):
        mask=frame.task_type.eq(task).to_numpy()
        rows.append({'strategy':strategy,'phase':phase,'task_type':task,
                     **metrics(frame[mask],{k:v[:,mask] for k,v in values.items()})})
    return rows


def safety_gate(frame, values):
    fallback=[]
    for task in sorted(frame.task_type.unique()):
        mask=frame.task_type.eq(task).to_numpy()
        sub={k:v[:,mask] for k,v in values.items()}
        measured=metrics(frame[mask],sub)
        if measured['quality_loss_worst']>.05+1e-12 or measured['pass_loss_worst']>.02+1e-12:
            fallback.append(task)
    return fallback


def paired_ci(frame, values, repetitions=800):

    records=[]
    for task in ['ALL']+sorted(frame.task_type.unique()):
        mask=np.ones(len(frame),bool) if task=='ALL' else frame.task_type.eq(task).to_numpy()
        sub=frame[mask]
        groups,index=np.unique(sub.group.to_numpy(),return_inverse=True)
        counts=np.bincount(index)
        q=values['q'][:,mask].mean(axis=0)
        cost=values['cost'][:,mask].mean(axis=0)
        delta=sub.q_l.to_numpy()-q
        pass_delta=sub.q_l.ge(.5).to_numpy().astype(float)-(values['q'][:,mask]>=.5).mean(axis=0)
        a=np.bincount(index,weights=delta); b=np.bincount(index,weights=pass_delta)
        c=np.bincount(index,weights=cost); d=np.bincount(index,weights=sub.cost_l.to_numpy())
        rng=np.random.default_rng(SEED+709); draws=[]
        for _ in range(repetitions):
            sample=rng.integers(0,len(groups),len(groups)); n=counts[sample].sum()
            draws.append([a[sample].sum()/n,b[sample].sum()/n,1-c[sample].sum()/d[sample].sum()])
        low,high=np.quantile(draws,[.025,.975],axis=0)
        records.append({'task_type':task,'quality_loss_low':low[0],'quality_loss_high':high[0],
            'pass_loss_low':low[1],'pass_loss_high':high[1], 'cost_saving_low':low[2],'cost_saving_high':high[2],
            'scope':'conditional_prompt_group_bootstrap_not_retraining_or_server_uncertainty'})
    return records


def feedback_pilot(frame, prediction, selected, fallback, learn, rep):
    rng=np.random.default_rng(SEED+2000+rep)
    groups=frame.group.unique().copy(); rng.shuffle(groups)
    task_names=sorted(frame.task_type.unique())
    residual_sum={t:0. for t in task_names}; observed_n={t:0 for t in task_names}
    all_values={}; logs=[]; audit_flags=np.zeros(len(frame),bool)
    for batch,group_batch in enumerate(np.array_split(groups,10)):
        idx=np.flatnonzero(frame.group.isin(group_batch).to_numpy()); sub=frame.iloc[idx].reset_index(drop=True)
        pred=slice_prediction(prediction,np.isin(np.arange(len(frame)),idx))

        pred={**pred,'mu':pred['mu'].copy(),'scores':pred['scores'][:1].copy()}
        bias=np.array([np.clip(residual_sum[t]/(200+observed_n[t]),-.1,.1) if learn else 0. for t in sub.task_type])
        pred['scores']+=bias
        values=apply(sub,pred,selected,fallback)
        audits=set(group_batch[rng.random(len(group_batch))<.10])
        audited=sub.group.isin(audits).to_numpy(); audit_flags[idx]=audited

        extra_s=audited & (values['calls_s'][0]==0)
        extra_l=audited & (values['calls_l'][0]==0)
        values['calls_s'][0]+=extra_s; values['calls_l'][0]+=extra_l
        values['cost'][0]+=extra_s*sub.cost_s.to_numpy()+extra_l*sub.cost_l.to_numpy()
        for name,value in values.items():
            if name not in all_values: all_values[name]=np.empty((1,len(frame)),dtype=value.dtype)
            all_values[name][:,idx]=value
        before={t:float(np.clip(residual_sum[t]/(200+observed_n[t]),-.1,.1)) for t in task_names}
        if learn:
            for task in task_names:
                mask=audited & sub.task_type.eq(task).to_numpy()
                residual_sum[task]+=float((sub.delta_q.to_numpy()[mask]-prediction['mu'][idx][mask]).sum())
                observed_n[task]+=int(mask.sum())
        logs.append({'learn':learn,'rep':rep,'batch':batch,'requests':len(idx),
            'audit_requests':int(audited.sum()),'extra_s':int(extra_s.sum()),'extra_l':int(extra_l.sum()),
            'bias_before':before,'observed_after':dict(observed_n),
            'updated_groups':sorted(audits) if learn else [], 'labels_available_after_batch':True})
    return all_values,logs,audit_flags


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'runs/experiment')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args(); out=args.out.resolve()
    frame=joblib.load(out/'frame.joblib'); frame,cost_meta=fit_cost(frame)
    predictions=joblib.load(out/'router_predictions.joblib')
    save(out/'cost_predictor.json',cost_meta)
    select_mask=frame.phase.eq('select').to_numpy()
    select=frame[select_mask].reset_index(drop=True)
    parameters={}; model_choices=[]; grids=out/'grids';grids.mkdir(exist_ok=True)
    for family,pred in predictions.items():
        cache=out/('selection_'+family+'.json')
        if args.resume and cache.exists(): chosen=json.loads(cache.read_text(encoding='utf-8'))
        else:
            chosen,grid=tune(select,slice_prediction(pred,select_mask))
            save(cache,chosen); grid.to_csv(grids/(family+'.csv.gz'),index=False,compression='gzip')
        parameters[family]=chosen
        values=apply(select,slice_prediction(pred,select_mask),chosen)
        model_choices.append({'family':family,**metrics(select,values)})
        print('POLICY SELECT',family,'cost_saving=',model_choices[-1]['cost_saving'],flush=True)
    ranking=pd.DataFrame(model_choices).sort_values(['model_cost_usd_mean','large_call_rate','quality_mean'],ascending=[True,True,False])
    winner=str(ranking.iloc[0].family)
    ranking.to_csv(out/'model_selection.csv',index=False)
    save(out/'selected_parameters.json',parameters)
    save(out/'model_choice.json',{'selected_on':'selection_partition_only','family':winner,
        'objective':'minimum historical model cost with each task select mean/pass loss <=0.01',
        'fallback_rule':'confirm task mean loss>0.05 or pass loss>0.02 -> all L for that task; aggregate>0.02 -> all L',
        'no_test_reselection':True})

    confirm_mask=frame.phase.eq('confirm').to_numpy(); confirm=frame[confirm_mask].reset_index(drop=True)
    raw_guard=apply(confirm,slice_prediction(predictions[winner],confirm_mask),parameters[winner])
    fallback=safety_gate(confirm,raw_guard)
    protected_guard=apply(confirm,slice_prediction(predictions[winner],confirm_mask),parameters[winner],fallback)
    measured=metrics(confirm,protected_guard)
    if measured['quality_loss_worst']>.02+1e-12 or measured['pass_loss_worst']>.02+1e-12:
        fallback=sorted(frame.task_type.unique())
        protected_guard=apply(confirm,slice_prediction(predictions[winner],confirm_mask),parameters[winner],fallback)
    save(out/'safety_calibration.json',{'family':winner,'fallback_tasks':fallback,
        'raw_candidate':metrics(confirm,raw_guard),'protected':metrics(confirm,protected_guard),
        'not_a_distribution_free_guarantee':True})
    print('FROZEN WINNER',winner,'fallback=',fallback,flush=True)
    results=[]; tasks=[]; ledgers=[]
    test_mask=frame.phase.eq('test').to_numpy(); test=frame[test_mask].reset_index(drop=True)
    for family,pred in predictions.items():
        values=apply(test,slice_prediction(pred,test_mask),parameters[family])
        results.append({'strategy':family,'phase':'test',**metrics(test,values)})
        tasks.extend(task_metrics(test,values,family,'test'));ledgers.append(ledger(test,values,family,'test'))
    active=apply(test,slice_prediction(predictions[winner],test_mask),parameters[winner],fallback)
    results.append({'strategy':'Selected_protected','phase':'test',**metrics(test,active)})
    tasks.extend(task_metrics(test,active,'Selected_protected','test'))
    ledgers.append(ledger(test,active,'Selected_protected','test'))
    for label,use_large in [('All_L',True),('All_S',False)]:
        values=baseline(test,use_large)
        results.append({'strategy':label,'phase':'test',**metrics(test,values)})
        tasks.extend(task_metrics(test,values,label,'test'));ledgers.append(ledger(test,values,label,'test'))

    fixed_pred={'mu':np.zeros(len(frame)),'std':np.zeros(len(frame)),'scores':np.zeros((1,len(frame))),'k_grid':[0.]}
    fixed_params,fixed_grid=tune(select,slice_prediction(fixed_pred,select_mask),fixed_cascade=True)
    save(out/'fixed_cascade_parameters.json',fixed_params)
    fixed_grid.to_csv(grids/'fixed_cascade.csv.gz',index=False,compression='gzip')
    values=apply(test,slice_prediction(fixed_pred,test_mask),fixed_params)
    results.append({'strategy':'Fixed_S_cascade','phase':'test',**metrics(test,values)})
    tasks.extend(task_metrics(test,values,'Fixed_S_cascade','test'));ledgers.append(ledger(test,values,'Fixed_S_cascade','test'))

    for label,disable in [('Selected_no_Post','post'),('Selected_no_VC2','vc2')]:
        altered=copy.deepcopy(parameters[winner])
        for task,setting in altered.items():
            if disable=='post': setting['parameters']['gamma']=None
            else: setting['parameters']['chooser']='L'
        values=apply(test,slice_prediction(predictions[winner],test_mask),altered,fallback)
        results.append({'strategy':label,'phase':'test',**metrics(test,values)})
        tasks.extend(task_metrics(test,values,label,'test'));ledgers.append(ledger(test,values,label,'test'))


    old_run=HERE/'support/rf_baseline'
    old_cache=old_run/'trained_models.joblib'
    if old_cache.exists():
        old_manifest=json.loads((old_run/'experiment.json').read_text(encoding='utf-8'))
        audit=json.loads((out/'q1_source_audit.json').read_text(encoding='utf-8'))
        old_feature_hashes=[v for k,v in old_manifest['source_sha256'].items() if k.endswith('features.npy')]
        assert old_feature_hashes==[audit['old_feature_sha256']]
        old_models,old_predictions,_=joblib.load(old_cache)
        mu=old_predictions['Random Forest']['mu']
        x=np.load(out/'features.npy',allow_pickle=False)
        with threadpool_limits(limits=2):
            spread=np.stack([t.predict(x[:,:392]) for t in old_models['Random Forest'].estimators_]).std(axis=0)
        control={'mu':mu,'std':spread,'scores':mu[None],'k_grid':[0.,.5,1.]}
        control_params,control_grid=tune(select,slice_prediction(control,select_mask))
        save(out/'rf_without_pre_parameters.json',control_params)
        control_grid.to_csv(grids/'RF_without_Pre.csv.gz',index=False,compression='gzip')
        values=apply(test,slice_prediction(control,test_mask),control_params)
        results.append({'strategy':'RF_without_Pre','phase':'test',**metrics(test,values)})
        tasks.extend(task_metrics(test,values,'RF_without_Pre','test'));ledgers.append(ledger(test,values,'RF_without_Pre','test'))
        save(out/'rf_ablation_source.json',{'cached_model':str(old_cache),'sha256':digest(old_cache),
             'same_original_features_sha256':True,'not_eligible_for_main_model_choice':True,
             'reason':'Prior architecture selected in earlier research; descriptive matched ablation only'})

    select_active=apply(select,slice_prediction(predictions[winner],select_mask),parameters[winner],fallback)
    budget=float(select_active['calls_l'].mean()); random_values={}
    for rep in range(10):
        rng=np.random.default_rng(SEED+400+rep)
        large=rng.random(len(test))<budget
        from routing import execute
        values=execute(test,large.astype(float)[None],{'lambda':0.,'tau':.5,'gamma':None,'chooser':'L'})
        for name,v in values.items(): random_values.setdefault(name,[]).append(v)
    random_values={k:np.concatenate(v,axis=0) for k,v in random_values.items()}
    results.append({'strategy':'Random_matched_L_budget','phase':'test',**metrics(test,random_values)})
    tasks.extend(task_metrics(test,random_values,'Random_matched_L_budget','test'))
    ledgers.append(ledger(test,random_values,'Random_matched_L_budget','test'))
    pd.DataFrame(results).to_csv(out/'quality_cost_comparison.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(tasks).to_csv(out/'by_task.csv',index=False,encoding='utf-8-sig')
    pd.concat(ledgers).to_csv(out/'decisions.csv.gz',index=False,compression='gzip')
    pd.DataFrame(paired_ci(test,active)).to_csv(out/'selected_confidence_intervals.csv',index=False)

    feedback_rows=[]; feedback_logs=[]; feedback_ledgers=[]
    for rep in range(3):
        for learn in [False,True]:
            values,logs,audit=feedback_pilot(test,slice_prediction(predictions[winner],test_mask),parameters[winner],fallback,learn,rep)
            name='Audit_residual_update' if learn else 'Audit_frozen_control'
            feedback_rows.append({'strategy':name,'rep':rep,'audit_requests':int(audit.sum()),**metrics(test,values)})
            feedback_logs.extend(logs)
            detail=ledger(test,values,name,'feedback');detail['pilot_rep']=rep;detail['audited']=audit
            feedback_ledgers.append(detail)
    pd.DataFrame(feedback_rows).to_csv(out/'feedback_comparison.csv',index=False)
    save(out/'feedback_audit.json',feedback_logs)
    pd.concat(feedback_ledgers).to_csv(out/'feedback_decisions.csv.gz',index=False,compression='gzip')
    joblib.dump({'frame':frame,'parameters':parameters,'winner':winner,'fallback_tasks':fallback,
                 'predictions':predictions,'cost_meta':cost_meta},out/'routing_state.joblib',compress=3)
    save(out/'evaluation_complete.json',{'completed':True,'winner':winner,'fallback_tasks':fallback,
        'main_strategy':'Selected_protected','research_tolerances_not_problem_requirements':True,
        'feedback_default_enabled':False, 'extra_same_model_compute_enabled':False,
        'reason_no_extra_thinking':'no matched quality outcomes for extra-token or repeated same-model actions'})
    print(pd.DataFrame(results).to_string(index=False),flush=True)


if __name__=='__main__':main()
