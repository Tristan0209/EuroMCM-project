import argparse
import json
import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from prepare import save
from q1_adapter import BASE, TWO, VALID, digest, load_post_source
from routing import apply, slice_prediction
from input_paths import frozen_features
import neural_adapter

HERE=Path(__file__).resolve().parent
MIDS={'s':'mistralai/mixtral-8x7b-chat','l':'gpt-4-1106-preview'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'runs/work1_integrated_20261016')
    parser.add_argument('--resume',action='store_true',help='Reuse passed checks after a targeted interface-only fix')
    args=parser.parse_args();out=args.out.resolve();checks={}
    previous=json.loads((out/'verification.json').read_text(encoding='utf-8')).get('checks',{}) if args.resume else {}
    def check(name,fn):
        if previous.get(name,{}).get('passed'):
            checks[name]={**previous[name],'reused_after_interface_only_fix':True}
            print('VERIFY REUSE',name,flush=True)
            return
        start=time.perf_counter()
        try:
            detail=fn();checks[name]={'passed':True,'details':detail,'wall_s':time.perf_counter()-start}
            print('VERIFY PASS',name,flush=True)
        except Exception as exc:
            checks[name]={'passed':False,'error':repr(exc),'wall_s':time.perf_counter()-start}
            print('VERIFY FAIL',name,repr(exc),flush=True)
    state=joblib.load(out/'routing_state.joblib');frame=state['frame'];test=frame[frame.phase.eq('test')].reset_index(drop=True)
    x=np.load(out/'features.npy',allow_pickle=False);pre=joblib.load(out/'pre_full_train.joblib')
    audit=json.loads((out/'q1_source_audit.json').read_text(encoding='utf-8'))
    def source():
        assert digest(audit['raw_source'])==audit['raw_sha256']
        assert all(digest(p)==h for p,h in audit['work1_source'].items())
        assert frame.groupby('group').phase.nunique().max()==1
        assert frame.phase.value_counts().to_dict()=={'train':21822,'test':7335,'confirm':3645,'select':3615}
        project=Path(audit['raw_source']).parents[2]
        cols=['sample_id','eval_name','prompt']+[m+s for m in MIDS.values() for s in ['', '|total_cost','|model_response']]
        raw=pd.read_csv(audit['raw_source'],usecols=cols).set_index('sample_id').loc[frame.request_id]
        np.testing.assert_array_equal(raw.eval_name,frame.eval_name)
        for action,mid in MIDS.items():
            np.testing.assert_allclose(raw[mid],frame['q_'+action],atol=1e-12)
            np.testing.assert_allclose(raw[mid+'|total_cost'],frame['cost_'+action],atol=1e-12)

        work=project/'Work 1/task1_response time, and environ mental impact'
        processed=pd.read_csv(work/'data/processed/azure_code.csv',nrows=len(test))
        original=pd.read_csv(project/'AI_datasets/Azure/AzureLLMInferenceTrace_code_1week.csv',usecols=['TIMESTAMP'],nrows=len(test))
        stamps=pd.to_datetime(original.TIMESTAMP,utc=True)
        np.testing.assert_allclose((stamps-stamps.iloc[0]).dt.total_seconds(),processed.arrival_s,atol=1e-9)
        return {'requests':len(frame),'paired_labels_checked':2*len(frame),'arrival_prefix_checked':len(test),'no_cross_phase_groups':True}
    check('source_and_split',source)
    def oof():
        train_groups=set(frame.loc[frame.phase.eq('train'),'group'])
        total=0
        for kind,file,manifest in [('Pre','pre_predictions.csv.gz','pre_provenance.json'),('Post','post_predictions.csv.gz','post_provenance.json')]:
            pred=pd.read_csv(out/file);meta=json.loads((out/manifest).read_text(encoding='utf-8'))
            for fit_id,part in pred.groupby('fit_id'):
                used=set(meta[fit_id]['training_groups'])
                assert used<=train_groups and used.isdisjoint(set(part.group))
                if part.split.eq('train').any():assert meta[fit_id]['oof']
            total+=len(pred)
        non_mc=~frame.task_kind.eq('multiple_choice')
        assert frame.loc[non_mc,['v_s','v_l','vc2_s','vc2_l']].isna().all().all()
        assert np.isfinite(frame.loc[~non_mc,['v_s','v_l','vc2_s','vc2_l']]).all().all()
        return {'prediction_rows':total,'same_prompt_and_counterpart_excluded_from_fits':True,'non_MC_quality_not_filled':True}
    check('Pre_Post_OOF_provenance',oof)
    def features_and_models():
        source_x=frozen_features(Path(audit['raw_source']).parents[2])/'features.npy'
        np.testing.assert_array_equal(np.load(source_x),x[:,:392])
        scaler=joblib.load(out/'pre_feature_scaler.joblib')
        np.testing.assert_allclose(x[:,392:],scaler.transform(frame[['pre_s','pre_l','pre_delta']]),atol=5e-7)
        subset=test.iloc[:100]
        for action,mid in MIDS.items():
            rows=pd.DataFrame({'token_length':subset.prompt_chars,'eval_name':subset.eval_name,'model_id':mid})
            values=np.clip(pre.models['A_Baseline'].predict(pre.basic(rows)[BASE]),0,1)
            np.testing.assert_allclose(values,subset['pre_'+action],atol=1e-9)
        roundoff={};mask=frame.phase.eq('test').to_numpy()
        with threadpool_limits(limits=2):
            for family in ['Ridge','Bayesian_Ridge','Logistic_Regression','CART','LightGBM','MLP','Random_Forest']:
                model=joblib.load(out/'routers'/f'{family}.joblib')[0]
                if family=='MLP':pred=model.predict(x[mask])[0]
                else:pred=np.clip(model.predict(x[mask]),-1,1)
                cached=state['predictions'][family]['mu'][mask]
                np.testing.assert_allclose(pred,cached,atol=1e-6,rtol=1e-6)
                roundoff[family]=float(np.max(np.abs(pred-cached)))
        return {'features':395,'new_model_predictions_reproduced':7,'maximum_batch_roundoff':roundoff}
    check('features_and_saved_models',features_and_models)
    def post_models():
        ns=load_post_source(out/'work1_source');bundle=joblib.load(out/'post_models.joblib')
        sample=test[test.task_kind.eq('multiple_choice')].iloc[:100]
        for action,mid in MIDS.items():
            rows=[]
            for req in sample.itertuples():
                own=ns['diagnostic'](req.prompt,getattr(req,'response_'+action),'multiple_choice')
                peer=ns['diagnostic'](req.prompt,getattr(req,'response_'+('l' if action=='s' else 's')),'multiple_choice')['answer_choice']
                rows.append({**own,'model_id':mid,'peer_valid':int(peer is not None),
                    'peer_agree':float(own['answer_choice']==peer) if own['answer_choice'] is not None and peer is not None else np.nan})
            rows=pd.DataFrame(rows);rows['model_id']=pd.Categorical(rows.model_id,categories=bundle['model_names'])
            for name,cols in [('V',VALID),('VC2',TWO)]:
                values=np.clip(bundle['models'][name].predict(rows[cols]),0,1)
                np.testing.assert_allclose(values,sample[name.lower()+'_'+action],atol=1e-9)
        return {'answers_checked':200,'both_assessors_reproduced':True,'peer_is_actual_S_L_pair':True}
    check('Post_saved_prediction_replay',post_models)
    def ledger_check():
        paths=pd.read_csv(out/'decisions.csv.gz').merge(test[['request_id','q_s','q_l','cost_s','cost_l']],on='request_id',validate='many_to_one')
        np.testing.assert_array_equal(paths.final_l.astype(int),paths.final_l)
        np.testing.assert_allclose(paths.q,np.where(paths.final_l,paths.q_l,paths.q_s),atol=1e-12)
        np.testing.assert_allclose(paths.cost,paths.calls_s*paths.cost_s+paths.calls_l*paths.cost_l,atol=1e-12)
        assert ((paths.calls_s+paths.calls_l)>=1).all()
        assert (~paths.upgraded | ((paths.calls_s==1)&(paths.calls_l==1))).all()
        table=pd.read_csv(out/'quality_cost_comparison.csv')
        for strategy,part in paths.groupby('strategy'):
            for _,rep in part.groupby('rep'):
                assert len(rep)==len(test) and not rep.request_id.duplicated().any()
            row=table[table.strategy.eq(strategy)].iloc[0]
            np.testing.assert_allclose([part.q.mean(),part.q.ge(.5).mean(),part.cost.mean()],
                                       [row.quality_mean,row.pass_rate,row.model_cost_usd_mean],atol=1e-12)
        mask=frame.phase.eq('test').to_numpy()
        values=apply(test,slice_prediction(state['predictions'][state['winner']],mask),state['parameters'][state['winner']],state['fallback_tasks'])
        final=paths[paths.strategy.eq(state['main_strategy'])].set_index('request_id').loc[test.request_id]
        np.testing.assert_allclose(values['q'][0],final.q,atol=1e-12)
        np.testing.assert_array_equal(values['initial_l'][0],final.initial_l)
        np.testing.assert_allclose(values['cost'][0],final.cost,atol=1e-12)
        return {'ledger_rows':len(paths),'strategies':int(paths.strategy.nunique()),'initial_and_paid_final_paths_reproduced':True}
    check('all_strategy_quality_cost_ledger',ledger_check)
    def feedback():
        paths=pd.read_csv(out/'feedback_decisions.csv.gz').merge(test[['request_id','q_s','q_l','cost_s','cost_l']],on='request_id',validate='many_to_one')
        np.testing.assert_allclose(paths.q,np.where(paths.final_l,paths.q_l,paths.q_s),atol=1e-12)
        np.testing.assert_allclose(paths.cost,paths.calls_s*paths.cost_s+paths.calls_l*paths.cost_l,atol=1e-12)
        assert (((paths.calls_s==1)&(paths.calls_l==1))|~paths.audited).all()
        assert ((paths.calls_s<=1)&(paths.calls_l<=1)).all()
        logs=json.loads((out/'feedback_audit.json').read_text(encoding='utf-8'))
        for row in logs:
            assert row['labels_available_after_batch']
            if row['learn']:assert len(row['updated_groups'])<=row['audit_requests']
        return {'feedback_ledger_rows':len(paths),'batch_logs':len(logs),'audit_generation_costs_included':True}
    check('paid_feedback_audit',feedback)
    def resources():
        scenarios=pd.read_csv(out/'resource_scenarios.csv')
        meta=json.loads((out/'resource_assumptions.json').read_text(encoding='utf-8'))
        np.testing.assert_allclose(scenarios.assumed_gco2e,scenarios.assumed_total_kwh*meta['carbon_factor'],atol=1e-10)
        np.testing.assert_allclose(scenarios.assumed_selfhost_total_eur,
            scenarios.assumed_selfhost_electricity_eur+scenarios.assumed_selfhost_hardware_eur,atol=1e-10)
        assert scenarios[['actual_end_to_end_latency_s','actual_total_energy_kwh']].isna().all().all()
        jobs=pd.read_csv(out/'primary_call_schedule.csv.gz');timeline=pd.read_csv(out/'primary_power_timeline.csv.gz')
        np.testing.assert_allclose(jobs.finish_s-jobs.start_s,jobs.service_s,atol=1e-9)
        assert (jobs.start_s>=jobs.arrival_s-1e-9).all()
        for _,part in jobs.groupby(['strategy','action','replica']):
            part=part.sort_values('start_s');assert np.all(part.start_s.to_numpy()[1:]>=part.finish_s.to_numpy()[:-1]-1e-9)
        primary=scenarios[(scenarios.large_service_multiplier==3)&(scenarios.arrival_multiplier==1)&(scenarios.replicas_per_model==1)&(scenarios.rep==0)]
        for strategy,part in timeline.groupby('strategy'):
            row=primary[primary.strategy.eq(strategy)].iloc[0]
            np.testing.assert_allclose(part.facility_kwh.sum(),row.assumed_server_kwh,atol=1e-10)
        return {'conditional_scenario_rows':len(scenarios),'primary_call_rows':len(jobs),
                'energy_and_cost_conserved':True,'API_USD_and_selfhost_EUR_separate':True,'actuals_not_filled_with_zero':True}
    check('FCFS_energy_monetary_accounting',resources)
    def inference():
        from integrated_api import IntegratedRouter
        router=IntegratedRouter(out);mask=frame.phase.eq('test').to_numpy()
        cached=apply(test,slice_prediction(state['predictions'][state['winner']],mask),state['parameters'][state['winner']],state['fallback_tasks'])
        examples=[]
        for desired in [False,True]:
            eligible=np.flatnonzero(cached['initial_l'][0]==desired)
            for i in eligible[:2]:
                r=test.iloc[i];result=router.route(r.prompt,r.eval_name)
                assert result['initial_action']==('L' if desired else 'S')
                assert result['LLM_calls_executed']==0
                examples.append({'request_id':r.request_id,**result})
        assert router.route('Unregistered service request','unknown_registry_task')['initial_action']=='L'
        try:router.choose_after_upgrade('x','A',None,'hellaswag')
        except ValueError:pass
        else:raise AssertionError('Free peer accepted')
        save(out/'inference_check.json',{'examples':examples,'unknown_task_fallback':'L','paid_LLM_calls':0,'passed':True})
        return {'fresh_Pre_route_examples':len(examples),'unknown_registry_fallback':True,'no_paid_LLM_API_executed':True}
    check('fresh_inference_API',inference)
    results={'passed':all(c['passed'] for c in checks.values()),'checks':checks,
        'does_not_establish_real_system_accuracy_or_unseen_holdout_guarantee':True}
    save(out/'verification.json',results)
    snapshot=out/'source';snapshot.mkdir(exist_ok=True)
    for p in HERE.glob('*.py'):shutil.copy2(p,snapshot/p.name)
    save(out/'source_manifest.json',{p.name:digest(p) for p in HERE.glob('*.py')})
    if not results['passed']:raise AssertionError('See verification.json')
    print('ALL INTEGRATION CHECKS PASSED',flush=True)


if __name__=='__main__':main()
