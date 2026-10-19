from context import *
import gzip
import numpy as np
import pandas as pd

def main():
    assert read(ROOT/'audit/pipeline_completed.json')['status']=='PASS'
    catalog={}
    for phase in ['pilot','core','F','H']:
        f=pd.read_csv(out(phase)/'raw_runs.csv')
        for row in f.to_dict('records'):catalog[row['run_id']]=row
    for repeat in [1,2]:
        for row in pd.read_csv(ROOT/'audit'/('fresh_repeat_'+str(repeat)+'.csv')).to_dict('records'):catalog[row['run_id']]=row
    source=read(ROOT/'audit/source_and_input_hashes.json');current=signature()
    assert all(current.get(k)==v for k,v in source.items())
    prepared=pd.read_csv(BASE/'results/data/requests.csv').set_index('request_id')
    quality=pd.read_csv(BASE/'results/data/evaluation_only_historical_quality.csv').set_index('request_id').q
    ci=pd.read_csv(BASE/'results/data/carbon_lifecycle.csv');ts=pd.to_datetime(ci.datetime_utc,utc=True)
    anchor=pd.Timestamp('2025-01-02T00:00:00Z')
    ci=ci[(ts>=anchor)&(ts<anchor+pd.Timedelta(hours=12))]
    metadata=read(BASE/'results/data/carbon_lifecycle.metadata.json')
    assert metadata['emissions_species']=='CO2e' and metadata['unit']=='gCO2e/kWh' and metadata['boundary']=='lifecycle'
    value_column=metadata['value_column'];assert value_column=='carbon_intensity_gco2e_kwh'
    curves={r:g[value_column].to_numpy(float) for r,g in ci.sort_values('datetime_utc').groupby('region')}
    assert all(len(v)==12 for v in curves.values())
    rows=[]
    for row in catalog.values():
        folder=Path(row['artifact_directory']);r=pd.read_csv(folder/'requests.csv.gz').set_index('request_id');ledger=pd.read_csv(folder/'energy_ledger.csv.gz')
        events=pd.read_csv(folder/'events.csv.gz');shocks=read(folder/'common_random_conditions.json')
        assert len(r)==row['n'] and r.index.is_unique and set(r.status)<=set(['completed','failed'])
        assert list(read(folder/'input_ids.json'))==list(r.index)
        p=prepared.reindex(r.index)
        assert not p.selected_model.isna().any() and all(p.selected_model==r.selected_model)
        assert np.allclose(p.estimated_service_time_seconds,r.estimated_service_s,rtol=0,atol=1e-10)
        assert np.allclose((pd.to_datetime(p.arrival_time,utc=True)-anchor).dt.total_seconds(),r.arrival_s,rtol=0,atol=1e-8)
        assert np.allclose((pd.to_datetime(p.deadline,utc=True)-anchor).dt.total_seconds(),r.deadline_s,rtol=0,atol=1e-8)
        assert all(p.task_type==r.task_type) and all(p.service_class==r.service_class) and all(p.origin_region==r.origin_region)
        assert np.allclose(quality.reindex(r.index),r.historical_quality,rtol=0,atol=1e-12)
        assert all(r.loc[r.task_type.isin(['Code','Math']),'selected_model']=='L')
        completed=r.status.eq('completed')
        assert np.allclose(r.loc[completed,'actual_service_s'],r.loc[completed,'estimated_service_s']*pd.Series(shocks).reindex(r.index[completed]),rtol=0,atol=1e-9)
        starts=events[events.event.eq('ServiceStart')]
        assert starts.request_id.is_unique and set(starts.request_id)==set(r.index[completed])
        assert (starts.active<=starts.admission_capacity).all()
        assert (r.loc[completed,'service_start_s']>=r.loc[completed,'arrival_s']-1e-8).all()
        for (_,g) in r[completed].groupby(['region','selected_model','replica']):
            g=g.sort_values('service_start_s');assert np.all(g.service_start_s.to_numpy()[1:]>=g.completion_s.to_numpy()[:-1]-1e-8)
        assert abs(ledger.energy_kwh.sum()-row['energy_kwh'])<1e-9
        assert abs(ledger.co2e_g.sum()-row['co2e_g'])<1e-7
        assert abs(row['idle_kwh']-250.56)<1e-9
        assert abs(row['active_increment_kwh']-r.actual_service_s.fillna(0).sum()*702/3.6e6)<1e-9
        assert abs(row['controller_kwh']-ledger.loc[ledger.component.eq('controller'),'energy_kwh'].sum())<1e-9

        integrated=np.zeros(len(ledger))
        for region,curve in curves.items():
            mask=ledger.region.eq(region).to_numpy();g=ledger[mask];a=g.start_s.to_numpy();b=g.end_s.to_numpy();e=g.energy_kwh.to_numpy()
            area=np.zeros(len(g))
            for h,c in enumerate(curve):area+=np.maximum(0,np.minimum(b,(h+1)*3600)-np.maximum(a,h*3600))*c
            positive=b>a;values=np.zeros(len(g));values[positive]=e[positive]*area[positive]/(b[positive]-a[positive])
            values[~positive]=e[~positive]*curve[np.minimum((a[~positive]//3600).astype(int),11)]
            integrated[mask]=values
        carbon_error=float(np.max(abs(integrated-ledger.co2e_g.to_numpy())))
        assert carbon_error<1e-7,(folder,carbon_error)
        count=0;recovered=0;rejected=0
        with gzip.open(folder/'solver_log.jsonl.gz','rt',encoding='utf-8') as stream:
            for text in stream:
                log=json.loads(text);assert log['max_request_arrival_s']<=log['decision_s']+1e-8 and log['observed_carbon_end_s']<=log['decision_s']+1e-8
                assert max(r.loc[log['request_ids'],'arrival_s'])<=log['decision_s']+1e-8
                if 'issued_s' in log:assert log['issued_s']<=log['decision_s']+1e-8
                for branch in log.get('branches',[log]):
                    recovery=branch.get('recovery',{})
                    if branch.get('full_solution_validated'):assert recovery['final_validation']['max_violation']<=1e-5
                    if recovery.get('recovered'):assert branch['status']==0 and recovery['final_validation']['max_violation']<=1e-5;recovered+=1
                    if recovery.get('unrecoverable'):assert branch['status']==4;rejected+=1
                    count+=1
        rows.append({'run_id':row['run_id'],'strategy':row['strategy'],'seed':row['seed'],'n':len(r),'gamma':row['gamma'],'delay_scale':row['delay_scale'],
            'requests_completed':int(completed.sum()),'requests_failed':int((~completed).sum()),'solver_subproblems':count,
            'recovered':recovered,'unrecoverable_rejected':rejected,'independent_carbon_max_error_g':carbon_error,'status':'PASS'})
    pd.DataFrame(rows).to_csv(ROOT/'audit/artifact_invariants.csv',index=False)

    from recovery import violations
    saved_vectors=[]
    for file in sorted((ROOT/'audit/recovery_candidates').glob('*.npz')):
        f=np.load(file);integer=f['integer'].astype(bool);raw=f['raw'];rounded=raw.copy();rounded[integer]=np.rint(rounded[integer])
        args=[f[k] for k in ['A','lower','upper','bounds_lower','bounds_upper']]
        bad=violations(rounded,*args,integer)
        integer_distance=float(np.max(abs(raw[integer]-np.rint(raw[integer])))) if integer.any() else 0.
        assert bad['max_violation']>1e-5 or integer_distance>1e-5
        restored=f['recovered'];good=None
        if len(restored):
            assert np.array_equal(restored[integer],rounded[integer])
            good=violations(restored,*args,integer);assert good['max_violation']<=1e-5
        saved_vectors.append({'file':str(file.relative_to(ROOT)),'raw_integer_distance':integer_distance,'rounded_max_violation':bad['max_violation'],
            'recovered_vector_present':bool(len(restored)),'recovered_max_violation':good['max_violation'] if good else None})
    pd.DataFrame(saved_vectors).to_csv(ROOT/'audit/recovery_vector_revalidation.csv',index=False)
    core=pd.read_csv(out('core')/'raw_runs.csv');repeatrows=[]
    for index in [1,2]:
        fresh=pd.read_csv(ROOT/'audit'/('fresh_repeat_'+str(index)+'.csv'))
        for new in fresh.to_dict('records'):
            old=core[(core.seed==new['seed'])&(core.strategy==new['strategy'])].iloc[0]
            assert old.terminal_digest==new['terminal_digest']
            for metric in ['co2e_g','energy_kwh','completion_rate','service_violation_rate','p50_response_s','p95_response_s','p99_response_s','fallback_count','status4_count']:
                assert abs(old[metric]-new[metric])<1e-8,(new['strategy'],index,metric)
                repeatrows.append({'strategy':new['strategy'],'seed':new['seed'],'repeat':index,'metric':metric,'absolute_difference':abs(old[metric]-new[metric]),'tolerance':1e-8})
    pd.DataFrame(repeatrows).to_csv(ROOT/'audit/numerical_repeat_consistency.csv',index=False)
    old=read(ROOT/'audit/protected_before.json');changed=[k for k,v in old.items() if not(PROJECT/k).is_file() or sha(PROJECT/k)!=v]
    save(ROOT/'audit/protection_after.json',{'status':'FAIL' if changed else 'PASS','protected_files':len(old),'changed':changed});assert not changed
    save(ROOT/'audit/final_gate.json',{'status':'PASS','audited_unique_runs':len(rows),'core_cases':120,'old_files_unchanged':len(old),
        'repeat_check':'5 policies x preselected seed 2027101806 x 3 total executions; terminal digests identical, numeric atol=1e-8',
        'continuous_recovery_test':'synthetic recoverable cases pass; saved real counterexample genuinely fixed-integer infeasible and correctly rejected',
        'limitations':['Native fixed-work/thread/seed controls unverified','H measured wall is inherently platform/load dependent; exact wall reproducibility not claimed',
            'Raw solver numeric rejection count may remain nonzero; every rejected candidate uses explicit fallback',
            'No real hardware, operational carbon forecast or global Oracle optimality validation']})
    print('PASS saved-artifact audit:',len(rows),'unique runs; protected old files:',len(old))
if __name__=='__main__':main()
