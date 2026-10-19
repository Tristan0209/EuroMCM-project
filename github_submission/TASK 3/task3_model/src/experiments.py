import gzip
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

from .util import ROOT,DATA,read,save,sha,input_signature
from .domain import load_inputs,Availability,shocks_for
from .carbon import Carbon
from .des import Simulation
from .accounting import evaluate


def audit_run(sim,records):
    if set(records.request_id)!=set(sim.q) or records.request_id.duplicated().any():raise AssertionError('Lost/duplicate terminal record')
    for row in records.itertuples():
        q=sim.q[row.request_id]
        if row.selected_model!=q.model:raise AssertionError('Model type changed')
        if row.status=='completed':
            if not sim.network[(q.origin,row.region)]['allowed']:raise AssertionError('Prohibited destination')
            if row.service_start_s<q.arrival-1e-6:raise AssertionError('Prearrival start')
            if not np.isclose(row.actual_service_s,q.service*sim._shocks[q.request_id]):raise AssertionError('Service multiplier/shock mismatch')
            identity=sim.cfg['router_seconds']+row.intentional_delay_s+row.network_delay_s+row.queue_wait_s+row.actual_service_s
            if not np.isclose(row.response_s,identity,atol=1e-5):raise AssertionError('Latency components counted twice')
    intervals={}
    for row in records[records.status.eq('completed')].itertuples():
        key=(row.region,row.selected_model,int(row.replica))
        intervals.setdefault(key,[]).append((row.service_start_s,row.completion_s))
        if row.replica>=sim.availability.at(row.region,row.selected_model,row.service_start_s):raise AssertionError('Unavailable replica admitted a new job')
    for block in intervals.values():
        block.sort()
        if any(right>next_left+1e-6 for (_,right),(next_left,_) in zip(block,block[1:])):raise AssertionError('Physical replica overlapped')
    if any(x['observed_end_s']>x['issued_s']+1e-8 for x in sim.forecast_audit):raise AssertionError('Future observed carbon')
    if sim.strategy=='MPC' and any(x['max_request_arrival_s']>x['decision_s']+1e-8 for x in sim.solver_log):raise AssertionError('Future arrivals leaked')
    return True


def run_one(strategy,n=7335,rep=0,scenario='normal',load=1.,delay_scale=1.,spread=1.,network_scale=1.,defer_fraction=1.,boundary='lifecycle',model=None):
    core=read(ROOT/'configs/core.json')
    model={**core['model'],**(model or {}),'seed':core['seed']}
    config={'strategy':strategy,'n':n,'rep':rep,'scenario':scenario,'load':load,'delay_scale':delay_scale,
        'spread':spread,'network_scale':network_scale,'defer_fraction':defer_fraction,'boundary':boundary,'model':model}
    config['implementation_sha256']={name:sha(ROOT/'src'/name) for name in ['domain.py','carbon.py','des.py','milp.py','accounting.py','power.py']}
    legacy_key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    config['inputs_sha256']=input_signature()['sha256']
    key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()[:16]
    folder=ROOT/'results/runs'/key
    if (folder/'metrics.json').is_file():return {**read(folder/'metrics.json'),'run_id':key}
    legacy=ROOT/'results/runs'/legacy_key/'metrics.json'
    if legacy.is_file():
        metric=read(legacy)
        if metric.get('audited_inputs_sha256')==config['inputs_sha256']:
            return {**metric,'run_id':legacy_key}
    folder.mkdir(parents=True,exist_ok=True);save(folder/'config.json',config)
    started=time.perf_counter()
    requests,servers,network,anchor=load_inputs(n,load,delay_scale,defer_fraction,network_scale)
    carbon=Carbon.load(anchor,boundary,spread)
    availability=Availability(scenario,anchor)
    shocks=shocks_for(rep,[q.request_id for q in requests])

    quality=pd.read_csv(DATA/'evaluation_only_historical_quality.csv').set_index('request_id').q.to_dict()
    save(folder/'input_ids.json',[q.request_id for q in requests])
    save(folder/'common_random_conditions.json',shocks)
    sim=Simulation(requests,servers,network,availability,carbon,shocks,strategy,model).run()
    summary,records,ledger=evaluate(sim,sim.end,quality)
    audit_run(sim,records)
    records.to_csv(folder/'requests.csv.gz',index=False,compression='gzip')
    ledger.to_csv(folder/'energy_ledger.csv.gz',index=False,compression='gzip')
    pd.DataFrame(sim.log).to_csv(folder/'events.csv.gz',index=False,compression='gzip')
    with gzip.open(folder/'solver_log.jsonl.gz','wt',encoding='utf-8') as f:
        for log in sim.solver_log:f.write(json.dumps(log,ensure_ascii=False,default=lambda x:x.item())+'\n')
    if sim.forecasts:pd.DataFrame(sim.forecasts).drop_duplicates(['issued_at_utc','datetime_utc','region']).to_csv(folder/'issued_forecasts.csv.gz',index=False,compression='gzip')
    summary.update({k:v for k,v in config.items() if k!='model'})
    summary.update({'run_id':key,'runtime_seconds':time.perf_counter()-started,'all_invariants_passed':True,
        'input_ids_sha256':sha(folder/'input_ids.json'),'shocks_sha256':sha(folder/'common_random_conditions.json'),
        'solver_backend':'HiGHS 1.2.2 / SciPy 1.9.1','source_code_sha256':{p.name:sha(p) for p in (ROOT/'src').glob('*.py')}})
    save(folder/'metrics.json',summary)
    print(f"RUN {strategy:6s} n={n:4d} rep={rep} {scenario} load={load} g={summary['co2e_g']:.6f} SLA={summary['service_violation_rate']:.4f} wall={summary['runtime_seconds']:.2f}s",flush=True)
    return summary


def record_group(name,cases):
    result=[]
    for label,kwargs in cases:
        value=run_one(**kwargs);result.append({**value,'experiment':name,'setting':label})
        pd.DataFrame(result).to_csv(ROOT/'results'/('experiment_'+name+'.csv'),index=False)
    return result


def scale(sizes):
    core=read(ROOT/'configs/core.json');cases=[]
    for n in sizes:
        for strategy in core['strategies']:cases.append((str(n),{'strategy':strategy,'n':n,'rep':0}))
    result=record_group('scale',cases)
    if not all(x['all_invariants_passed'] for x in result):raise RuntimeError('Scale gate failed')
    save(ROOT/'results/audit/scale_gate.json',{'sizes_completed':sizes,'all_invariants_passed':True,
        'no_global_optimality_claim':True,'runs':[v['run_id'] for v in result]})


def suite():
    core=read(ROOT/'configs/core.json');reps=core['repetitions'];strategies=core['strategies'];n=core['sensitivity_requests']
    record_group('main',[('normal',{'strategy':s,'n':7335,'rep':rep,'model':{'export_forecasts':s=='MPC' and rep==0}})
        for rep in reps for s in strategies])
    record_group('demand',[(str(load),{'strategy':s,'n':n,'rep':rep,'load':load})
        for load in [.5,1.,2.] for rep in reps for s in strategies])
    record_group('availability',[(scenario,{'strategy':s,'n':n,'rep':rep,'scenario':scenario})
        for scenario in ['normal','local_failure','high_load_capacity_reduction'] for rep in reps for s in strategies])
    record_group('delay',[(str(factor),{'strategy':s,'n':n,'rep':rep,'delay_scale':factor})
        for factor in [.25,.5,1.,2.] for rep in reps for s in strategies])
    forecast_cases=[]
    for method,noise in [('persistence',0.),('historical_hour',0.),('persistence',.3),('persistence',.8)]:
        forecast_cases += [(method+'_'+str(noise),{'strategy':'MPC','n':n,'rep':rep,
            'model':{'forecast_method':method,'forecast_noise':noise}}) for rep in reps]
    forecast_cases += [('perfect_foresight_restricted',{'strategy':'Oracle','n':n,'rep':rep}) for rep in reps]
    record_group('forecast',forecast_cases)
    m=core['boundary_requests'];boundary_cases=[]
    axes={'spread':[0.,.25,.5,1.], 'load':[1.,2.,4.,8.], 'network_scale':[1.,10.,100.],
          'solve_delay_seconds':[.01,.1,1.,5.], 'forecast_noise':[0.,.3,.8,1.5], 'defer_fraction':[0.,.25,.5,1.]}
    for axis,values in axes.items():
        for value in values:
            for rep in reps:
                for s in ['Fixed','Greedy','MPC']:
                    kwargs={'strategy':s,'n':m,'rep':rep}
                    if axis in ['solve_delay_seconds','forecast_noise']:
                        if s=='MPC':kwargs['model']={axis:value}
                    else:kwargs[axis]=value
                    boundary_cases.append((axis+'='+str(value),kwargs))
    record_group('boundary',boundary_cases)
    record_group('horizon',[(f'{h}h_{interval}s',{'strategy':'MPC','n':m,'rep':rep,
        'model':{'horizon_seconds':h*3600,'reoptimization_seconds':interval}})
        for h in [1,3,6] for interval in [300,900,1800] for rep in reps])
    record_group('methodology',[('direct_CO2e',{'strategy':s,'n':n,'rep':rep,'boundary':'direct'}) for rep in reps for s in strategies])
    record_group('measured_controller',[('charge_wall_time',{'strategy':s,'n':m,'rep':rep,'model':{'measured_wall_fraction':1.}})
        for rep in reps for s in strategies])


def full_stress():
    core=read(ROOT/'configs/core.json')
    cases=[]
    for scenario,load in [('normal',2.),('local_failure',1.),('high_load_capacity_reduction',1.)]:
        for rep in core['repetitions']:
            for strategy in core['strategies']:
                cases.append((scenario+'_load'+str(load),{'strategy':strategy,'n':7335,'rep':rep,'scenario':scenario,'load':load}))
    record_group('full_stress',cases)
