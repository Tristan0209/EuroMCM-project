from common_a2 import *
import argparse
import gzip
import time
import traceback
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from src.domain import load_inputs,Availability,shocks_for
from src.carbon import Carbon
from src.accounting import evaluate
from src.experiments import audit_run
from v2.des import Simulation


def audit_extended(sim,records,ledger):
    audit_run(sim,records)
    for e in sim.log:
        if e['event']=='ServiceStart':
            assert e['replica']<sim.availability.at(e['region'],e['model'],e['time_s'])
            assert e['replica']<sim.servers[(e['region'],e['model'])]['capacity']
    for component in ['idle','active_increment','network','controller']:
        assert ledger.loc[ledger.component.eq(component),'energy_kwh'].ge(0).all()
    assert records.status.isin(['completed','failed']).all()
    assert records.loc[records.status.eq('failed'),'failure_reason'].notna().all()
    assert len(records)==len(sim.requests)
    return True


def execute(folder,strategy,rep,cfg,n=7335,scenario='normal',inputs=None,shocks=None,quality=None):
    if folder.exists():raise FileExistsError('Never overwrite an A2 execution: '+str(folder))
    folder.mkdir(parents=True)
    save(folder/'config.json',{'strategy':strategy,'service_rep':rep,'n':n,'scenario':scenario,'model':cfg,'code':code_signature()})
    started=time.perf_counter()
    requests,servers,network,anchor=inputs or load_inputs(n)
    availability=Availability(scenario,anchor)
    carbon=Carbon.load(anchor)
    shocks=shocks if shocks is not None else shocks_for(rep,[q.request_id for q in requests])
    quality=quality if quality is not None else pd.read_csv(BASE/'results/data/evaluation_only_historical_quality.csv').set_index('request_id').q.to_dict()
    save(folder/'input_ids.json',[q.request_id for q in requests]);save(folder/'common_random_conditions.json',shocks)
    sim=Simulation(requests,servers,network,availability,carbon,shocks,strategy,cfg).run()
    metrics,records,ledger=evaluate(sim,sim.end,quality)
    audit_extended(sim,records,ledger)
    records.to_csv(folder/'requests.csv.gz',index=False,compression='gzip')
    ledger.to_csv(folder/'energy_ledger.csv.gz',index=False,compression='gzip')
    events=pd.DataFrame(sim.log);events.to_csv(folder/'events.csv.gz',index=False,compression='gzip')
    with gzip.open(folder/'solver_log.jsonl.gz','wt',encoding='utf-8') as f:
        for x in sim.solver_log:f.write(json.dumps(x,default=lambda a:a.item())+'\n')


    record_text=records.sort_values('request_id').to_csv(index=False,float_format='%.12g',na_rep='NA')
    event_text=events.to_csv(index=False,float_format='%.12g',na_rep='NA')
    solver_decisions=[{k:v for k,v in x.items() if k not in ['wall_seconds','option_warnings']} for x in sim.solver_log]
    metrics.update({'runtime_s':time.perf_counter()-started,'invariants_passed':True,'service_rep':rep,
        'request_digest':hashlib.sha256(record_text.encode()).hexdigest(),'event_digest':hashlib.sha256(event_text.encode()).hexdigest(),
        'solver_decision_digest':hashlib.sha256(json.dumps(solver_decisions,sort_keys=True,default=lambda a:a.item()).encode()).hexdigest(),
        'input_id_digest':sha(folder/'input_ids.json'),'shock_digest':sha(folder/'common_random_conditions.json'),
        'code_signature':code_signature(),'folder':str(folder.relative_to(HERE))})
    save(folder/'metrics.json',metrics)
    return metrics


def compare_repeats(rows):
    frame=pd.DataFrame(rows);tol=read(HERE/'configs/protocol.json')['tolerances'];checks=[]
    for (strategy,rep),g in frame.groupby(['strategy','service_rep']):
        reasons=[]
        if len(g)!=3:reasons.append('not_three_executions')
        for column in ['request_digest','event_digest','solver_decision_digest','shock_digest','input_id_digest']:
            if g[column].nunique()!=1:reasons.append(column+'_differs')
        for column in ['co2e_g','energy_kwh','p95_response_s','completion_rate','service_violation_rate']:
            limit=tol['rate' if column.endswith('rate') else 'time_seconds' if column.endswith('_s') else column]
            if g[column].max()-g[column].min()>limit:reasons.append(column+'_outside_tolerance')
        checks.append({'strategy':strategy,'service_rep':rep,'executions':len(g),'status':'FAIL' if reasons else 'PASS','reasons':';'.join(reasons),
            'carbon_span_g':g.co2e_g.max()-g.co2e_g.min(),'p95_span_s':g.p95_response_s.max()-g.p95_response_s.min()})
    pd.DataFrame(checks).to_csv(HERE/'repeat_consistency.csv',index=False)
    return checks


def main():
    args=argparse.ArgumentParser();args.add_argument('--phase',choices=['full','stress'],default='full');args=args.parse_args()
    tree=ET.parse(HERE/'audit/tests_v2.xml').getroot()
    suite=next(iter(tree)) if tree.tag=='testsuites' else tree
    assert int(suite.attrib['tests'])>=45 and int(suite.attrib['failures'])+int(suite.attrib['errors'])==0,'All tests must pass first'
    frozen=read(HERE/'configs/solver_protocol_frozen.json')
    assert frozen['core_source']==code_signature(),'Frozen source/config changed after solver pilot'
    mainframe=pd.read_csv(BASE/'results/experiment_main.csv')
    rows=[];errors=[]
    if args.phase=='stress':
        plan=[(s,r,e,scenario) for e in range(3) for scenario in ['local_failure','high_load_capacity_reduction'] for r in [0] for s in ['Fixed','Greedy','MPC','Oracle']]
    else:plan=[(s,r,e,'normal') for e in range(3) for r in range(3) for s in ['Fixed','Greedy','MPC','Oracle']]
    for strategy,rep,execution,scenario in plan:
        old=mainframe[(mainframe.strategy==strategy)&(mainframe.rep==rep)].iloc[0]
        cfg=read(BASE/'results/runs'/old.run_id/'config.json')['model']
        cfg={**cfg,**frozen['model_overrides']}


        folder=HERE/('stress_runs' if args.phase=='stress' else 'runs')/f'{strategy}_rep{rep}_exec{execution}_{scenario}'
        try:
            value=execute(folder,strategy,rep,cfg,scenario=scenario)
            value.update({'execution':execution,'scenario':scenario,'original_run_id':old.run_id})
            rows.append(value)
            print('A2',args.phase,strategy,rep,execution,scenario,'g=%.9f'%value['co2e_g'],'timeouts',value['solver_timeout_count'],'wall=%.2f'%value['runtime_s'],flush=True)
        except Exception as exc:
            errors.append({'strategy':strategy,'rep':rep,'execution':execution,'scenario':scenario,'error':repr(exc),'traceback':traceback.format_exc()})
            save(HERE/'audit'/('errors_'+args.phase+'.json'),errors)
            print('A2 EXECUTION ERROR',errors[-1],flush=True)
        pd.DataFrame(rows).to_csv(HERE/('executions.csv' if args.phase=='full' else 'stress_executions.csv'),index=False)
    if args.phase=='full':
        consistency=compare_repeats(rows)
        checks=[{'item':'existing_25_plus_20_boundary_tests','status':'PASS','details':dict(suite.attrib)},
            {'item':'36_full_trace_executions_completed','status':'PASS' if len(rows)==36 and not errors else 'FAIL','details':{'completed':len(rows),'errors':errors}},
            {'item':'same_environment_triple_repeat','status':'PASS' if len(consistency)==12 and all(c['status']=='PASS' for c in consistency) else 'FAIL','details':consistency},
            {'item':'physical_capacity_conservation_energy_causality','status':'PASS' if len(rows)==36 and all(x['invariants_passed'] for x in rows) else 'FAIL'},
            {'item':'no_timeout_binds_new_protocol','status':'PASS' if len(rows)==36 and all(x['solver_timeout_count']==0 for x in rows) else 'FAIL','details':{'timeouts':sum(x['solver_timeout_count'] for x in rows)}},
            {'item':'real_hardware_and_operational_asof_carbon','status':'NOT VERIFIED'},
            {'item':'unrestricted_global_Oracle_optimum','status':'NOT VERIFIED'},
            {'item':'native_HiGHS_thread_seed_work_limit_control','status':'NOT VERIFIED','details':'Unavailable/ineffective; ample-budget empirical protocol used'}]
        gate=all(x['status']!='FAIL' for x in checks)
        save(HERE/'audit/preliminary_gate.json',{'A2':'PASS' if gate else 'FAIL','checks':checks,'B_C_D':'NOT STARTED until protection and change explanation checks complete'})
        print('A2 NUMERIC/LOGIC GATE',gate,flush=True)

if __name__=='__main__':main()
