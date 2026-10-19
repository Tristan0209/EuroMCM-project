from context import *
import argparse,copy,gzip,time,traceback
from functools import lru_cache
import numpy as np
import pandas as pd
from src.domain import load_inputs
from src.carbon import Carbon
from src.accounting import evaluate
from run_a2 import audit_extended
from v2.des import Simulation
from policies import NestedSimulation
from spatial import SpatialSimulation

@lru_cache(None)
def inputs(n):return load_inputs(n)
@lru_cache(None)
def carbon():
    anchor=pd.Timestamp('2025-01-02T00:00:00Z')
    f=pd.read_csv(BASE/'results/data/carbon_lifecycle.csv');times=pd.to_datetime(f.datetime_utc,utc=True)
    f=f[(times>=anchor-pd.Timedelta(hours=24))&(times<anchor+pd.Timedelta(hours=13))]
    return Carbon(f,anchor,read(BASE/'results/data/carbon_lifecycle.metadata.json'))
class NormalAvailability:
    def __init__(self,servers):self.servers=servers;self.events=[(0.,r,m,int(p['capacity'])) for (r,m),p in servers.items()]
    def at(self,r,m,t):return self.servers[(r,m)]['capacity']

def run_case(policy,seed,n=7335,gamma=1.35,delay_scale=0.):
    cfg={**read(BASE/'configs/core.json')['model'],**read(A2/'configs/solver_protocol_frozen.json')['model_overrides']}
    cfg.update(service_guard=gamma,holding_seconds=read(ROOT/'configs/protocol.json')['preserved']['holding_seconds'])
    if policy=='E-Spatial_single_branch':cfg={**cfg,'holding_seconds':[0.]}
    signature=read(ROOT/'audit/source_and_input_hashes.json')
    case={'policy':policy,'seed':seed,'n':n,'gamma':gamma,'delay_scale':delay_scale,'model':cfg,'provenance_digest':digest(signature)}
    key=digest(case)[:20];folder=ROOT/'runs'/key
    if (folder/'metrics.json').is_file():return read(folder/'metrics.json')
    assert not folder.exists(),'Incomplete run preserved; inspect and use an explicit new version before retry: '+str(folder)
    folder.mkdir(parents=True);save(folder/'config.json',case);began=time.perf_counter()
    try:
        jobs,servers,net,_=inputs(n);av=NormalAvailability(servers)
        full_shocks=read(EXT/'common_shocks'/('seed_'+str(seed)+'.json'))
        shocks={q.request_id:full_shocks[q.request_id] for q in jobs}
        save(folder/'input_ids.json',[q.request_id for q in jobs]);save(folder/'common_random_conditions.json',shocks)
        if policy in ['E-Spatial','E-Full','E-Spatial_single_branch']:
            if policy=='E-Spatial_single_branch':cfg={**cfg,'holding_seconds':[0.]}
            engine_policy='E-Spatial' if policy=='E-Spatial_single_branch' else policy
            sim=NestedSimulation(jobs,servers,net,av,carbon(),shocks,'MPC',cfg,policy=engine_policy,delay_scale=delay_scale).run()
        elif policy=='Spatial_MILP':
            assert delay_scale==0.
            sim=SpatialSimulation(jobs,servers,net,av,carbon(),shocks,'MPC',cfg).run()
        else:
            assert delay_scale==0.,'Greedy has no MILP wall overhead in this controlled comparison'
            sim=Simulation(jobs,servers,net,av,carbon(),shocks,policy,cfg).run()

        quality=pd.read_csv(BASE/'results/data/evaluation_only_historical_quality.csv').set_index('request_id').q.to_dict()
        value,records,ledger=evaluate(sim,43200,quality)

        records.to_csv(folder/'requests.csv.gz',index=False,compression='gzip')
        ledger.to_csv(folder/'energy_ledger.csv.gz',index=False,compression='gzip')
        pd.DataFrame(sim.log).to_csv(folder/'events.csv.gz',index=False,compression='gzip')
        with gzip.open(folder/'solver_log.jsonl.gz','wt',encoding='utf-8') as stream:
            for row in sim.solver_log:stream.write(json.dumps(row,default=lambda x:x.item())+'\n')
        audit_extended(sim,records,ledger)
        assert abs(value['idle_kwh']-250.56)<1e-9
        assert abs(value['active_increment_kwh']-records.actual_service_s.fillna(0).sum()*585*1.2/3.6e6)<1e-9
        assert abs(value['controller_kwh']-sum(b-a for a,b,_,_ in sim.controls)*12/3.6e6)<1e-9
        assert len(records)==n and records.request_id.is_unique
        assert all(log['max_request_arrival_s']<=log['decision_s']+1e-8 and log['observed_carbon_end_s']<=log['decision_s']+1e-8 for log in sim.solver_log)
        attribution=ledger[ledger.request_id.notna()].groupby(['request_id','component'])[['energy_kwh','co2e_g']].sum().unstack('component',fill_value=0)
        attribution.columns=['_'.join(c) for c in attribution.columns]
        attribution=records[['request_id','status','service_constraint_violation','historical_quality']].set_index('request_id').join(attribution).fillna(0)
        attribution.to_csv(folder/'request_attribution.csv.gz',compression='gzip')
        value.update(strategy=policy,seed=seed,n=n,gamma=gamma,delay_scale=delay_scale,run_id=key,
            artifact_directory=str(folder),provenance='NEW_EXECUTION',all_invariants_passed=True,
            runtime_seconds=time.perf_counter()-began,input_ids_digest=digest([q.request_id for q in jobs]),
            common_shocks_digest=digest(shocks),mean_solver_seconds=value['solver_wall_seconds']/max(1,value['solver_calls']),
            incremental_co2e_g=value['co2e_g']-value['idle_co2e_g'],
            milp_unassigned_total=sum(log.get('unassigned_count',0) for log in sim.solver_log),
            post_compute_rejections=sum(len(log.get('post_compute_rejected_ids',[])) for log in sim.solver_log),
            subproblem_timeout_count=sum(log.get('subproblem_timeout_count',0) for log in sim.solver_log),
            policy_hold_mean_s=float(np.mean([log.get('holding_s',0.) for log in sim.solver_log])) if sim.solver_log else 0.,
            nonzero_hold_decisions=sum(log.get('holding_s',0)>0 for log in sim.solver_log))
        branches=[b for log in sim.solver_log for b in log.get('branches',[log])]
        recoveries=[b['recovery'] for b in branches if 'recovery' in b]
        for name in ['illegal_rounded_candidate','attempted','recovered','unrecoverable']:
            value['recovery_'+name+'_count']=sum(bool(r[name]) for r in recoveries)
        value.update(status4_count=sum(b['status']==4 for b in branches),
            native_timeout_count=sum(b['status']==1 for b in branches),
            actual_native_milp_calls=sum(b['engine'].startswith('HiGHS') for b in branches),
            recovery_lp_wall_seconds=sum(r.get('lp_wall_seconds',0.) for r in recoveries),
            recovery_lp_iterations=sum(r.get('lp_iterations') or 0 for r in recoveries),
            controller_computing_seconds=sum(b-a for a,b,_,_ in sim.controls),
            recovery_scope='ALL subproblems, including unused E branches; explicit fallback is top-level DES count')
        value['terminal_digest']=digest(records.sort_values('request_id').to_csv(index=False,float_format='%.12g'))
        save(folder/'metrics.json',value)
        print('FINAL',policy,'n',n,'seed',seed,'gamma',gamma,'scale',delay_scale,'C %.6f'%value['co2e_g'],'wall %.2f'%value['runtime_seconds'],flush=True)
        return value
    except Exception as exc:
        save(folder/'failure.json',{'exception':repr(exc),'traceback':traceback.format_exc()});raise

def group(phase,cases,previous=None):
    target=out(phase);rows=[] if previous is None else previous.to_dict('records')
    for kwargs in cases:
        row=run_case(**kwargs);rows.append(row)
        pd.DataFrame(rows).to_csv(target/'raw_runs.csv',index=False)
    if not cases:pd.DataFrame(rows).to_csv(target/'raw_runs.csv',index=False)
    return pd.DataFrame(rows)

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['pilot','core','F','H'],required=True);a=p.parse_args()
    assert read(ROOT/'audit/preflight_gate.json')['status']=='PASS'
    current=signature();frozen=read(ROOT/'audit/source_and_input_hashes.json')
    assert all(current.get(k)==v for k,v in frozen.items()),'Frozen production source/input changed'
    if a.phase=='pilot':
        cases=[dict(policy=s,seed=SEEDS[0],n=1000) for s in ['Greedy','Spatial_MILP','MPC','E-Spatial','E-Full']]
    elif a.phase=='core':
        cases=[dict(policy=s,seed=seed) for seed in SEEDS for s in ['Greedy','Spatial_MILP','MPC','E-Spatial','E-Full']]
    elif a.phase=='F':
        assert (out('core')/'raw_runs.csv').exists()
        cases=[dict(policy=s,seed=seed,gamma=g) for g in [1.2,1.35,1.5,1.75] for seed in SEEDS[:6] for s in ['Greedy','E-Spatial','E-Full']]
    else:
        assert len(pd.read_csv(out('core')/'raw_runs.csv'))==120
        cases=[dict(policy=s,seed=seed,delay_scale=k) for k in [0.,.5,1.,2.,5.] for seed in SEEDS[:6] for s in ['E-Spatial','E-Full']]
        cases += [dict(policy='E-Spatial_single_branch',seed=seed,delay_scale=k) for k in [1.,5.] for seed in SEEDS[:6]]
    group(a.phase,cases)
if __name__=='__main__':main()
