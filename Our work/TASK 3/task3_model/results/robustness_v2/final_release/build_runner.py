from context import *
def main():
    s=(PREVIOUS/'experiments.py').read_text(encoding='utf-8')
    s=s.replace('from policies import NestedSimulation','from policies import NestedSimulation\nfrom spatial import SpatialSimulation')
    start=s.index('def old_rows():');end=s.index('def run_case(',start)
    s=s[:start]+s[end:]
    s=s.replace("read(ROOT/'configs/protocol.json')['E']['holding_seconds']","read(ROOT/'configs/protocol.json')['preserved']['holding_seconds']")
    s=s.replace("    signature=read(ROOT/'audit/source_and_input_hashes.json')","    if policy=='E-Spatial_single_branch':cfg={**cfg,'holding_seconds':[0.]}\n    signature=read(ROOT/'audit/source_and_input_hashes.json')")
    s=s.replace("log['holding_s'] for log", "log.get('holding_s',0.) for log")
    s=s.replace("if policy in ['E-Spatial','E-Full']:","if policy in ['E-Spatial','E-Full','E-Spatial_single_branch']:\n            if policy=='E-Spatial_single_branch':cfg={**cfg,'holding_seconds':[0.]}\n            engine_policy='E-Spatial' if policy=='E-Spatial_single_branch' else policy")
    s=s.replace('policy=policy,delay_scale=delay_scale','policy=engine_policy,delay_scale=delay_scale')
    s=s.replace("        else:\n            assert delay_scale==0.","        elif policy=='Spatial_MILP':\n            assert delay_scale==0.\n            sim=SpatialSimulation(jobs,servers,net,av,carbon(),shocks,'MPC',cfg).run()\n        else:\n            assert delay_scale==0.")
    marker="        value['terminal_digest']="
    extra='''        branches=[b for log in sim.solver_log for b in log.get('branches',[log])]
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
'''
    s=s.replace(marker,extra+marker)
    s=s[:s.index('def main():')]
    s+='''def main():
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
'''
    (ROOT/'experiments.py').write_text(s,encoding='utf-8')
if __name__=='__main__':main()
