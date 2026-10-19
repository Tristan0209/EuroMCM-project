import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import sys
import time
sys.dont_write_bytecode = True
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE),str(ROOT)]
import numpy as np
import pandas as pd
from experiment import scenarios,config,SEEDS,POLICIES,DRY_SEED,N
from task4_core.sources import read,sha,RELEASE,BASE
from task4_core.serialization import save,digest
from task4_core.world import ScenarioBuilder,ExogenousWorld
from task4_core.registry import reusable
from task4_core.validation import validate_execution
from task4_core.window import WindowedSpatial,SPATIAL,SOURCE_IDENTITIES
from simple_runner import SimpleRunner


def preflight():
    usage = shutil.disk_usage(HERE)

    planned_instances = (3*7335+11002)*3
    estimated = planned_instances*3819.301625
    budget = max(2*estimated,2*1024**3)
    if usage.free < budget+2*1024**3:
        raise OSError('Insufficient disk reserve for this 24-execution study')
    audit = HERE/'audit'
    report = {'status':'PASS','disk_free_bytes':usage.free,'linear_bytes_p95':estimated,
              'reserved_study_bytes':budget,'additional_free_floor_bytes':2*1024**3,
              'estimated_serial_minutes':12*46.9264830573*1.125/60,
              'no_C1_C8_runs':True,'strategy_execution_limit':24,'max_instances':11002}
    save(audit/f'preflight_{time.time_ns()}.json',report)
    print(json.dumps(report,indent=2),flush=True)


def inspect_reuse(builder,runner):
    paths = list((ROOT/'runs').glob('*/executions/*/attempt_*/record.json'))
    target_ids = set(SEEDS)
    candidates = []
    for path in paths:
        rec = read(path)
        if rec.get('status')!='COMPLETE_VALIDATED':
            continue
        world = read(path.parents[3]/'worlds'/rec['world_id']/'world.json')
        if len(world['instances'])!=N or world['seed'] not in target_ids:
            continue
        candidates.append(str(path))
    historic = pd.read_csv(RELEASE/'FINAL_COMPARISON.csv')
    historic_seeds = sorted(int(s) for s in historic.seed.unique())
    assert not set(SEEDS)&set(historic_seeds)
    assert not candidates, 'Unexpected existing same-seed baseline: inspect complete signature before running'
    save(HERE/'audit/baseline_reuse_audit.json',{
        'status':'NO_EXACT_MATCH','existing_task4_complete_records_examined':len(paths),
        'new_baseline_seeds':SEEDS,'historic_task3_seeds':historic_seeds,'same_seed_full_N_candidates':candidates,
        'reason':'Task4 Pilot is 800/1000 instances and different seeds; Task3 frozen shocks use different seeds/worlds and old recording contract. No identical baseline is re-executed.',
        'historical_baseline_reused_as_new_samples':False,'historical_reference_csv':str(RELEASE/'FINAL_COMPARISON.csv')})


def run(dry=False):
    preflight()
    folder = HERE/('reproduced_dry_run' if dry else 'reproduced_runs')
    folder.mkdir(exist_ok=True)
    if not dry:
        assert read(HERE/'reproduced_dry_run/completion.json')['status']=='PASS','Small dry-run must pass first'
    protocol = config()
    cfg_path = HERE/'protocol.json'
    if cfg_path.exists():
        assert digest(read(cfg_path))==digest(protocol)
    else:
        save(cfg_path,protocol)
    builder = ScenarioBuilder()
    runner = SimpleRunner(folder)
    if not dry and not (HERE/'audit/baseline_reuse_audit.json').exists():
        inspect_reuse(builder,runner)
    save(folder/f'source_before_{time.time_ns()}.json',builder.source_hashes)
    rows = []
    zero_reference = None
    for seed in ([DRY_SEED] if dry else SEEDS):
        base = builder.build(scenarios(80 if dry else N)['Baseline'],seed)
        jobs,route = base.routed_requests(runner.router)
        canonical = builder.raw.set_index('request_id')
        assert (route.set_index('request_id').selected_model==canonical.loc[route.request_id].selected_model).all()
        for name,params in scenarios(80 if dry else N).items():
            world = base if name=='Baseline' else builder.build(params,seed)
            again = builder.build(params,seed)
            assert world.world_id==again.world_id,'Identical seed must reproduce all arrays'
            if name in ['Prediction Error','Demand Growth']:
                original = base.instances.set_index('instance_id')
                shared = world.instances.set_index('instance_id').loc[original.index]
                pd.testing.assert_frame_equal(shared,original,check_exact=True)
                for i in original.index:
                    assert world.service_shocks[i]==base.service_shocks[i]
            _,this_route = world.routed_requests(runner.router)
            fallback = this_route.task_type.isin(['Code','Math'])
            assert this_route.loc[fallback,'selected_model'].eq('L').all()
            for policy in POLICIES:
                rec = runner.execute(world,policy,{'study':'simple_v1','scenario':name,'seed':seed,'dry_run':dry})
                rows.append({'scenario':name,'seed':seed,'policy':policy,'world_id':world.world_id,
                             'record':str(Path(rec['folder'])/'record.json'),'status':rec['status'],'reused':rec.get('reused',False)})
                save(folder/f'progress_{time.time_ns()}.json',rows)
                print(name,seed,policy,rec['status'],'REUSED' if rec.get('reused') else 'NEW',flush=True)
                if rec['status']!='COMPLETE_VALIDATED':
                    raise RuntimeError('Failed execution retained: '+rec['folder'])
                world.verify()
    changed = [p for p,h in builder.source_hashes.items() if sha(p)!=h]
    assert not changed,changed
    save(folder/'completion.json',{'status':'PASS','dry_run':dry,'rows':rows,'source_changes':changed,
         'C1_C8_executions':0,'simple_new_executions':sum(not r['reused'] for r in rows),
         'unique_worlds':len({r['world_id'] for r in rows}),'upstream_training':0})


def validate():
    completion = read(HERE/'reproduced_runs/completion.json')
    assert len(completion['rows'])==24
    result = []
    for row in completion['rows']:
        rec = read(row['record'])
        assert reusable(rec,rec['config'])
        checks = validate_execution(Path(row['record']).parent)
        result.append({**row,**checks})
        print('VALIDATED',row['scenario'],row['seed'],row['policy'],flush=True)
    save(HERE/'audit'/f'independent_validation_{time.time_ns()}.json',{'status':'PASS','executions':24,'checks':result})


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=['preflight','dry-run','run','validate'])
    args = parser.parse_args()
    if args.mode=='preflight':preflight()
    elif args.mode=='validate':validate()
    else:run(args.mode=='dry-run')
