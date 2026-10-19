import argparse
import os
from pathlib import Path
import sys
import time
import traceback

sys.dont_write_bytecode = True
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from task4_core.sources import read, sha
from task4_core.serialization import digest, save
from task4_core.protocol import core_plan, pilot_config, PROTOCOL_VERSION
from task4_core.world import Scenario, ScenarioBuilder
from task4_core.runner import Task4Runner
from task4_core.registry import reusable
from task4_core.validation import validate_execution
from task4_core.analysis import paired_rows, rebound_rows


def pilot_contract(config):
    if config.get('formal_enabled') is not False:
        raise ValueError('Formal gate must explicitly remain false')
    if config['policies'] != ['Greedy','Spatial_MILP']:
        raise ValueError('Pilot compares the two declared main policies')
    seeds = config['seeds']
    if not 1 <= len(seeds) <= 3 or len(set(seeds)) != len(seeds) or not all(isinstance(s,int) for s in seeds):
        raise ValueError('Pilot requires <=3 distinct integer seeds')
    scenarios = config['scenarios']
    if not 1 <= len(scenarios) <= 8 or len({s['name'] for s in scenarios}) != len(scenarios):
        raise ValueError('Pilot requires <=8 uniquely named scenarios')
    for row in scenarios:
        s = Scenario(**row['parameters'])
        s.validate()
        if s.n + int(s.n*s.growth) > 1000:
            raise ValueError('Pilot maximum 1000 instances after growth')
        if s.frozen_shock_seed is not None:
            raise ValueError('New pilot uses independent reserved seeds, not frozen compatibility shocks')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['dry-run','pilot','validate'])
    parser.add_argument('--config',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    if ROOT not in out.parents:
        raise ValueError('Output must be inside TASK 4')
    if args.mode == 'validate':
        reports = []
        for path in sorted((out/'executions').glob('*/attempt_*/record.json')):
            record = read(path)
            if record['status'] == 'COMPLETE_VALIDATED':
                if not reusable(record,record['config']):
                    reports.append({'record':str(path),'status':'HASH_FAILURE'})
                else:
                    try:
                        reports.append({'record':str(path),**validate_execution(path.parent)})
                    except Exception as exc:
                        reports.append({'record':str(path),'status':'VALIDATION_FAILURE','exception':repr(exc)})
            else:
                reports.append({'record':str(path),'status':record['status']})


        groups = {}
        for report in reports:
            signature = read(Path(report['record'])).get('signature',report['record'])
            groups.setdefault(signature,[]).append(report['status'])
        usable = sum('PASS' in statuses for statuses in groups.values())
        expected = None
        if (out/'pilot_config.json').exists():
            config = read(out/'pilot_config.json')
            expected = len(config['scenarios'])*len(config['seeds'])*2
        status = 'PASS' if groups and usable==len(groups) and (expected is None or usable==expected) else 'FAIL_OR_INCOMPLETE'
        validation = out/'validations'/str(time.time_ns())
        validation.mkdir(parents=True)
        save(validation/'results.json',{'status':status,'records':reports,'usable_signatures':usable,
                                       'expected_pilot_executions':expected,'historical_failures_retained':True})
        print(status, len(reports), str(validation),flush=True)
        return 0 if status == 'PASS' else 1
    if args.mode == 'dry-run':
        if out.exists():
            raise ValueError('Dry-run requires a new output directory')
        out.mkdir(parents=True)
        rows = core_plan()
        save(out/'core_plan.json',{'protocol':PROTOCOL_VERSION,'status':'PLAN_ONLY_NOT_EXECUTED',
                                  'formal_enabled':False,'execution_references':len(rows),'cases':rows})
        config = read(args.config) if args.config else pilot_config()
        pilot_contract(config)
        save(out/'pilot_config.json',config)
        save(out/'summary.json',{'protocol':PROTOCOL_VERSION,'formal_executions':0,'planned_core_references':len(rows),
                                 'nominal_unique_case_keys_not_yet_execution_hash_validated':len({digest({k:r[k] for k in ['scenario','seed','policy']}) for r in rows}),
                                 'pilot_worlds':len(config['scenarios'])*len(config['seeds']),
                                 'pilot_executions':2*len(config['scenarios'])*len(config['seeds'])})
        print('DRY_RUN_ONLY',len(rows),'Core references; no DES/inference',flush=True)
        return 0
    config = read(args.config) if args.config else pilot_config()
    pilot_contract(config)
    if out.exists() and not args.resume:
        raise ValueError('Existing output preserved; use --resume for exact configuration')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'pilot_config.json').exists():
        if digest(read(out/'pilot_config.json')) != digest(config):
            raise ValueError('Resume configuration differs')
    else:
        save(out/'pilot_config.json',config)
    began = time.perf_counter()
    builder = ScenarioBuilder()
    initialization_wall = time.perf_counter()-began
    save(out/'source_hashes_before.json',builder.source_hashes)
    runner = Task4Runner(out)
    records, worlds = [], []
    for row in config['scenarios']:
        for seed in config['seeds']:
            start = time.perf_counter()
            ref = {'scenario':row['name'],'seed':seed,'phase':'PILOT_NOT_FORMAL'}
            try:
                world = builder.build(Scenario(**row['parameters']),seed)
                generation_wall = time.perf_counter()-start
                for policy in config['policies']:
                    record = runner.execute(world,policy,reference=ref)
                    records.append(record)
                    print(row['name'],seed,policy,record['status'],'REUSED' if record.get('reused') else 'NEW',flush=True)
                worlds.append({**ref,'world_id':world.world_id,'generation_wall_s':generation_wall,
                               'pair_wall_s_including_generation_reuse_checks':time.perf_counter()-start})
            except Exception as exc:
                attempt = out/'exceptions'/str(time.time_ns())
                attempt.mkdir(parents=True)
                save(attempt/'failure.json',{**ref,'exception':repr(exc),'traceback':traceback.format_exc()})
                worlds.append({**ref,'status':'FAILED','error_directory':str(attempt),
                               'pair_wall_s_including_generation_reuse_checks':time.perf_counter()-start})

    all_complete = len(records)==len(config['scenarios'])*len(config['seeds'])*2 and all(
        r['status']=='COMPLETE_VALIDATED' for r in records)
    if all_complete:
        frame,pairs = paired_rows(records)
        frame.to_csv(out/'pilot_metrics.csv',index=False)
        pairs.to_csv(out/'paired_metrics.csv',index=False)

        normal_ids = {w['world_id'] for w in worlds if w.get('scenario')=='normal'}
        growth_ids = {w['world_id'] for w in worlds if w.get('scenario')=='growth_hosts'}
        rebound_rows(frame[frame.world_id.isin(normal_ids|growth_ids)]).to_csv(out/'growth_diagnostic.csv',index=False)
    changed = [p for p,h in builder.source_hashes.items() if sha(p)!=h]
    snapshot = out/'summaries'/str(time.time_ns())
    snapshot.mkdir(parents=True)
    summary = {'status':'PASS' if all_complete and not changed else 'FAIL_OR_INCOMPLETE','phase':'PILOT_ONLY',
               'formal_M1_M8_executions':0,'training_executed':False,'worlds':worlds,
               'records':[str(Path(r['folder'])/'record.json') for r in records],
               'source_changes':changed,'initialization_wall_s':initialization_wall,'wall_s':time.perf_counter()-began,
               'not_evidence_for_tail_or_superiority':True}
    save(snapshot/'pilot_summary.json',summary)
    print(summary['status'], str(snapshot),flush=True)
    return 0 if summary['status']=='PASS' else 1


if __name__=='__main__':
    sys.exit(main())
