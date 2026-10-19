import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
import numpy as np
import pandas as pd
from task4_core.sources import read
from task4_core.serialization import save


def bytes_at(path):
    return sum(p.stat().st_size for p in path.rglob('*') if p.is_file())


def main(out):
    run = ROOT/'runs/pilot_v3_001'
    summary_path = sorted((run/'summaries').glob('*/pilot_summary.json'))[0]
    summary = read(summary_path)
    if summary['status']!='PASS' or len(summary['worlds'])!=14:
        raise ValueError('Pilot completeness gate')
    rows,worlds = [],[]
    lookup = {w['world_id']:w for w in summary['worlds']}
    for path in summary['records']:
        rec = read(path)
        folder = Path(rec['folder'])
        bench,metrics = read(folder/'benchmark.json'),read(folder/'metrics.json')
        case = lookup[rec['world_id']]
        rows.append({**bench,'scenario':case['scenario'],'execution_directory_bytes':bytes_at(folder),
                     'native_highs_calls':metrics['highs_calls'],
                     'completed':metrics['completed'],'failed':metrics['failed'],'censored':metrics['censored'],
                     'energy_kwh':metrics['energy_kwh'],'co2e_g':metrics['co2e_g'],
                     'fallback_count':metrics['fallback_count'],'solver_timeout_count':metrics['solver_timeout_count'],
                     'numerical_rejection_count':metrics['observed_mechanism_counts']['milp_numerical_rejection'],
                     'model_selection_harm':metrics['model_selection_harm'],'end_to_end_harm':metrics['end_to_end_harm']})
    frame = pd.DataFrame(rows)
    for case in summary['worlds']:
        part = frame[frame.world_id.eq(case['world_id'])]
        n = int(part.requests.iloc[0])
        world_folder = run/'worlds'/case['world_id']
        size = bytes_at(world_folder)
        worlds.append({**case,'requests':n,'world_bytes':size,
                       'execution_bytes_pair':int(part.execution_directory_bytes.sum()),
                       'total_bytes':size+int(part.execution_directory_bytes.sum()),
                       'normalized_pair_s_per_7335':case['pair_wall_s_including_generation_reuse_checks']*7335/n,
                       'bytes_per_request_world_and_two_policies':(size+part.execution_directory_bytes.sum())/n})
    world_frame = pd.DataFrame(worlds)
    agg = frame.groupby(['scenario','policy']).agg(executions=('requests','size'),n=('requests','first'),
        simulation_mean_s=('simulation_wall_s','mean'),execution_mean_s=('execution_wall_s_before_final_manifest','mean'),
        solver_calls_mean=('solver_calls','mean'),native_highs_calls_mean=('native_highs_calls','mean'),solver_wall_mean_s=('solver_wall_s','mean'),
        peak_rss_max_bytes=('sampled_peak_rss_bytes','max'),files_mean_bytes=('execution_directory_bytes','mean'),
        completed_sum=('completed','sum'),failed_sum=('failed','sum'),censored_sum=('censored','sum'),
        fallback_sum=('fallback_count','sum'),timeouts_sum=('solver_timeout_count','sum'),numeric_rejects_sum=('numerical_rejection_count','sum')).reset_index()
    norm = world_frame.normalized_pair_s_per_7335
    pair_mean = float(norm.mean())
    pair_p95 = float(norm.quantile(.95))
    bpi = float(world_frame.bytes_per_request_world_and_two_policies.mean())
    bpi95 = float(world_frame.bytes_per_request_world_and_two_policies.quantile(.95))
    greedy = frame[frame.policy.eq('Greedy')]
    greedy_norm = float((greedy.execution_wall_s_before_final_manifest*7335/greedy.requests).mean())
    estimates = []
    for module,refs,pairs,load in [('C1',48,24,1),('C2',192,96,1),('C3',4000,2000,1),
                                  ('C4',120,60,1.75),('C6',192,96,1)]:
        estimates.append({'module':module,'execution_references':refs,'world_pairs':pairs,'mean_N_multiplier':load,
                          'linear_serial_hours':pairs*load*pair_mean/3600,
                          'planning_hours_2x_empirical_p95':pairs*load*pair_p95*2/3600,
                          'linear_disk_GiB':pairs*load*7335*bpi/2**30,
                          'disk_reserve_GiB_2x_p95':pairs*load*7335*bpi95*2/2**30,
                          'status':'ESTIMATE_ONLY_NOT_FORMAL_EXECUTION'})
    estimates.append({'module':'C8','execution_references':6,'world_pairs':None,
                      'linear_serial_hours':6*greedy_norm/3600,
                      'planning_hours_2x_empirical_p95':6*greedy_norm*4/3600,
                      'linear_disk_GiB':6*7335*bpi/2**30,
                      'disk_reserve_GiB_2x_p95':6*7335*bpi95*2/2**30,
                      'status':'UNMEASURED_ALL_L_USING_GREEDY_PROXY; storage conservatively includes full paired-world allowance'})
    extensions = []
    for name,refs,pairs,load in [('M1_grid',84,42,1),('M2_confirmation',96,48,1),('M4_boundary',192,96,2),
                                  ('M6_32points',512,256,1),('M6_seed_confirmation',128,64,1),('M8_AllL',100,50,1)]:
        extensions.append({'name':name,'additional_executions':refs,'linear_hours':pairs*load*pair_mean/3600,
                           'planning_hours_2x_p95':pairs*load*pair_p95*2/3600,
                           'linear_disk_GiB':pairs*load*7335*bpi/2**30,'assumed_N_multiplier':load})
    totals = {k:sum(e[k] for e in estimates) for k in ['linear_serial_hours','planning_hours_2x_empirical_p95',
                                                   'linear_disk_GiB','disk_reserve_GiB_2x_p95']}
    value = {'status':'MEASURED_PILOT_AND_LABELLED_EXTRAPOLATION','pilot_summary':str(summary_path),
        'world_count':14,'executions':28,'distinct_seeds_per_scenario':2,'formal_executions':0,
        'pilot_initialization_wall_s':summary['initialization_wall_s'],'pilot_wall_s':summary['wall_s'],
        'world_pair_mean_s':float(world_frame.pair_wall_s_including_generation_reuse_checks.mean()),
        'world_pair_p95_s':float(world_frame.pair_wall_s_including_generation_reuse_checks.quantile(.95)),
        'normalized_pair_mean_s_7335':pair_mean,'normalized_pair_p95_s_7335':pair_p95,
        'bytes_per_instance_paired_mean':bpi,'bytes_per_instance_paired_p95':bpi95,
        'sampled_max_rss_bytes':int(frame.sampled_peak_rss_bytes.max()),'measured_world_and_execution_bytes':int(world_frame.total_bytes.sum()),
        'original_summary_source_changes':summary['source_changes'],'core_estimates':estimates,'core_raw_reference_totals':totals,
        'extensions':extensions,'assumptions':['linear N extrapolation, not an upper bound or confidence interval',
            '14 paired worlds across 7 scenarios, only 2 independent seeds per scenario',
            'full N=7335 pressure MILP congestion may be nonlinear; 30s is per solve, not per world',
            'full logs and world JSON/CSV counted; failure counterexamples may add unbounded extra storage',
            'raw references retained; no discount until strict execution-key reuse',
            'C5 postprocessing adds no DES; C7 optional not included',
            'P95 from small Pilot is descriptive only; twice P95 is a planning factor, not a tail guarantee']}
    if out.exists():
        raise ValueError('New output directory required')
    out.mkdir(parents=True)
    frame.to_csv(out/'execution_measurements.csv',index=False)
    world_frame.to_csv(out/'world_measurements.csv',index=False)
    agg.to_csv(out/'scenario_policy_measurements.csv',index=False)
    save(out/'runtime_estimates_v3.json',value)
    print(json.dumps({k:v for k,v in value.items() if k not in ['core_estimates','extensions','assumptions']},indent=2))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    main(args.out.resolve())
