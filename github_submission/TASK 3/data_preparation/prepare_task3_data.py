import platform
from pathlib import Path

import numpy as np
import pandas as pd

from common import (HERE, PROJECT, OUT, MIDS, paths, read_json, save_json, digest,
                    requests_source, decisions_source, training_lengths, service_estimates, utc_strings)
from data_extensions import build_extensions


def main():
    OUT.mkdir(exist_ok=True)
    src = paths()
    cfg = read_json(HERE / 'config.json')
    if cfg['start_datetime_utc'] != '2024-05-10T00:00:00Z':
        raise ValueError('Preserve the 2024 baseline anchor; use align_time.py for carbon-covered dates')
    save_json(OUT/'effective_config.json',cfg)

    protected = sorted(set(p for root in [PROJECT/'task2_complete', PROJECT/'Work 1']
                           for p in root.rglob('*') if p.is_file()))
    protected += [src['routerbench'], src['azure']]
    before = {str(p.relative_to(PROJECT)): digest(p) for p in protected}
    save_json(OUT / 'source_integrity.json', {'before': before, 'after': None})
    inventory = [{'path': str(p.relative_to(PROJECT)), 'bytes': p.stat().st_size}
                 for p in PROJECT.rglob('*') if p.is_file() and '.git' not in p.parts
                 and HERE not in p.parents]
    save_json(OUT / 'project_inventory.json', {'files': inventory,
        'requested_document_found': False,
        'equivalent_complete_document': str((src['run']/'RESULTS_zh.md').relative_to(PROJECT)),
        'note': 'Exact filename 第二问完整求解过程.md absent; RESULTS_zh.md is the full 14-section solution report.'})
    frame = requests_source()
    splits = pd.read_csv(src['run']/'splits.csv')
    decisions = decisions_source()
    ids = splits[splits.phase.eq(cfg['source_phase'])].request_id
    assert len(ids) == len(decisions) and set(ids) == set(decisions.request_id)
    assert not decisions.upgraded.any() and (decisions.initial_l == decisions.final_l).all()
    f = frame.set_index('request_id').loc[ids].reset_index()
    d = decisions.set_index('request_id').loc[ids]
    f['selected_model'] = np.where(d.final_l.to_numpy(), 'L', 'S')
    pre_variant = read_json(src['run']/'pre_selection.json')['winner']
    pre = pd.read_csv(src['run']/'pre_predictions.csv.gz', usecols=['request_id', 'model_id', pre_variant, 'fit_id'])
    for a, mid in MIDS.items():
        p = pre[pre.model_id.eq(mid)].set_index('request_id')
        f['q_hat_' + a] = f.request_id.map(p[pre_variant])
        assert f.request_id.map(p.fit_id).eq('Pre_full_train').all()
    stats = training_lengths(frame)
    stats.to_csv(OUT/'training_length_statistics.csv', index=False)
    profile = read_json(src['resource_config'])['profile']

    f = f.iloc[np.random.default_rng(cfg['arrival_permutation_seed']).permutation(len(f))].reset_index(drop=True)
    service = service_estimates(f, stats, cfg, profile)
    azure = pd.read_csv(src['azure'], usecols=['arrival_s'], nrows=len(f)).arrival_s.to_numpy()
    assert len(azure) == len(f) and np.isfinite(azure).all() and (np.diff(azure) >= 0).all()
    gaps = azure - azure[0]
    assert gaps[-1] > 0


    span = (len(f)-1)*service['L']['total'].mean()/cfg['reference_utilization']/cfg['arrival_multiplier']
    arrival = pd.Timestamp(cfg['start_datetime_utc']) + pd.to_timedelta(gaps/gaps[-1]*span, unit='s')
    classes = cfg['service_classes']
    proportions = np.array([c['proportion'] for c in classes], dtype=float)
    assert np.isclose(proportions.sum(), 1) and (proportions >= 0).all()
    counts = np.floor(proportions*len(f)).astype(int)
    remainder_order = np.argsort(-(proportions*len(f)-counts), kind='stable')
    for j in remainder_order[:len(f)-counts.sum()]:
        counts[j] += 1
    assignments = np.repeat(np.arange(len(classes)), counts)
    np.random.default_rng(cfg['service_class_seed']).shuffle(assignments)
    f['service_class'] = [classes[j]['name'] for j in assignments]
    f['max_delay_seconds'] = [classes[j]['max_delay_seconds'] for j in assignments]
    f['priority'] = [classes[j]['priority'] for j in assignments]
    f['arrival_time'] = utc_strings(arrival)
    f['deadline'] = utc_strings(arrival + pd.to_timedelta(f.max_delay_seconds, unit='s'))
    f['origin_region'] = np.random.default_rng(cfg['origin_region_seed']).choice(cfg['regions'], len(f))
    f['source_partition'] = cfg['source_phase']
    f['input_tokens_proxy'] = f.input_tokens_proxy.astype(int)
    for a in ['S', 'L']:
        f['estimated_service_time_' + a + '_seconds'] = service[a]['total']
    f['estimated_output_tokens_proxy'] = np.where(f.selected_model.eq('S'), service['S']['output'], service['L']['output'])
    f['estimated_ttft_seconds'] = np.where(f.selected_model.eq('S'), service['S']['first'], service['L']['first'])
    f['estimated_service_time_seconds'] = np.where(f.selected_model.eq('S'), service['S']['total'], service['L']['total'])

    f['estimated_energy_kwh'] = f.estimated_service_time_seconds*(profile['busy_w']-profile['idle_w'])*profile['pue']/3.6e6
    keep = ['request_id','task_type','selected_model','arrival_time','deadline','max_delay_seconds','priority',
            'estimated_service_time_seconds','estimated_energy_kwh','q_hat_S','q_hat_L','service_class',
            'origin_region','source_partition','eval_name','source_row','prompt_chars','input_tokens_proxy',
            'estimated_output_tokens_proxy','estimated_ttft_seconds','estimated_service_time_S_seconds','estimated_service_time_L_seconds']
    f[keep].to_csv(OUT/'requests.csv', index=False)
    servers = pd.DataFrame([{'region':r, 'model_type':a, 'capacity':cfg['replicas_per_region_per_model'],
        'p_idle_w':profile['idle_w'], 'p_busy_w':profile['busy_w'], 'pue':profile['pue'],
        'service_time_multiplier':1 if a=='S' else cfg['large_service_multiplier'], 'availability':1.0,
        'parameter_source':'power/PUE/multipliers inherited_task2_assumed; regional capacity task3_assumed',
        'capacity_unit':'independent_nonbatched_replicas'} for r in cfg['regions'] for a in ['S','L']])
    servers.to_csv(OUT/'server_config.csv', index=False)
    availability = [{'datetime_utc':cfg['start_datetime_utc'], 'region':r.region, 'model_type':r.model_type,
                     'available_capacity':int(r.capacity), 'provenance':'task3_assumed_static_initial'} for r in servers.itertuples()]
    availability += [{**row, 'provenance':'task3_assumed_override'} for row in cfg['availability_overrides']]
    pd.DataFrame(availability).to_csv(OUT/'server_availability.csv', index=False)
    net = pd.DataFrame([{'origin_region':a, 'destination_region':b,
        'latency_ms':cfg['network_assumed_latency_ms'][a+'-'+b],
        'transfer_energy_kwh':cfg['network_assumed_transfer_energy_kwh']['local' if a==b else 'cross_region'],
        'allowed':int([a,b] not in cfg['network_blocked_pairs']), 'provenance':'assumed',
        'energy_scope':'per_request_fixed_transfer_no_measured_payload'} for a in cfg['regions'] for b in cfg['regions']])
    net.to_csv(OUT/'network_config.csv', index=False)
    precheck_summaries=build_extensions(OUT,f,servers,net,cfg)
    shocks = []
    for rep, seed in enumerate(cfg['service_shock_seeds']):
        sigma = profile['residual_log_sigma']
        shock = np.exp(np.random.default_rng(seed).normal(-0.5*sigma*sigma, sigma, len(f)))
        shocks.extend({'request_id':rid, 'rep':rep, 'seed':seed, 'service_multiplier_shock':float(value)}
                      for rid,value in zip(f.request_id,shock))
    pd.DataFrame(shocks).to_csv(OUT/'service_random_conditions.csv', index=False)
    old_resource = read_json(src['run']/'resource_assumptions.json')
    params = {
        'schema_version':2, 'scope':'data_preparation_only_no_scheduling_or_optimization',
        'historical_observations':{'routerbench':'historical prompts/IDs/split; no labels in requests',
            'azure':'only first N arrival_s gap shape; no RouterBench real arrival pairing',
            'train_output_lengths':'historical regex proxy lengths; aggregate task x model from train only'},
        'inherited_task2_assumptions':{'profile':profile,
            'ttft_formula':'x=input_tokens_proxy/1000; first=0.05+0.2*x+0.01*x*x',
            'decode_formula':'y=(estimated_output_tokens_proxy-1)/1000; decode=8*y+0.1*x*y+0.02*y*y',
            'service_formula':'(first+decode)*model_multiplier; excludes queue/network/controller',
            'large_service_multipliers':[1,2,3], 'primary_large_service_multiplier':3, 'small_service_multiplier':1,
            'residual_noise':{'distribution':'exp(N(-sigma^2/2,sigma^2))', 'sigma':profile['residual_log_sigma'],
                'mean':1,'same_shock_for':'TTFT/decode/S/L/regions/strategies of the same request and rep',
                'prediction_uses':'mean one; realized shock must not be exposed to scheduler before service',
                'simulator_only_file':'service_random_conditions.csv', 'seeds':cfg['service_shock_seeds']},
            'queue_assumption':'Task2 separate nonbatched FCFS S/L pools; no queues constructed here',
            'replicas_per_model_pool_task2':[1,2], 'controller':old_resource['controller'],
            'network_control_s':read_json(src['resource_config'])['network_control_s'],
            'arrival_multipliers':[0.5,1,2], 'reference_utilization_L_pool':0.6,
            'facility_energy_formula':'PUE/3.6e6 * (sum_pool(capacity*idle_w)*H + sum_jobs((busy_w-idle_w)*service_s)) + controller_kwh',
            'controller_energy_formula':'router_seconds*requests*incremental_w*PUE/3.6e6; final route has no Post checks',
            'source_code':str(src['resource_core'].relative_to(PROJECT)),
            'source_resource_scenarios':str((src['run']/'resource_scenarios.csv').relative_to(PROJECT))},
        'task3_new_assumptions':{'config':cfg, 'completion_deadline':True, 'priority_direction':'larger means more urgent',
            'classes_independent_of':'task/quality/actual output; shuffled exact quotas with separate RNG',
            'origin_region':'uniform independent simulated metadata, no historical origin inference',
            'regional_capacity':'1 replica per model per region; total 3 S + 3 L, versus Task2 1+1 primary; all strategies share identical resources',
            'availability_semantics':'piecewise constant from event timestamp, persists to next event; default no outage; no automatic pool increase',
            'prediction_method':'rounded mean output proxy by task/model from train only; plug into inherited equations, not a fitted Pre model',
            'prediction_limit':'plug-in estimate ignores output variance and is not a calibrated service-time distribution',
            'estimated_energy_scope':'marginal facility busy-over-idle kWh, excludes idle base/controller/network; do not use as total facility energy',
            'network_semantics':'one-way ingress transfer per request; fixed energy proxy; add inherited 0.05s control separately only once',
            'service_multiplier_semantics':'request estimate already includes S/L multiplier; regional multiplier is reference metadata, do not multiply twice',
            'arrival_span_seconds':float(span), 'task2_span_seconds_not_reused':old_resource['reference_arrival_span_s'],
            'arrival_span_compatibility':'recomputed from prospective L estimate, not current test output; same Azure shape and permutation as Task2',
            'absolute_time_anchor':'synthetic UTC date aligned with historical Azure day; configurable before real carbon experiments'},
        'routing':{'family':'Pre_direct','variant':pre_variant,'strategy':cfg['source_strategy'],
            'fallback_tasks':['Code','Math'], 'unknown_source_fallback':'L', 'no_retraining':True,
            'parameters_source':'selected_parameters.json / final_model_choice.json / cost_predictor.json'},
        'fair_comparison':{'request_sequence':'requests.csv identical bytes and order',
            'resources':'same server_config/network_config/server_availability and rep for every policy',
            'randomness':'lookup service_random_conditions by request_id,rep; never draw by processing order',
            'energy_window':'choose common observation horizon after all compared runs; charge all idle pools',
            'regional_expansion':'shared six replicas for every policy; no policy-exclusive servers'},
        'carbon':{'real_hourly_data_available':False,'annual_factor_from_task2_used':False,
            'actuals':'retrospective accounting only', 'decision_inputs':'forecasts with issued_at_utc <= decision time only',
            'coverage_required':'all DE/FR/ES hours from min arrival.floor(hour) to max completion deadline.floor(hour); later overruns need more coverage',
            'no_interpolation_or_synthetic_fallback':True},
        'sources':{k:str(v.relative_to(PROJECT)) for k,v in src.items()}}
    params['carbon']['real_hourly_data_available']=(OUT/cfg['carbon_filename']).is_file()
    params['carbon']['availability_means']='file_present_only; validation_report records schema and coverage validation'
    params['carbon']['schema']='CO2 -> carbon_intensity_gco2_kwh and gCO2/kWh; CO2e -> carbon_intensity_gco2e_kwh and gCO2e/kWh; mandatory matching species/methodology'
    params['task3_new_assumptions']['precheck']=precheck_summaries
    params['task3_new_assumptions']['availability_scenarios']='normal/local_failure/high_load_capacity_reduction; external assumed capacity loss, fixed physical fleet'
    params['task3_new_assumptions']['time_alignment']='baseline always 2024; align_time.py creates separate translated copy after real carbon validation'
    save_json(OUT/'resource_parameters.json', params)
    after = {str(p.relative_to(PROJECT)):digest(p) for p in protected}
    save_json(OUT/'source_integrity.json', {'before':before, 'after':after, 'unchanged':before==after})
    derivations = {
        'requests.csv':{'sources':['RouterBench IDs via paired_requests','splits.csv','decisions.csv.gz','pre_predictions.csv.gz','azure','training_length_statistics.csv'],
            'generation':'protected test/rep0 selection; saved A_Baseline q_hat; train-only output proxy -> inherited service; Azure gap mapping; independent simulated classes/origins'},
        'server_config.csv':{'sources':['resource_config','resource_assumptions.json','config.json'],
            'generation':'inherited assumed power/PUE/S-L multipliers; task3 assumed regional replication'},
        'server_availability.csv':{'sources':['server_config.csv','config.json'],
            'generation':'piecewise-constant initial capacities plus explicit optional overrides; default static'},
        'network_config.csv':{'sources':['config.json'],'generation':'all nine origin-destination pairs, all values assumed, allow/block interface'},
        'resource_parameters.json':{'sources':['resource_core','resource_config','resource_assumptions.json','config.json'],
            'generation':'source-attributed inherited equations and new assumptions; no carbon annual factor copied'},
        'training_length_statistics.csv':{'sources':['paired_requests train IDs','paired_outcomes train rows only'],
            'generation':'task/model mean historical output proxy rounded half-up, six groups'},
        'service_random_conditions.csv':{'sources':['config.json','resource_config'],
            'generation':'mean-one lognormal sigma=.15; three inherited seeds; simulator-only common random conditions'},
        'source_integrity.json':{'sources':['all Task1/Task2 files','routerbench','azure'],
            'generation':'SHA256 before/after export; validation rechecks current source files'},
        'project_inventory.json':{'sources':['project directory'],'generation':'read-only full file listing excluding .git and this new directory'},
        'interface_test_report.json':{'sources':['test_data_contracts.py'],'generation':'eight synthetic software tests, fixtures deleted after use'}
    }
    files = {}
    for p in sorted(OUT.iterdir()):
        if p.suffix not in {'.csv','.json'} or p.name in {'validation_report.json','data_manifest.json'} or p.name.startswith('carbon_'):
            continue
        files[p.name] = {'sha256':digest(p), 'rows':len(pd.read_csv(p)) if p.suffix=='.csv' else None,
                         **derivations.get(p.name,{'sources':['config.json','data_extensions.py','feasibility_precheck.py'],
                             'generation':'assumed independent availability scenarios, arithmetic contracts and necessary-condition checks; not a scheduling solution'})}
    save_json(OUT/'data_manifest.json', {'schema_version':2,'files':files,
        'source_paths':{k:str(v.relative_to(PROJECT)) for k,v in src.items()},
        'source_hashes':before, 'configuration':cfg, 'config_sha256':digest(HERE/'config.json'),
        'script_hashes':{p.name:digest(p) for p in HERE.glob('*.py')},
        'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__},
        'partition':{'exported':'test','requests':len(f),'full_source_counts':splits.phase.value_counts().to_dict(),
                     'models':f.selected_model.value_counts().to_dict(),'task_counts':f.task_type.value_counts().to_dict(),
                     'service_classes':f.service_class.value_counts().to_dict()},
        'units':{'all_datetimes':'ISO8601 explicit UTC Z or +00:00','delay/service':'seconds','latency_ms':'milliseconds',
                 'power':'W per independent replica','energy':'kWh','quality':'dimensionless benchmark Score prediction [0,1]',
                 'capacity':'independent simultaneous requests, one per replica','availability':'fraction [0,1]',
                 'lengths':'regex proxy counts (prompt_chars is character count)','carbon':'species-specific gCO2/kWh or gCO2e/kWh; matching column, metadata and methodology required'},
        'assumptions':'resource_parameters.json: inherited_task2_assumptions vs task3_new_assumptions',
        'exact_document_name_missing':True,'full_document_read':'task2_complete/runs/work1_integrated_20261016/RESULTS_zh.md',
        'carbon':{'status':'file_present_unvalidated' if (OUT/cfg['carbon_filename']).is_file() else 'missing_pending_user_file','expected_file':'output/carbon_intensity.csv',
                  'expected_metadata':'output/carbon_intensity.metadata.json','no_real_or_proxy_carbon_generated':True},
        'validation_report':{'generator':'validate_task3_data.py','self_hash_excluded':True}})
    print('Generated:', {k:v['rows'] for k,v in files.items()})


if __name__ == '__main__':
    main()
