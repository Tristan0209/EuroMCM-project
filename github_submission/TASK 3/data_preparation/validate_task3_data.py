import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import (HERE, PROJECT, OUT, MIDS, paths, read_json, save_json, digest, requests_source,
                    decisions_source, training_lengths, service_estimates, frozen_route)
from carbon_loader import load_carbon, utc_column, assert_compatible_carbon, carbon_for_decision
from resource_contracts import service_at_pool, incremental_energy_kwh, facility_energy_components
from feasibility_precheck import precheck, capacity_intervals

DEFAULT_OUT = OUT


def validate(require_carbon=False, data_dir=None):
    global OUT
    OUT=Path(data_dir).resolve() if data_dir else DEFAULT_OUT
    if HERE not in OUT.parents:
        raise ValueError('Data directory must be inside data_preparation')
    cfg, src = read_json(OUT/'effective_config.json'), paths()
    checks = {}

    def check(name, fn):
        try:
            detail = fn()
            checks[name] = {'status':'passed','details':detail}
        except Exception as exc:
            checks[name] = {'status':'failed','details':str(exc)}

    def ensure(condition, detail):
        if not bool(condition):
            raise ValueError(detail)
        return detail

    f = pd.read_csv(OUT/'requests.csv')
    server = pd.read_csv(OUT/'server_config.csv')
    net = pd.read_csv(OUT/'network_config.csv')
    avail = pd.read_csv(OUT/'server_availability.csv')
    source = requests_source()
    splits = pd.read_csv(src['run']/'splits.csv')
    d = decisions_source().set_index('request_id')
    origin = source.set_index('request_id').loc[f.request_id].reset_index()
    variant = read_json(src['run']/'pre_selection.json')['winner']
    check('unique_request_ids',lambda:ensure(f.request_id.notna().all() and not f.request_id.duplicated().any(),str(len(f))+' unique IDs'))
    check('source_partition_conserved', lambda:ensure(
        set(f.request_id)==set(splits[splits.phase.eq(cfg['source_phase'])].request_id)
        and len(f)==len(d) and f.source_partition.eq('test').all()
        and (origin.task_type.to_numpy()==f.task_type.to_numpy()).all(),
        'Exact frozen test ID set/count/task metadata, no train/select/confirm in request export'))
    def raw_source():
        raw = pd.read_csv(src['routerbench'],usecols=['sample_id','eval_name'])
        assert not raw.sample_id.duplicated().any()
        selected = raw.iloc[f.source_row.to_numpy()].reset_index(drop=True)
        assert np.array_equal(selected.sample_id,f.request_id)
        assert np.array_equal(selected.eval_name,f.eval_name)
        assert len(source)==len(splits) and set(source.request_id)==set(splits.request_id)
        assert splits.groupby('group').phase.nunique().max()==1
        return {'raw_rows':len(raw),'paired_rows':len(source),'exported_test_rows':len(f),
                'ID_and_benchmark_match_original_source_row':True,'no_prompt_group_crosses_partition':True}
    check('RouterBench_source_row_and_group_partitions',raw_source)
    check('frozen_SL_routes',lambda:ensure(
        np.array_equal(f.selected_model,np.where(d.loc[f.request_id].final_l,'L','S'))
        and not d.upgraded.any() and (d.initial_l==d.final_l).all()
        and (d.calls_l+d.calls_s).eq(1).all(), 'Exact Selected_finite_sample_guard rep0 routing; '+str(f.selected_model.value_counts().to_dict())))
    check('Code_Math_fallback',lambda:ensure(f.loc[f.task_type.isin(['Code','Math']),'selected_model'].eq('L').all(),'Code/Math all L'))

    def quality_and_route():
        pre = pd.read_csv(src['run']/'pre_predictions.csv.gz', usecols=['request_id','model_id',variant,'fit_id'])
        for a,mid in MIDS.items():
            p = pre[pre.model_id.eq(mid)].set_index('request_id').loc[f.request_id]
            assert np.allclose(f['q_hat_'+a],p[variant],rtol=0,atol=1e-14)
            assert p.fit_id.eq('Pre_full_train').all()
        provenance = read_json(src['run']/'pre_provenance.json')['Pre_full_train']
        train_groups = set(splits[splits.phase.eq('train')].group)
        assert set(provenance['training_groups']) == train_groups
        assert train_groups.isdisjoint(set(splits[splits.phase.eq('test')].group))
        registered = set(source[source.split.eq('train')].eval_name)
        cost = read_json(src['run']/'cost_predictor.json')
        choice = read_json(src['run']/'final_model_choice.json')
        params = read_json(src['run']/'selected_parameters.json')[choice['family']]['General']['parameters']
        assert choice['family']=='Pre_direct' and set(choice['fallback_tasks'])=={'Code','Math'}
        assert params['gamma'] is None
        replay = [frozen_route(r.task_type,r.eval_name,r.q_hat_S,r.q_hat_L,r.prompt_chars,
                              registered,cost,params,choice['fallback_tasks']) for r in f.itertuples()]
        assert np.array_equal(replay,f.selected_model)
        assert frozen_route('General','__unregistered__',1,0,100,registered,cost,params,choice['fallback_tasks'])=='L'
        assert frozen_route('Unknown','hellaswag',1,0,100,registered,cost,params,choice['fallback_tasks'])=='L'
        return 'Every cached S/L prediction matched; train provenance disjoint from test; all frozen decisions replayed; unknown fallback exercised'
    check('prediction_provenance_and_route_replay',quality_and_route)

    def saved_model():
        import joblib
        from threadpoolctl import threadpool_limits
        sys.path.insert(0,str(PROJECT/'task2_complete'))
        model = joblib.load(src['run']/'pre_full_train.joblib')
        for a,mid in MIDS.items():
            rows = pd.DataFrame({'token_length':f.prompt_chars,'eval_name':f.eval_name,'model_id':mid})
            assert variant=='A_Baseline'
            with threadpool_limits(limits=2):
                values = np.clip(model.models[variant].predict(model.basic(rows)[['token_length','eval_name','model_id']]),0,1)
            np.testing.assert_allclose(values,f['q_hat_'+a],rtol=0,atol=1e-12)
        return 'Saved Pre weights reproduced all '+str(2*len(f))+' quality predictions without fitting'
    check('saved_Pre_model_full_test_replay',saved_model)

    def timing():
        arrival, deadline = utc_column(f.arrival_time,'arrival_time'),utc_column(f.deadline,'deadline')
        assert arrival.is_monotonic_increasing and (deadline>=arrival).all()
        np.testing.assert_allclose((deadline-arrival).dt.total_seconds(),f.max_delay_seconds,rtol=0,atol=2e-6)
        assert f.max_delay_seconds.ge(0).all()
        for c in cfg['service_classes']:
            mask = f.service_class.eq(c['name'])
            assert f.loc[mask,'priority'].eq(c['priority']).all()
            assert f.loc[mask,'max_delay_seconds'].eq(c['max_delay_seconds']).all()
        classes=cfg['service_classes']
        proportions=np.array([c['proportion'] for c in classes])
        counts=np.floor(proportions*len(f)).astype(int)
        order=np.argsort(-(proportions*len(f)-counts),kind='stable')
        for j in order[:len(f)-counts.sum()]:
            counts[j]+=1
        assignments=np.repeat(np.arange(len(classes)),counts)
        np.random.default_rng(cfg['service_class_seed']).shuffle(assignments)
        assert np.array_equal(f.service_class,[classes[j]['name'] for j in assignments])
        assert np.array_equal(f.origin_region,np.random.default_rng(cfg['origin_region_seed']).choice(cfg['regions'],len(f)))

        ids = splits[splits.phase.eq('test')].request_id.to_numpy()
        ids = ids[np.random.default_rng(cfg['arrival_permutation_seed']).permutation(len(ids))]
        assert np.array_equal(ids,f.request_id)
        stats = training_lengths(source)
        profile = read_json(src['resource_config'])['profile']
        estimates = service_estimates(f,stats,cfg,profile)
        span = (len(f)-1)*estimates['L']['total'].mean()/cfg['reference_utilization']/cfg['arrival_multiplier']
        azure = pd.read_csv(src['azure'],usecols=['arrival_s'],nrows=len(f)).arrival_s.to_numpy()
        seconds = (arrival-pd.Timestamp(cfg['start_datetime_utc'])).dt.total_seconds()
        np.testing.assert_allclose(seconds,(azure-azure[0])/(azure[-1]-azure[0])*span,rtol=0,atol=2e-6)
        return {'completion_deadlines':True,'UTC':True,'arrival_span_s':float(span),
                'service_class_by_task':pd.crosstab(f.task_type,f.service_class).to_dict(),
                'no_deadline_feasibility_claim':'Only validates time contract; queue/network/outage deadline feasibility not yet simulated'}
    check('arrival_deadline_classes_UTC',timing)

    def leakage_and_estimates():
        banned = {'q','score','quality_true','response','response_s','response_l','output_tokens_proxy','output_s','output_l','cost_usd'}
        assert not (banned & set(f.columns))
        assert source[source.split.eq('train')].request_id.isin(f.request_id).sum()==0
        stats = training_lengths(source)
        saved = pd.read_csv(OUT/'training_length_statistics.csv')
        pd.testing.assert_frame_equal(stats,saved,check_exact=False,rtol=1e-12)
        profile = read_json(src['resource_config'])['profile']
        estimates = service_estimates(f,stats,cfg,profile)

        for col in ['prompt_chars','input_tokens_proxy','source_row']:
            np.testing.assert_array_equal(f[col],origin[col])
        for a in ['S','L']:
            np.testing.assert_allclose(f['estimated_service_time_'+a+'_seconds'],estimates[a]['total'])
        chosen = np.where(f.selected_model.eq('S'),estimates['S']['total'],estimates['L']['total'])
        outputs = np.where(f.selected_model.eq('S'),estimates['S']['output'],estimates['L']['output'])
        np.testing.assert_allclose(f.estimated_output_tokens_proxy,outputs)
        np.testing.assert_allclose(f.estimated_service_time_seconds,chosen)
        np.testing.assert_allclose(f.estimated_energy_kwh,chosen*(profile['busy_w']-profile['idle_w'])*profile['pue']/3.6e6)
        chosen_first=np.where(f.selected_model.eq('S'),estimates['S']['first'],estimates['L']['first'])
        np.testing.assert_allclose(f.estimated_ttft_seconds,chosen_first)
        assert ((f.estimated_ttft_seconds>0)&(f.estimated_ttft_seconds<=f.estimated_service_time_seconds)).all()
        for col in ['estimated_service_time_seconds','estimated_energy_kwh','q_hat_S','q_hat_L']:
            assert np.isfinite(f[col]).all() and f[col].ge(0).all()
        assert f[['q_hat_S','q_hat_L']].le(1).all().all()
        return 'Train-only length aggregates and every service/energy estimate recomputed; current outcomes excluded; predictive accuracy on real hardware unverified'
    check('no_current_outcome_leakage_resource_replay',leakage_and_estimates)

    def capacities():
        assert set(server.region)==set(cfg['regions'])
        assert set(server.model_type)=={'S','L'} and len(server)==6
        assert not server.duplicated(['region','model_type']).any()
        for col in ['capacity','p_idle_w','p_busy_w','pue','service_time_multiplier','availability']:
            assert np.isfinite(server[col]).all()
        assert (server.capacity>0).all() and (server.capacity%1==0).all()
        assert server.pue.ge(1).all() and server.p_idle_w.ge(0).all() and (server.p_busy_w>=server.p_idle_w).all()
        assert server.availability.between(0,1).all() and server.service_time_multiplier.gt(0).all()
        for r in server.itertuples():
            assert r.capacity==cfg['replicas_per_region_per_model']
            assert r.service_time_multiplier==(1 if r.model_type=='S' else cfg['large_service_multiplier'])
        avail['parsed_time'] = utc_column(avail.datetime_utc,'availability datetime_utc')
        assert not avail.duplicated(['parsed_time','region','model_type']).any()
        lookup = server.set_index(['region','model_type'])
        for r in avail.itertuples():
            assert (r.region,r.model_type) in lookup.index
            assert np.isfinite(r.available_capacity) and r.available_capacity%1==0
            assert 0<=r.available_capacity<=lookup.loc[(r.region,r.model_type),'capacity']
        for r in server.itertuples():
            initial = avail[(avail.region==r.region)&(avail.model_type==r.model_type)]
            assert initial.parsed_time.min()<=utc_column(f.arrival_time,'arrival').min()
        return 'Six valid pools; all availability events bounded by configured capacity; events persist until next event'
    check('server_capacity_and_availability',capacities)

    def allowed_regions():
        assert len(net)==9 and not net.duplicated(['origin_region','destination_region']).any()
        assert set(net.origin_region)==set(cfg['regions'])==set(net.destination_region)
        assert net.allowed.isin([0,1]).all() and net.provenance.eq('assumed').all()
        for col in ['latency_ms','transfer_energy_kwh']:
            assert np.isfinite(net[col]).all() and net[col].ge(0).all()
        pools=set(zip(server.loc[server.capacity.gt(0),'region'],server.loc[server.capacity.gt(0),'model_type']))
        allowed={origin:set(block.destination_region) for origin,block in net[net.allowed.eq(1)].groupby('origin_region')}
        for r in f.itertuples():
            assert any((region,r.selected_model) in pools for region in allowed.get(r.origin_region,set())), 'No allowed configured pool for '+r.request_id
        return 'Every request has an allowed configured model pool; arrival-time availability is not treated as deadline feasibility'
    check('network_and_allowed_service_regions',allowed_regions)

    def energy_contract():
        contract=read_json(OUT/'resource_accounting_contract.json')
        assert contract['reference_service_already_scaled'] is True
        profile=read_json(src['resource_config'])['profile']
        multipliers={model:block.service_time_multiplier.tolist() for model,block in server.groupby('model_type')}
        for r in f.itertuples():
            ref=1 if r.selected_model=='S' else cfg['large_service_multiplier']
            for multiplier in multipliers[r.selected_model]:
                assert np.isclose(service_at_pool(r.estimated_service_time_seconds,ref,multiplier),
                                  r.estimated_service_time_seconds)
            assert np.isclose(incremental_energy_kwh(r.estimated_service_time_seconds,profile['idle_w'],
                profile['busy_w'],profile['pue']),r.estimated_energy_kwh)
        components=facility_energy_components(100,1,10,175,250,1.2)
        assert np.isclose(components['facility_kwh'],(250*10+175*90)*1.2/3.6e6)
        return 'Per-request incremental energy, two-state total-energy identity and single application of model multiplier verified'
    check('energy_and_service_multiplier_contract',energy_contract)

    scenario_summaries=[]
    def scenarios_and_prechecks():
        combined=pd.read_csv(OUT/'server_availability_scenarios.csv')
        assert set(combined.scenario)==set(cfg['availability_scenarios'])
        timing=pd.read_csv(OUT/'request_timing_constraints.csv')
        assert np.array_equal(timing.request_id,f.request_id)
        assert np.array_equal(utc_column(timing.deadline,'deadline'),utc_column(f.deadline,'deadline'))
        classes={c['name']:c for c in cfg['service_classes']}
        np.testing.assert_array_equal(timing.max_queue_wait_seconds,[classes[c]['max_queue_wait_seconds'] for c in f.service_class])
        np.testing.assert_allclose(utc_column(timing.latest_start_without_network_utc,'latest start').astype('int64')/1e9,
            utc_column(f.deadline,'deadline').astype('int64')/1e9-f.estimated_service_time_seconds,rtol=0,atol=2e-6)
        saved_i=pd.read_csv(OUT/'request_feasibility_precheck.csv')
        saved_b=pd.read_csv(OUT/'capacity_precheck.csv')
        saved_summary=read_json(OUT/'feasibility_precheck.json')['scenarios']
        for name,overrides in cfg['availability_scenarios'].items():
            table=pd.read_csv(OUT/('server_availability_'+name+'.csv'))
            expected=avail.drop(columns=['parsed_time'],errors='ignore').to_dict('records')
            for e in overrides:
                expected.append({'datetime_utc':(pd.Timestamp(cfg['start_datetime_utc'])+pd.Timedelta(seconds=e['offset_seconds'])).isoformat(),
                    'region':e['region'],'model_type':e['model_type'],'available_capacity':e['available_capacity'],
                    'provenance':'task3_assumed_'+name})
            pd.testing.assert_frame_equal(pd.DataFrame(expected),table)
            pd.testing.assert_frame_equal(combined[combined.scenario.eq(name)].drop(columns='scenario').reset_index(drop=True),table)
            i,b,s=precheck(f,server,net,table,cfg,name)
            observed=saved_i[saved_i.scenario.eq(name)].reset_index(drop=True)
            assert np.array_equal(i.request_id,observed.request_id)
            assert np.array_equal(i.isolated_possible,observed.isolated_possible)
            observed_b=saved_b[saved_b.scenario.eq(name)].reset_index(drop=True)
            assert np.array_equal(b.necessary_bound_satisfied,observed_b.necessary_bound_satisfied)
            assert s==next(x for x in saved_summary if x['scenario']==name)
            scenario_summaries.append(s)
        return {'scenarios':scenario_summaries,'meaning':'necessary conditions only; failures are scenario diagnostics, no feasible schedule claimed'}
    check('availability_scenarios_wait_deadline_capacity_prechecks',scenarios_and_prechecks)

    def alignment():
        baseline=pd.read_csv(HERE/'baselines/2024_original/requests.csv')
        if (OUT/'time_alignment.json').exists():
            audit=read_json(OUT/'time_alignment.json')
            current_baseline=pd.read_csv(DEFAULT_OUT/'requests.csv')
            assert audit['baseline_requests_sha256']==digest(DEFAULT_OUT/'requests.csv')
            pd.testing.assert_frame_equal(current_baseline.drop(columns=['arrival_time','deadline']),f.drop(columns=['arrival_time','deadline']))
            shift=pd.Timedelta(seconds=audit['offset_seconds'])
            for col in ['arrival_time','deadline']:
                assert (utc_column(f[col],col)-utc_column(current_baseline[col],col)==shift).all()
            return {'translated':True,'offset_seconds':shift.total_seconds(),'non_time_columns_unchanged':True}
        pd.testing.assert_frame_equal(baseline,f)
        assert digest(OUT/'server_availability.csv')==digest(HERE/'baselines/2024_original/server_availability.csv')
        return 'Original 2024 request table and base availability preserved byte-for-byte; alignment is separate'
    check('baseline_preservation_or_time_alignment',alignment)

    def reproducible():
        shock = pd.read_csv(OUT/'service_random_conditions.csv')
        assert not shock.duplicated(['request_id','rep']).any()
        sigma = read_json(src['resource_config'])['profile']['residual_log_sigma']
        for rep,seed in enumerate(cfg['service_shock_seeds']):
            block = shock[shock.rep.eq(rep)].set_index('request_id').loc[f.request_id]
            expected = np.exp(np.random.default_rng(seed).normal(-0.5*sigma*sigma,sigma,len(f)))
            np.testing.assert_allclose(block.service_multiplier_shock,expected,rtol=1e-14)
            assert block.seed.eq(seed).all()
        assert len(shock)==len(f)*len(cfg['service_shock_seeds'])
        manifest = read_json(OUT/'data_manifest.json')
        for name, record in manifest['files'].items():
            assert digest(OUT/name)==record['sha256'], 'Artifact hash mismatch '+name
        assert digest(HERE/'config.json')==manifest['config_sha256']
        for name,sha in manifest['script_hashes'].items():
            assert digest(HERE/name)==sha, 'Script hash mismatch '+name
        return 'Immutable common request/resource hashes and request-keyed mean-one shocks verified; no policy simulation has been executed'
    check('common_resources_randomness_and_artifact_hashes',reproducible)

    def unchanged():
        integrity = read_json(OUT/'source_integrity.json')
        assert integrity['before']==integrity['after']
        for name,sha in integrity['before'].items():
            assert digest(PROJECT/name)==sha, 'Source changed '+name
        return str(len(integrity['before']))+' protected Task1/Task2/raw source files unchanged since generation'
    check('original_files_unchanged',unchanged)
    check('interface_and_information_boundary_tests', lambda:ensure(
        read_json(OUT/'interface_test_report.json')['passed'],
        str(read_json(OUT/'interface_test_report.json')['tests_run'])+' interface tests passed, including alignment, capacity, units, accounting and information boundaries'))
    carbon_data=None
    carbon_path = OUT/cfg['carbon_filename']
    if not carbon_path.exists():
        checks['real_hourly_carbon']={'status':'pending','details':'Missing '+str(carbon_path)+'. No fabricated/annual/price proxy used; final carbon experiments blocked.'}
    else:
        def carbon():
            nonlocal carbon_data
            data = load_carbon(carbon_path,OUT/cfg['carbon_metadata_filename'],
                required_start=pd.Timestamp(f.arrival_time.min()),required_end=pd.Timestamp(f.deadline.max()),
                expected_regions=set(server.region))
            assert data.attrs['metadata']['data_kind']=='actual', 'Actual accounting file must declare data_kind actual; forecasts use a separate file'
            carbon_data=data
            return {'rows':len(data),'coverage':'arrival through max completion deadline; overruns require extension',
                    'sha256':digest(carbon_path),'metadata_sha256':digest(OUT/cfg['carbon_metadata_filename']),
                    'physical_truth_not_verified':'Schema/source declaration is checked; provider methodology requires review'}
        check('real_hourly_carbon',carbon)
    forecast_path=OUT/cfg['forecast_filename']
    if not forecast_path.is_file():
        checks['decision_time_carbon_forecasts']={'status':'pending','details':'No forecast file provided; no future-actual substitution'}
    else:
        def forecasts():
            forecast=load_carbon(forecast_path,OUT/cfg['forecast_metadata_filename'])
            assert forecast.attrs['metadata']['data_kind']=='forecast'
            if carbon_data is None:
                raise ValueError('Validate actual carbon boundary before comparing forecasts')
            assert_compatible_carbon(carbon_data,forecast)


            for value in f.arrival_time:
                carbon_for_decision(forecast,value,[value])
            return {'arrival_hour_issuance_coverage':'passed','full_MPC_horizon':'not_configured_not_verified',
                'sha256':digest(forecast_path),'metadata_sha256':digest(OUT/cfg['forecast_metadata_filename']),
                'accuracy':'not_verified'}
        check('decision_time_carbon_forecasts',forecasts)
    checks['physical_calibration_and_new_holdout']={'status':'not_verified','details':'No matched hardware measurements/new independent holdout. Inherited test is previously inspected. No routing quality guarantee asserted.'}
    failed = [k for k,v in checks.items() if v['status']=='failed']
    actual_ready=checks['real_hourly_carbon']['status']=='passed'
    forecast_ready=checks['decision_time_carbon_forecasts']['status']=='passed'
    report = {'data_contract_checks_passed':not failed,
              'ready_for_real_carbon_accounting_inputs':actual_ready and not failed,
              'decision_time_arrival_hour_forecasts_ready':forecast_ready and not failed,
              'ready_for_real_carbon_experiments':actual_ready and forecast_ready and not failed,
              'MPC_horizon_and_joint_scheduling_feasibility':'not_verified_not_implemented',
              'scenario_precheck_summaries':scenario_summaries,
              'failed_checks':failed,'checks':checks,'rows':{'requests':len(f),'server_config':len(server),
                'server_availability':len(avail),'network_config':len(net)},
              'no_MILP_MPC_DES_models_built':True,
              'report_source':'validate_task3_data.py; current saved files, not copied Task2 verification claims'}
    save_json(OUT/'validation_report.json',report)
    print('Validation:', 'PASSED with pending external data' if not failed else 'FAILED '+str(failed))
    if require_carbon and checks['real_hourly_carbon']['status']!='passed':
        return 2
    return 1 if failed else 0


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-carbon',action='store_true',help='Exit 2 if real hourly carbon input is missing')
    parser.add_argument('--data-dir',type=Path,help='Validate a separately aligned dataset')
    args = parser.parse_args()
    raise SystemExit(validate(args.require_carbon,args.data_dir))
