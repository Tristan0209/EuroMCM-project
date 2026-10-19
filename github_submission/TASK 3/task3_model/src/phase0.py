import importlib
import io
import platform
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import scipy

from .util import ROOT,PROJECT,PREP,DATA,read,save,sha,snapshot,input_signature
from .power import upstream_paths,migrate_power,profile


def upstream_tests():
    sys.path.insert(0,str(PREP))
    suite_module=importlib.import_module('test_data_contracts')
    alignment=importlib.import_module('align_time')

    shutil.copyfile(PREP/'config.json',ROOT/'config.json')
    original_temp=tempfile.TemporaryDirectory
    def redirected(*args,**kwargs):
        kwargs['dir']=ROOT/'results/tmp'
        return original_temp(*args,**kwargs)
    stream=io.StringIO()
    with patch.object(tempfile,'TemporaryDirectory',redirected),patch.object(alignment,'HERE',ROOT):
        outcome=unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(suite_module))
    (ROOT/'results/audit/upstream_tests.txt').write_text(stream.getvalue(),encoding='utf-8')
    captured={'legacy_interface_tests':{'passed':outcome.wasSuccessful(),'tests':outcome.testsRun},
        'legacy_full_validator':'superseded: old paths and independent-replica power keys do not describe current upstream',
        'current_data_validation':'explicit current-source checks in verify_current_sources; no legacy power pass claimed'}
    save(ROOT/'results/audit/upstream_validation.json',captured)
    if not outcome.wasSuccessful():
        raise RuntimeError('Upstream interface tests or current-data contracts failed; see audit logs')
    return outcome.testsRun,captured


def normalize_carbon():
    tables={name:[] for name in ['lifecycle','direct']};audit=[];sources={}
    for region in ['DE','FR','ES']:
        files=list((PREP/'Carbon Intensity').glob('*'+region+'-2025-hourly.csv'))
        if len(files)!=1:raise ValueError('Ambiguous/missing real carbon CSV '+region)
        path=files[0];f=pd.read_csv(path)
        if not f['Zone id'].eq(region).all():raise ValueError('Wrong region in '+str(path))

        times=pd.to_datetime(f['Datetime (UTC)'],utc=True,errors='raise')
        expected=pd.date_range('2025-01-01T00:00:00Z','2025-12-31T23:00:00Z',freq='h')
        if times.duplicated().any() or set(times)!=set(expected):raise ValueError('Carbon hour continuity failure '+region)
        for boundary,label in [('lifecycle','Life cycle'),('direct','direct')]:
            col=next(c for c in f if c.startswith('Carbon intensity ') and c.endswith('('+label+')'))
            values=pd.to_numeric(f[col],errors='raise')
            if not np.isfinite(values).all() or not values.ge(0).all():raise ValueError('Invalid real carbon values')
            tables[boundary].append(pd.DataFrame({'datetime_utc':times,'region':region,
                'carbon_intensity_gco2e_kwh':values,'provider_estimated':f['Data estimated'],
                'provider_data_source':f['Data source'],'provider_estimation_method':f['Data estimation method']}))
        sources[region]={'path':str(path.relative_to(PROJECT)),'sha256':sha(path)}
        audit.append({'region':region,'rows':len(f),'first':str(times.min()),'last':str(times.max()),
            'duplicates':int(times.duplicated().sum()),'provider_estimated_hours':int(f['Data estimated'].sum()),
            'critical_nulls':0,'timezone_rule':'UTC declared by original Datetime (UTC) column'})
    for boundary,parts in tables.items():
        f=pd.concat(parts,ignore_index=True).sort_values(['datetime_utc','region'])
        f.to_csv(DATA/('carbon_'+boundary+'.csv'),index=False)
        save(DATA/('carbon_'+boundary+'.metadata.json'),{'data_kind':'actual','emissions_species':'CO2e','unit':'gCO2e/kWh',
            'value_column':'carbon_intensity_gco2e_kwh','resolution':'hourly','timezone':'UTC',
            'source':'User-provided snapshots dated 2026-02-10; Electricity Maps-style export; provenance inferred from format and provider fields',
            'raw_files':sources,'boundary':boundary,'accounting_basis':'consumption_based_provider_methodology',
            'methodology_url':'https://www.electricitymaps.com/data/methodology',
            'GWP_reference':'Provider-reported CO2eq preserved, no gas conversion; precise snapshot-specific GWP factors not embedded in CSV and not independently reconstructed',
            'methodology_verification':'column labels verified; source attribution inferred, historical export factor versions unverified',
            'revision_warning':'retrospective 2026 snapshot; exact 2025 as-of publication values unavailable'})

    f=pd.read_csv(DATA/'carbon_lifecycle.csv')[['datetime_utc','region','carbon_intensity_gco2e_kwh']]
    f.rename(columns={'carbon_intensity_gco2e_kwh':'carbon_intensity_gco2_kwh'}).to_csv(DATA/'carbon_intensity.csv',index=False)
    meta=read(DATA/'carbon_lifecycle.metadata.json')
    meta.update({'value_column':'carbon_intensity_gco2_kwh','legacy_column_alias':True,
        'alias_warning':'legacy gco2 name contains provider CO2e, not pure CO2; canonical lifecycle file uses gco2e'})
    save(DATA/'carbon_intensity.metadata.json',meta)
    return audit


def align_inputs():
    sys.path.insert(0,str(PREP))
    from align_time import alignment_delta,shift_times
    source=pd.read_csv(PREP/'output/requests.csv')
    carbon=pd.read_csv(DATA/'carbon_lifecycle.csv')
    carbon['datetime_utc']=pd.to_datetime(carbon.datetime_utc,utc=True)
    carbon.attrs['metadata']={'data_kind':'actual'}

    anchor=carbon.datetime_utc.min()+pd.Timedelta(hours=24)
    delta=alignment_delta(source,carbon,anchor.isoformat())
    aligned=shift_times(source,delta,['arrival_time','deadline'])
    aligned.to_csv(DATA/'requests.csv',index=False)
    for name in ['server_config.csv','network_config.csv','service_random_conditions.csv','request_timing_constraints.csv',
                 'resource_accounting_contract.json','resource_parameters.json']:
        shutil.copyfile(PREP/'output'/name,DATA/name)
    constraints=pd.read_csv(DATA/'request_timing_constraints.csv')
    constraints=shift_times(constraints,delta,['arrival_time','deadline','latest_start_without_network_utc'])
    constraints.to_csv(DATA/'request_timing_constraints.csv',index=False)
    for name in ['server_availability.csv']+['server_availability_'+s+'.csv' for s in read(PREP/'config.json')['availability_scenarios']]:
        table=shift_times(pd.read_csv(PREP/'output'/name),delta,['datetime_utc'])
        table.to_csv(DATA/name,index=False)
    pd.testing.assert_frame_equal(source.drop(columns=['arrival_time','deadline']),aligned.drop(columns=['arrival_time','deadline']))
    save(DATA/'alignment.json',{'anchor_utc':anchor.isoformat(),'offset_seconds':delta.total_seconds(),
        'rule':'earliest complete common coverage after exactly 24h warmup; no intensity-based date selection',
        'baseline_sha256':sha(PREP/'output/requests.csv'),'non_time_columns_unchanged_before_power_migration':True,
        'power_migration_changed_columns':['estimated_energy_kwh'],
        'real_business_arrivals':False,'upstream_helper_reused':'align_time.alignment_delta + shift_times',
        'destination_adapter':'legacy align_dataset destination guard bypassed by calling pure helpers, not changing upstream code'})
    aligned=migrate_power(aligned)
    run=upstream_paths()['run']

    d=pd.read_csv(run/'decisions.csv.gz',usecols=['strategy','phase','rep','request_id','final_l','q','task_type'])
    d=d[d.strategy.eq('Selected_finite_sample_guard')&d.phase.eq('test')&d.rep.eq(0)].set_index('request_id').loc[aligned.request_id]
    if not np.array_equal(np.where(d.final_l,'L','S'),aligned.selected_model):raise ValueError('Frozen routing mismatch')
    if not aligned.loc[aligned.task_type.isin(['Code','Math']),'selected_model'].eq('L').all():raise ValueError('Risk fallback changed')
    d[['q']].reset_index().to_csv(DATA/'evaluation_only_historical_quality.csv',index=False)
    return aligned


def verify_current_sources(f):

    import common
    import joblib
    from threadpoolctl import threadpool_limits
    paths=upstream_paths();run=paths['run'];p=profile()
    source_map={**paths,'routerbench':PROJECT/'AI_datasets/RouterBench/routerbench_0shot.csv',
        'azure':PROJECT/'Work 1/task1_response time, and environ mental impact/data/processed/azure_code.csv'}
    with patch.object(common,'paths',return_value=source_map):
        source=common.requests_source();stats=common.training_lengths(source)
    saved=pd.read_csv(PREP/'output/training_length_statistics.csv')
    pd.testing.assert_frame_equal(stats,saved,check_exact=False,rtol=1e-12)
    estimates=common.service_estimates(f,stats,read(PREP/'config.json'),p)
    for m in ['S','L']:
        np.testing.assert_allclose(f['estimated_service_time_'+m+'_seconds'],estimates[m]['total'],rtol=1e-12)
    chosen=np.where(f.selected_model.eq('S'),estimates['S']['total'],estimates['L']['total'])
    np.testing.assert_allclose(f.estimated_service_time_seconds,chosen,rtol=1e-12)
    np.testing.assert_allclose(f.estimated_energy_kwh,chosen*(p['gpu_busy_w']-p['gpu_idle_w'])*p['pue']/3.6e6,rtol=1e-12)
    if f.request_id.duplicated().any() or len(f)!=7335:raise ValueError('Request ID/count changed')
    splits=pd.read_csv(run/'splits.csv')
    if set(f.request_id)!=set(splits[splits.phase.eq('test')].request_id):raise ValueError('Frozen test partition changed')
    if splits.groupby('group').phase.nunique().max()!=1:raise ValueError('Cross-partition prompt leakage')
    source=source.set_index('request_id').loc[f.request_id]
    for column in ['task_type','eval_name','source_row','prompt_chars','input_tokens_proxy']:
        np.testing.assert_array_equal(source[column],f[column])
    pre=pd.read_csv(run/'pre_predictions.csv.gz',usecols=['request_id','model_id','A_Baseline','fit_id'])
    sys.path.insert(0,str(paths['task2']))
    model=joblib.load(run/'pre_full_train.joblib')
    for m,mid in common.MIDS.items():
        rows=pre[pre.model_id.eq(mid)].set_index('request_id').loc[f.request_id]
        np.testing.assert_allclose(f['q_hat_'+m],rows.A_Baseline,rtol=0,atol=1e-14)
        if not rows.fit_id.eq('Pre_full_train').all():raise ValueError('Wrong predictive provenance')
        features=pd.DataFrame({'token_length':f.prompt_chars,'eval_name':f.eval_name,'model_id':mid})
        with threadpool_limits(limits=2):
            values=np.clip(model.models['A_Baseline'].predict(model.basic(features)[['token_length','eval_name','model_id']]),0,1)
        np.testing.assert_allclose(values,f['q_hat_'+m],rtol=0,atol=1e-12)
    choice=read(run/'final_model_choice.json');parameters=read(run/'selected_parameters.json')['Pre_direct']['General']['parameters']
    if choice['family']!='Pre_direct' or set(choice['fallback_tasks'])!={'Code','Math'} or parameters['gamma'] is not None:raise ValueError('Frozen router policy changed')
    raw_source=pd.read_json(paths['paired_requests'],lines=True)
    registry=set(raw_source[raw_source.split.eq('train')].eval_name)
    cost=read(run/'cost_predictor.json')
    replay=[common.frozen_route(r.task_type,r.eval_name,r.q_hat_S,r.q_hat_L,r.prompt_chars,registry,cost,parameters,choice['fallback_tasks']) for r in f.itertuples()]
    np.testing.assert_array_equal(replay,f.selected_model)
    legacy=read(ROOT/'configs/upstream.json')['legacy_archive']
    old=pd.read_csv(ROOT/legacy/'results/data/requests.csv')
    pd.testing.assert_frame_equal(old.drop(columns='estimated_energy_kwh'),f.drop(columns='estimated_energy_kwh'),check_exact=False,rtol=1e-12)
    frozen_quality=pd.read_csv(ROOT/legacy/'results/data/evaluation_only_historical_quality.csv')
    pd.testing.assert_frame_equal(frozen_quality,pd.read_csv(DATA/'evaluation_only_historical_quality.csv'),check_exact=False,rtol=1e-12)
    checks={'frozen_ID_model_quality_and_time_fields':True,'train_only_service_proxy':True,
        'saved_Pre_full_test_replay':2*len(f),'no_route_retraining_or_threshold_change':True,
        'new_energy_estimate_replayed':True,'six_fixed_hosts_six_serving_slots':True,
        'all_non_energy_request_fields_equal_legacy':True}
    from feasibility_precheck import precheck
    necessary={}
    for scenario in read(PREP/'config.json')['availability_scenarios']:
        individual,bounds,summary=precheck(f,pd.read_csv(DATA/'server_config.csv'),pd.read_csv(DATA/'network_config.csv'),
            pd.read_csv(DATA/('server_availability_'+scenario+'.csv')),read(PREP/'config.json'),scenario)
        individual.to_csv(DATA/('feasibility_individual_'+scenario+'.csv'),index=False)
        bounds.to_csv(DATA/('feasibility_capacity_'+scenario+'.csv'),index=False)
        necessary[scenario]=summary
    save(DATA/'feasibility_prechecks.json',necessary)
    files=[paths['resource_config'],paths['resource_core'],paths['paired_requests'],paths['paired_outcomes'],
        run/'resource_assumptions.json',run/'decisions.csv.gz',run/'pre_predictions.csv.gz',run/'pre_full_train.joblib',
        run/'selected_parameters.json',run/'final_model_choice.json',run/'cost_predictor.json',run/'splits.csv',
        source_map['routerbench'],source_map['azure']]
    save(DATA/'upstream_source_manifest.json',{'files':{str(x.relative_to(PROJECT)):sha(x) for x in files},
        'run':str(run.relative_to(PROJECT)),'checks':checks,'legacy_validation_superseded_explicitly':True})
    save(ROOT/'results/audit/current_source_validation.json',checks)
    return checks


def run():
    DATA.mkdir(parents=True,exist_ok=True)
    for name in ['audit','tmp']:(ROOT/'results'/name).mkdir(parents=True,exist_ok=True)
    before=snapshot();save(ROOT/'results/audit/protected_before.json',before)
    tests,validation=upstream_tests()
    carbon=normalize_carbon();f=align_inputs();current=verify_current_sources(f)
    after=snapshot()
    if before!=after:raise RuntimeError('An upstream file changed during Phase 0')
    checks={'upstream_tests':{'status':'passed','tests':tests},'upstream_contracts':{'status':'passed','scope':'current DGX source replay; legacy full validator superseded','details':current},
        'real_carbon_numeric_timezone_coverage':{'status':'passed','regions':carbon},
        'frozen_routing_and_alignment':{'status':'passed','requests':len(f),'models':f.selected_model.value_counts().to_dict()},
        'provider_as_of_and_exact_GWP_version':{'status':'not_verified','reason':'not encoded in historical CSV; preserve reported CO2e and document reforecast assumptions'},
        'prediction_pipeline':{'status':'pending','reason':'implemented and tested in subsequent model stage'},
        'problem_pdf':{'status':'missing','reason':'2026_Problem_Lambda.pdf absent; task statement supplied by user used'},
        'protected_files':{'status':'passed','count':len(before),'unchanged':True}}
    save(ROOT/'results/audit/phase0.json',{'checks':checks,'runtime':{'python':platform.python_version(),'scipy':scipy.__version__},
        'compatibility_issues':['legacy destination guard','naive strings explicitly labelled UTC','CO2e vs legacy gco2 alias',
            'legacy loader requires nonembedded numerical GWP horizon; new adapter preserves provider reported CO2e without inventing horizon']})
    manifest={'files':{p.name:sha(p) for p in DATA.iterdir() if p.is_file() and p.name!='manifest.json'},'source_protection':str(ROOT/'results/audit/protected_before.json')}
    save(DATA/'manifest.json',manifest)
    save(ROOT/'results/audit/phase0_input_manifest.json',manifest)
    input_signature.cache_clear()
    save(ROOT/'results/audit/runtime_input_signature.json',input_signature())
    print('PHASE0 PASSED',len(f),'requests; 3 x 8760 real carbon hours',flush=True)


if __name__=='__main__':run()
