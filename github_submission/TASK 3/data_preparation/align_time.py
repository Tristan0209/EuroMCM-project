import argparse
import copy
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from common import HERE, OUT, read_json, save_json, digest, utc_strings
from carbon_loader import load_carbon, utc_column
from data_extensions import build_extensions


def alignment_delta(requests, carbon, target_start_utc=None):
    if carbon.attrs['metadata']['data_kind'] != 'actual':
        raise ValueError('Alignment requires real actual carbon input, not synthetic or forecast data')
    old_start=utc_column(requests.arrival_time,'arrival_time').min()
    target=(carbon.datetime_utc.min() if target_start_utc is None else
            utc_column(pd.Series([target_start_utc]),'target_start_utc').iloc[0])
    if target.nanosecond:
        raise ValueError('Use a target start with at most microsecond precision')
    delta=target-old_start
    last=utc_column(requests.deadline,'deadline').max()+delta
    expected=pd.MultiIndex.from_product([pd.date_range(target.floor('h'),last.floor('h'),freq='h'),sorted(carbon.region.unique())])
    actual=pd.MultiIndex.from_frame(carbon[['datetime_utc','region']])
    if not expected.difference(actual).empty:
        raise ValueError('Requested alignment does not fit complete carbon coverage through completion deadlines')
    return delta


def shift_times(frame, delta, columns):
    shifted=frame.copy()
    for col in columns:
        shifted[col]=utc_strings(utc_column(frame[col],col)+delta)
    return shifted


def align_dataset(carbon_path, metadata_path=None, target_start=None, destination=None):
    cfg=read_json(OUT/'effective_config.json')
    live_config=read_json(HERE/'config.json')
    frozen_settings={k:v for k,v in cfg.items() if k!='time_alignment'}
    live_settings={k:v for k,v in live_config.items() if k!='time_alignment'}
    if frozen_settings != live_settings:
        raise ValueError('Non-alignment configuration changed; regenerate baseline before translating it')
    cfg['time_alignment']=live_config['time_alignment']
    requested=destination or HERE/cfg['time_alignment']['aligned_directory']
    target=Path(requested).resolve()
    if HERE not in target.parents or target==OUT or target in OUT.parents:
        raise ValueError('Aligned destination must be a separate subdirectory within data_preparation')
    if target.exists():
        raise FileExistsError('Choose a new destination to preserve existing aligned datasets: '+str(target))
    carbon=load_carbon(carbon_path,metadata_path)
    baseline=pd.read_csv(OUT/'requests.csv')
    configured=cfg['time_alignment'].get('target_start_utc')
    delta=alignment_delta(baseline,carbon,target_start or configured)
    f=shift_times(baseline,delta,['arrival_time','deadline'])
    cfg=copy.deepcopy(cfg)
    cfg['start_datetime_utc']=f.arrival_time.iloc[0]
    cfg['time_alignment'].update({'mode':'carbon_coverage','offset_seconds':delta.total_seconds(),
        'baseline_anchor_utc':'2024-05-10T00:00:00Z', 'target_start_utc':f.arrival_time.iloc[0],
        'selection_rule':'earliest covered hour unless explicit start specified; intensity values never select the date'})
    for row in cfg['availability_overrides']:
        row['datetime_utc']=(pd.Timestamp(row['datetime_utc'])+delta).isoformat()

    target.mkdir(parents=True)
    for name in ['server_config.csv','network_config.csv','training_length_statistics.csv',
                 'service_random_conditions.csv','source_integrity.json','project_inventory.json','interface_test_report.json']:
        shutil.copyfile(OUT/name,target/name)
    f.to_csv(target/'requests.csv',index=False)
    avail=shift_times(pd.read_csv(OUT/'server_availability.csv'),delta,['datetime_utc'])
    avail.to_csv(target/'server_availability.csv',index=False)
    save_json(target/'effective_config.json',cfg)
    build_extensions(target,f,pd.read_csv(target/'server_config.csv'),pd.read_csv(target/'network_config.csv'),cfg)
    params=read_json(OUT/'resource_parameters.json')
    params['task3_new_assumptions']['config']=cfg
    params['task3_new_assumptions']['absolute_time_anchor']='translated synthetic arrivals; not actual RouterBench arrivals in the carbon-data year'
    params['task3_new_assumptions']['precheck']=read_json(target/'feasibility_precheck.json')['scenarios']
    params['carbon']['real_hourly_data_available']=True
    save_json(target/'resource_parameters.json',params)
    carbon_path=Path(carbon_path)
    meta_path=Path(metadata_path) if metadata_path else carbon_path.with_suffix('.metadata.json')
    shutil.copyfile(carbon_path,target/cfg['carbon_filename'])
    shutil.copyfile(meta_path,target/cfg['carbon_metadata_filename'])
    for name in [cfg['forecast_filename'],cfg['forecast_metadata_filename']]:
        if (OUT/name).is_file():
            shutil.copyfile(OUT/name,target/name)
    audit={'baseline_requests_sha256':digest(OUT/'requests.csv'),'offset_seconds':delta.total_seconds(),
        'baseline_anchor_utc':baseline.arrival_time.iloc[0],'aligned_anchor_utc':f.arrival_time.iloc[0],
        'carbon_source_sha256':digest(carbon_path),'carbon_metadata_sha256':digest(meta_path),
        'non_time_columns_unchanged':True,'all_scenarios_use_same_translation':True,
        'real_RouterBench_arrivals':False,'no_carbon_values_used_for_date_selection':True}
    pd.testing.assert_frame_equal(baseline.drop(columns=['arrival_time','deadline']),f.drop(columns=['arrival_time','deadline']))
    np.testing.assert_allclose(np.diff(utc_column(f.arrival_time,'arrival').astype('int64')),
        np.diff(utc_column(baseline.arrival_time,'arrival').astype('int64')),rtol=0,atol=0)
    save_json(target/'time_alignment.json',audit)
    manifest=read_json(OUT/'data_manifest.json')
    manifest['configuration']=cfg
    manifest['config_sha256']=digest(HERE/'config.json')
    manifest['time_alignment']=audit
    manifest['carbon']={'status':'actual_schema_and_alignment_coverage_validated','unit':carbon.attrs['metadata']['unit'],
        'emissions_species':carbon.attrs['metadata']['emissions_species'],'source_path':str(carbon_path.resolve()),
        'no_real_or_proxy_carbon_generated':True,'provider_methodology_independently_verified':False}
    manifest['files']={p.name:{'sha256':digest(p),'rows':len(pd.read_csv(p)) if p.suffix=='.csv' else None,
        'sources':['2024 baseline manifest','real user-provided carbon file','effective_config.json'],
        'generation':'translate all synthetic time fields; recompute scenarios/prechecks; copy carbon unchanged'}
        for p in target.iterdir() if p.suffix in {'.csv','.json'}}
    manifest['script_hashes']={p.name:digest(p) for p in HERE.glob('*.py')}
    save_json(target/'data_manifest.json',manifest)
    return target


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--carbon',type=Path,required=True)
    parser.add_argument('--metadata',type=Path)
    parser.add_argument('--target-start',help='Explicit UTC start, default config value or earliest covered hour')
    parser.add_argument('--destination',type=Path)
    args=parser.parse_args()
    destination=align_dataset(args.carbon,args.metadata,args.target_start,args.destination)
    print('Aligned dataset:',destination)
    raise SystemExit(subprocess.call([sys.executable,'-B',str(HERE/'validate_task3_data.py'),'--data-dir',str(destination),'--require-carbon']))
