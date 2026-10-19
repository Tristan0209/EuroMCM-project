import math
import pandas as pd
from .util import ROOT, PROJECT, PREP, DATA, read, save, sha


def upstream_paths():
    config=read(ROOT/'configs/upstream.json')
    task2=PROJECT/config['task2_directory']
    paths={'task2':task2,'run':task2/config['task2_run'],
        'paired_requests':task2/'inputs/paired/requests.jsonl',
        'paired_outcomes':task2/'inputs/paired/outcomes.jsonl',
        'resource_config':PROJECT/'Work 1/task1_response time, and environ mental impact/config.json',
        'resource_core':PROJECT/'Work 1/task1_response time, and environ mental impact/core.py'}
    for name,path in paths.items():
        if not path.exists():raise FileNotFoundError('Current upstream '+name+': '+str(path))
    return paths


def profile():
    paths=upstream_paths()
    current=read(paths['resource_config'])['profile']
    recorded=read(paths['run']/'resource_assumptions.json')
    for field in ['shared_w','gpu_idle_w','gpu_busy_w','pue','prefill_coefficients','decode_coefficients','residual_log_sigma']:
        if current[field]!=recorded['profile'][field]:raise ValueError('Task1/Task2 profile mismatch: '+field)
    if recorded.get('submission_source_sha256',recorded['source_sha256'])!=sha(paths['resource_core']):raise ValueError('Task2 resource output uses a different Task1 core')
    p={**current}
    p['gpu_count_per_server']=int(p['gpu_count_per_server'])
    values=[p[k] for k in ['shared_w','gpu_idle_w','gpu_busy_w','pue','server_max_w']]
    if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite DGX profile')
    if not (p['shared_w']>=0 and 0<=p['gpu_idle_w']<=p['gpu_busy_w'] and p['pue']>=1):raise ValueError('Invalid DGX profile')
    peak=p['shared_w']+p['gpu_count_per_server']*p['gpu_busy_w']
    if peak>p['server_max_w']:raise ValueError('GPU profile exceeds configured IT ceiling')
    return p


def idle_base_w(pool):


    return pool['idle_base_w'] if 'idle_base_w' in pool else pool['capacity']*pool['idle']


def incremental_w(pool):
    value=pool['busy']-pool['idle']
    if value<0:raise ValueError('Negative busy-over-idle power')
    return value


def migrate_power(requests):
    p=profile();config=read(ROOT/'configs/upstream.json')
    if config['physical_hosts_per_pool']!=1:raise ValueError('This topology requires exactly one physical host per S/L pool')
    slots=config['serving_slots_per_pool'];gpus=p['gpu_count_per_server']
    if not isinstance(slots,int) or not 1<=slots<=gpus:raise ValueError('Serving slots must fit installed GPUs')
    base=p['shared_w']+gpus*p['gpu_idle_w'];delta=p['gpu_busy_w']-p['gpu_idle_w']
    servers=pd.read_csv(DATA/'server_config.csv')
    if not servers.capacity.eq(slots).all():raise ValueError('Migration cannot silently change frozen serving capacity')
    servers['host_id']=[r+'_'+m+'_DGX01' for r,m in zip(servers.region,servers.model_type)]
    servers['physical_hosts']=1;servers['gpu_count_per_host']=gpus
    servers['shared_w']=p['shared_w'];servers['gpu_idle_w']=p['gpu_idle_w'];servers['gpu_busy_w']=p['gpu_busy_w']
    servers['server_max_w']=p['server_max_w'];servers['pue']=p['pue']
    servers['p_idle_w']=base;servers['p_busy_w']=base+delta
    servers['capacity_unit']='admitted_gpu_service_slots_not_installed_gpu_count'
    servers['parameter_source']='current Task1 config + current Task2 resource_assumptions; hypothetical DGX proxy'
    servers['power_representation']='fixed_host_idle_plus_busy_gpu_increment'
    servers.to_csv(DATA/'server_config.csv',index=False)
    hosts=servers[['host_id','region','model_type','physical_hosts','gpu_count_per_host','shared_w',
        'gpu_idle_w','gpu_busy_w','pue','server_max_w']].copy()
    hosts['idle_it_power_w']=base;hosts['one_busy_slot_it_power_w']=base+delta
    hosts['admitted_serving_slots']=slots
    hosts.to_csv(DATA/'physical_host_config.csv',index=False)
    requests=requests.copy()
    requests['estimated_energy_kwh']=requests.estimated_service_time_seconds*delta*p['pue']/3.6e6
    requests.to_csv(DATA/'requests.csv',index=False)
    contract={'schema_version':3,'power_scenario':config['power_scenario'],
        'topology':'DE/FR/ES each have separate S and L hosts; six fixed powered 8-GPU hosts',
        'estimated_energy_kwh':'incremental facility busy-GPU-over-idle energy only',
        'increment_formula':'service_seconds*(gpu_busy_w-gpu_idle_w)*pue/3.6e6',
        'idle_base_formula':'sum_physical_hosts((shared_w+installed_gpus*gpu_idle_w)*pue*common_horizon_seconds)/3.6e6',
        'facility_total':'fixed_host_idle_base + busy_gpu_increment + network + controller; each exactly once',
        'reference_service_already_scaled':True,'installed_gpus_create_no_additional_admitted_capacity':True,
        'capacity_change_power':'capacity changes block admission only; hosts remain powered',
        'p_idle_w_p_busy_w_aliases':'whole-host idle and whole-host with ONE busy GPU; difference is the single-slot increment',
        'multiple_slots':'host baseline charged once, irrespective of admitted slot count',
        'controller':'separate inherited incremental computing-work proxy, not another host idle base',
        'profile':p,'source_config_sha256':sha(upstream_paths()['resource_config'])}
    save(DATA/'resource_accounting_contract.json',contract)
    parameters=read(DATA/'resource_parameters.json')
    parameters['legacy_power_profile']=parameters['inherited_task2_assumptions']['profile']
    inherited=parameters['inherited_task2_assumptions'];inherited['profile']=p
    inherited['facility_energy_formula']=contract['idle_base_formula']+' + incremental_busy + controller + network'
    inherited['source_resource_scenarios']=str((upstream_paths()['run']/'resource_scenarios.csv').relative_to(PROJECT))
    parameters['power_migration']={'source':'configs/upstream.json','six_hosts':True,'slots_per_pool':slots,
        'idle_it_w_per_host':base,'incremental_it_w_per_busy_slot':delta,
        'legacy_results':'archives/legacy_power_175_250_20261017',
        'power_values_are_proxy_scenarios_not_GPT4_or_Mixtral_measurements':True}
    parameters['scope']='task3_fixed_host_DGX_resource_scenario'
    save(DATA/'resource_parameters.json',parameters)
    return requests
