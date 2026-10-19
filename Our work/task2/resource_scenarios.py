import argparse
import heapq
import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from prepare import save
from q1_adapter import digest
from routing import SEED

HERE=Path(__file__).resolve().parent


def load_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def simulate(frame,path,arrivals,first,service,replicas,router_delay,check_delay,post_tasks,vc2_tasks,network):
    n=len(frame); events=[]; pools={a:[(0.,r) for r in range(replicas)] for a in ['S','L']}
    for pool in pools.values(): heapq.heapify(pool)
    for i in range(n):
        action='L' if bool(path.initial_l.iloc[i]) else 'S'
        heapq.heappush(events,(float(arrivals[i]+network+router_delay),i,0,action))
    finished=np.zeros(n);records=[];checks=np.zeros(n);vc_checks=np.zeros(n)
    while events:
        release,i,stage,action=heapq.heappop(events)
        available,replica=heapq.heappop(pools[action]);start=max(release,available)
        finish=start+service[action][i]
        heapq.heappush(pools[action],(finish,replica))
        records.append({'request_id':frame.request_id.iloc[i],'position':i,'stage':stage,'action':action,
            'arrival_s':release,'start_s':start,'finish_s':finish,'replica':replica,
            'service_s':service[action][i],'first_service_s':first[action][i],'queue_wait_s':start-release})
        task=frame.task_type.iloc[i]
        if stage==0 and action=='S':
            delay=check_delay if task in post_tasks else 0.
            checks[i]=float(task in post_tasks)
            if bool(path.upgraded.iloc[i]):
                heapq.heappush(events,(finish+delay,i,1,'L'))
            else: finished[i]=finish+delay
        elif stage==1:
            delay=check_delay if task in vc2_tasks and frame.task_kind.iloc[i]=='multiple_choice' else 0.
            vc_checks[i]=float(delay>0)
            finished[i]=finish+delay
        else: finished[i]=finish
    assert np.all(finished>=arrivals)
    jobs=pd.DataFrame(records)
    np.testing.assert_array_equal(jobs.groupby('position').size().reindex(range(n)),
                                   (path.calls_s+path.calls_l).to_numpy())
    return {'latencies':finished-arrivals,'finish':finished,'jobs':jobs,
            'controller_active_s':float(n*router_delay+check_delay*(checks.sum()+vc_checks.sum())),
            'checks':int(checks.sum()),'vc2_checks':int(vc_checks.sum())}


def energy(core, jobs, horizon, replicas, profile, intensity):





    shared = float(profile.get('shared_w', 1500.0))
    gpu_idle = float(profile.get('gpu_idle_w', 175.0))
    gpu_busy = float(profile.get('gpu_busy_w', 760.0))
    gpu_count = int(profile.get('gpus_per_server', 8))
    pue = float(profile['pue'])
    if not (0 < replicas <= gpu_count and 0 <= gpu_idle <= gpu_busy and shared >= 0 and pue >= 1):
        raise ValueError('Invalid DGX power profile/replica count')
    if horizon <= 0:
        raise ValueError('Observation horizon must be positive')
    base = shared + gpu_count * gpu_idle
    delta = gpu_busy - gpu_idle
    energy_kwh = 0.0
    timelines = []
    for action in ('S', 'L'):
        sub = jobs.loc[jobs.action.eq(action)].copy()
        events = [(0.0, 0), (float(horizon), 0)]
        for record in sub.itertuples():
            start, finish = float(record.start_s), float(record.finish_s)
            if not (0 <= start <= finish <= horizon + 1e-7):
                raise ValueError(f'Job outside common observation horizon: {action}')
            events.extend(((start, 1), (finish, -1)))
        events.sort(key=lambda p:p[0])
        busy = 0
        prev = 0.0
        for t, same in __import__('itertools').groupby(events, key=lambda p:p[0]):
            t = float(t)
            if t > prev:
                if not (0 <= busy <= replicas):
                    raise AssertionError('Concurrent busy GPUs exceed replicas')
                it_w = base + busy * delta
                kwh = it_w * (t-prev) * pue / 3.6e6
                energy_kwh += kwh
                timelines.append({'start_s':prev,'end_s':t,'busy_replicas':busy,
                    'it_power_w':it_w,'facility_kwh':kwh,'emissions_g':kwh*intensity,
                    'action':action})
            busy += sum(v for _,v in same)
            prev=t
        if busy != 0:
            raise AssertionError('Unclosed GPU activity interval')
    analytic=(2*base*horizon + delta*float(jobs.service_s.sum()))*pue/3.6e6
    np.testing.assert_allclose(energy_kwh,analytic,rtol=1e-10,atol=1e-12)
    return energy_kwh,pd.DataFrame(timelines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-project',type=Path,required=True)
    parser.add_argument('--out',type=Path,default=HERE/'runs/experiment')
    args=parser.parse_args();out=args.out.resolve();project=args.source_project.resolve();work=project/'Work 1'
    resource_root=work/'task1_response time, and environ mental impact'
    config=json.loads((resource_root/'config.json').read_text(encoding='utf-8'));profile=dict(config['profile'])
    profile.update({'shared_w':1500.0,'gpu_idle_w':175.0,'gpu_busy_w':760.0,'gpus_per_server':8,'pue':1.2})
    core=load_module(resource_root/'core.py','work1_resource_core')
    cost_meta=json.loads((work/'outputs/cost_summary.json').read_text(encoding='utf-8'))
    fees=pd.read_csv(work/'outputs/cost_summary.csv')
    fee_row=fees[(fees.region=='DE-LU')&(fees.arrival_multiplier==1)&(fees.replicas==1)].iloc[0]
    price=float(fee_row.cost_electricity_eur/fee_row.facility_energy_kwh)
    annual=pd.read_csv(resource_root/'data/processed/annual_carbon_factors.csv')
    ci=float(annual[annual.scenario.eq('EEA_EU27_2024')].iloc[0].intensity)
    state=joblib.load(out/'routing_state.joblib');frame=state['frame'];frame=frame[frame.phase.eq('test')].reset_index(drop=True)
    decisions=pd.read_csv(out/'decisions.csv.gz')
    primary_name=state.get('main_strategy','Selected_protected')
    labels=[primary_name,'All_L','All_S','Pre_direct','Fixed_S_cascade','Random_matched_L_budget','Final_no_Post']
    paths={name:decisions[(decisions.strategy==name)&(decisions.rep==0)].set_index('request_id').loc[frame.request_id].reset_index() for name in labels}
    azure=pd.read_csv(resource_root/'data/processed/azure_code.csv',nrows=len(frame))
    assert len(azure)==len(frame) and np.all(np.diff(azure.arrival_s)>=0)
    permutation=np.random.default_rng(SEED+501).permutation(len(frame))

    frame=frame.iloc[permutation].reset_index(drop=True)
    paths={name:path.iloc[permutation].reset_index(drop=True) for name,path in paths.items()}
    original=azure.arrival_s.to_numpy();original=original-original[0]
    deterministic={}
    for action,suffix in [('S','s'),('L','l')]:
        lengths=pd.DataFrame({'input_tokens':frame.input_tokens_proxy.to_numpy(),'output_tokens':frame['output_'+suffix].to_numpy()})
        deterministic[action]=core.predict_service(lengths,profile,stochastic=False)
    target_span=(len(frame)-1)*float((deterministic['L'][1]*3).mean())/.6
    arrival_shape=original/original[-1]*target_span
    rate=(cost_meta['dgx_price_usd']*cost_meta['usd_to_eur']*cost_meta['dgx_share_per_replica'])/(cost_meta['lifetime_years']*8760)
    chosen=state['parameters'][state['winner']]
    post_tasks={t for t,s in chosen.items() if s['parameters']['gamma'] is not None}-set(state['fallback_tasks'])
    vc2_tasks={t for t,s in chosen.items() if s['parameters']['chooser']=='VC2'}-set(state['fallback_tasks'])
    fixed=json.loads((out/'fixed_cascade_parameters.json').read_text(encoding='utf-8'))
    fixed_post={t for t,s in fixed.items() if s['parameters']['gamma'] is not None}
    fixed_vc={t for t,s in fixed.items() if s['parameters']['chooser']=='VC2'}
    pre_params=state['parameters']['Pre_direct']
    pre_post={t for t,s in pre_params.items() if s['parameters']['gamma'] is not None}
    pre_vc={t for t,s in pre_params.items() if s['parameters']['chooser']=='VC2'}
    records=[];primary_jobs=[];primary_timelines=[];sensitivity=[]
    for scale in [1.,2.,3.]:
        for load in [.5,1.,2.]:
            arrivals=arrival_shape/load
            for replicas in [1,2]:
                for rep in range(3):
                    shock=np.exp(np.random.default_rng(SEED+rep).normal(-.5*profile['residual_log_sigma']**2,profile['residual_log_sigma'],len(frame)))
                    first={a:deterministic[a][0]*shock*(scale if a=='L' else 1) for a in ['S','L']}
                    service={a:deterministic[a][1]*shock*(scale if a=='L' else 1) for a in ['S','L']}
                    cache={}
                    for name,path in paths.items():
                        router=.01 if name not in ['All_L','All_S','Fixed_S_cascade'] else 0.
                        post=post_tasks if name.startswith('Selected') else (fixed_post if name=='Fixed_S_cascade' else pre_post if name=='Pre_direct' else set())
                        peer=vc2_tasks if name==primary_name else pre_vc if name=='Pre_direct' else fixed_vc if name=='Fixed_S_cascade' else set()
                        if name=='Final_no_Post':post=set();peer=set()
                        cache[name]=simulate(frame,path,arrivals,first,service,replicas,router,.002,post,peer,config['network_control_s'])
                    common_horizon=max(max(v['finish']) for v in cache.values())
                    base_energy=None
                    for name,value in cache.items():
                        server,trace=energy(core,value['jobs'],common_horizon,replicas,profile,ci)
                        control=value['controller_active_s']*10*profile['pue']/3.6e6
                        total=server+control
                        hardware=2*replicas*common_horizon/3600*rate
                        electric=total*price
                        records.append({'strategy':name,'large_service_multiplier':scale,'arrival_multiplier':load,
                            'replicas_per_model':replicas,'rep':rep,'requests':len(frame),
                            'assumed_mean_latency_s':float(value['latencies'].mean()),
                            'assumed_p95_latency_s':float(np.quantile(value['latencies'],.95)),
                            'assumed_common_horizon_s':common_horizon,'assumed_server_kwh':server,
                            'assumed_controller_kwh':control,'assumed_total_kwh':total,
                            'assumed_gco2e':total*ci,'assumed_selfhost_electricity_eur':electric,
                            'assumed_selfhost_hardware_eur':hardware,'assumed_selfhost_total_eur':hardware+electric,
                            'historical_api_usd':float(paths[name].cost.sum()),
                            'actual_end_to_end_latency_s':None,'actual_total_energy_kwh':None,
                            'post_checks':value['checks'],'vc2_checks':value['vc2_checks']})
                        if scale==3 and load==1 and replicas==1 and rep==0:
                            primary_jobs.append(value['jobs'].assign(strategy=name))
                            primary_timelines.append(trace.assign(strategy=name))
                    if scale==3 and load==1 and replicas==1 and rep==0:

                        base=cache[primary_name]
                        server,_=energy(core,base['jobs'],common_horizon,replicas,profile,ci)
                        for seconds in [.01,.1,1.,5.]:
                            for watts in [10.,30.,80.]:
                                control_seconds=base['controller_active_s']+len(frame)*(seconds-.01)
                                sensitivity.append({'router_extra_s':seconds,'incremental_control_w':watts,
                                    'assumed_control_kwh':control_seconds*watts*profile['pue']/3.6e6,
                                    'assumed_server_kwh_held_fixed':server,
                                    'scope':'control_energy_only_holds_schedule_fixed_not_latency_resimulation'})
    pd.DataFrame(records).to_csv(out/'resource_scenarios.csv',index=False)
    pd.DataFrame(sensitivity).to_csv(out/'control_energy_sensitivity.csv',index=False)
    pd.concat(primary_jobs).to_csv(out/'primary_call_schedule.csv.gz',index=False,compression='gzip')
    pd.concat(primary_timelines).to_csv(out/'primary_power_timeline.csv.gz',index=False,compression='gzip')
    grouped=pd.DataFrame(records).groupby(['strategy','large_service_multiplier','arrival_multiplier','replicas_per_model'],sort=False)
    summary=grouped[['assumed_mean_latency_s','assumed_p95_latency_s','assumed_total_kwh','assumed_gco2e',
                     'assumed_selfhost_total_eur','assumed_selfhost_electricity_eur','assumed_selfhost_hardware_eur']].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(c).rstrip('_') if isinstance(c,tuple) else c for c in summary.columns]
    summary.to_csv(out/'resource_summary.csv',index=False)
    save(out/'resource_assumptions.json',{'source_core':str(resource_root/'core.py'),'source_sha256':digest(resource_root/'core.py'),
        'profile':profile,'large_service_multipliers':[1,2,3],'primary_large_service_multiplier':3,
        'arrival_shape_source':str(resource_root/'data/processed/azure_code.csv'),
        'arrival_assignment':'random RouterBench request order mapped to Azure gap shape only; not real same-request data',
        'reference_arrival_span_s':target_span,'reference_utilization_L_pool':.6,
        'fixed_common_infrastructure':'two separately powered 8-GPU DGX boxes (one S pool, one L pool); 1 or 2 active GPU serving slots per pool, unoccupied GPUs still idle; both powered for entire common horizon',
        'power_model':'per DGX: shared_w + busy_gpus*gpu_busy_w + (8-busy_gpus)*gpu_idle_w',
        'fixed_common_observation_horizon':'max final completion among compared strategies in each scenario',
        'controller':{'router_s':.01,'post_s':.002,'incremental_w':10,'measured':False},
        'lengths':'regex proxies, not actual model tokenizer counts; current output used only for ex-post resource evaluation',
        'carbon_factor':ci,'carbon_unit':'gCO2e/kWh','carbon_boundary':'annual_production_ghg_EEA_not_marginal',
        'constant_price_eur_per_kwh':price,
        'price_derivation':'Work 1 DE-LU load1/replica1 electricity EUR / facility kWh; held constant in a new counterfactual, not a current retail tariff',
        'raw_energy_charts_price_series_available':False,
        'dgx_power_parameters_source':'shared and idle assumed; busy GPU from specific ML.ENERGY workload proxy, not measured whole-system or GPT-4 power',
        'capex':{k:cost_meta[k] for k in ['dgx_price_usd','usd_to_eur','dgx_share_per_replica','lifetime_years']},
        'matching_warning':'GPT-4 hardware is unknown and not deployable on the assumed DGX; these are anonymous-backend scenarios',
        'api_and_selfhost_costs_added':False,'measured_GPT4_or_Mixtral_energy':False,
        'repetitions':3,'common_random_numbers':'same mean-one request shock for S/L and all strategies',
        'uncertainty_scope':'conditional Monte Carlo only, not fitted hardware uncertainty'})
    print('RESOURCE SCENARIOS',len(records),flush=True)


if __name__=='__main__':main()
