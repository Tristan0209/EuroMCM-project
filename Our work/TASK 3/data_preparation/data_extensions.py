import pandas as pd

from common import save_json, utc_strings
from feasibility_precheck import write_prechecks


def build_extensions(output, requests, servers, network, cfg):
    base=pd.read_csv(output/'server_availability.csv')
    anchor=pd.Timestamp(cfg['start_datetime_utc'])
    scenarios={}
    for name,overrides in cfg['availability_scenarios'].items():
        rows=base.to_dict('records')
        for event in overrides:
            rows.append({'datetime_utc':(anchor+pd.Timedelta(seconds=event['offset_seconds'])).isoformat(),
                'region':event['region'],'model_type':event['model_type'],
                'available_capacity':event['available_capacity'],'provenance':'task3_assumed_'+name})
        table=pd.DataFrame(rows)
        table.to_csv(output/('server_availability_'+name+'.csv'),index=False)
        scenarios[name]=table
    pd.concat([table.assign(scenario=name) for name,table in scenarios.items()],ignore_index=True).to_csv(
        output/'server_availability_scenarios.csv',index=False)
    classes={c['name']:c for c in cfg['service_classes']}
    timing=requests[['request_id','arrival_time','deadline','service_class']].copy()
    timing['max_queue_wait_seconds']=[classes[c]['max_queue_wait_seconds'] for c in requests.service_class]
    timing['latest_start_without_network_utc']=utc_strings(pd.to_datetime(requests.deadline,utc=True)
        -pd.to_timedelta(requests.estimated_service_time_seconds,unit='s'))
    timing['wait_definition']='queue_after_known_network_and_control_overheads'
    timing.to_csv(output/'request_timing_constraints.csv',index=False)
    save_json(output/'resource_accounting_contract.json',{
        'estimated_energy_kwh':'incremental facility busy-over-idle energy only',
        'increment_formula':'service_seconds*(p_busy_w-p_idle_w)*pue/3.6e6',
        'idle_base_formula':'configured_capacity*p_idle_w*common_horizon_seconds*pue/3.6e6',
        'facility_total':'idle_base + busy_increment + controller + network; each charged exactly once',
        'equivalent_two_state_formula':'pue*(busy_w*busy_server_seconds+idle_w*(capacity*horizon-busy_server_seconds))/3.6e6',
        'reference_service_already_scaled':True,
        'regional_adjustment':'reference_service_seconds * pool_multiplier / reference_model_multiplier',
        'forbidden':'do not multiply reference_service_seconds by the S/L multiplier again',
        'unavailability_power':'unavailable does not mean powered_off; configured idle base unchanged unless a separate measured power-state model is explicitly introduced',
        'high_load_capacity_reduction':'assumed external contention removes usable slots; no extra RouterBench arrivals or external-job energy is fabricated',
        'scope':'arithmetic interface only; no DES integration or schedules implemented'})
    summaries=write_prechecks(output,requests,servers,network,cfg,scenarios)
    return summaries
