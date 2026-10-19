from itertools import combinations

import numpy as np
import pandas as pd

from common import save_json
from carbon_loader import utc_column
from resource_contracts import service_at_pool


def capacity_intervals(availability, servers, start, end):
    events = availability.copy()
    events['time'] = utc_column(events.datetime_utc,'availability datetime_utc').astype('int64')/1e9
    if events.duplicated(['time','region','model_type']).any():
        raise ValueError('Duplicate availability event')
    lookup = servers.set_index(['region','model_type']).capacity.to_dict()
    for event in events.itertuples():
        key = (event.region,event.model_type)
        if (key not in lookup or not np.isfinite(event.available_capacity) or event.available_capacity % 1
                or not 0 <= event.available_capacity <= lookup[key]):
            raise ValueError('Availability outside fixed infrastructure capacity')
    result = {}
    for key in lookup:
        block = events[(events.region==key[0])&(events.model_type==key[1])].sort_values('time')
        prior = block[block.time.le(start)]
        if prior.empty:
            raise ValueError('Missing initial availability for '+str(key))
        c = int(prior.iloc[-1].available_capacity)
        last = start
        intervals = []
        for event in block[block.time.gt(start)&block.time.lt(end)].itertuples():
            intervals.append((last,event.time,c))
            last,c = event.time,int(event.available_capacity)
        intervals.append((last,end,c))
        result[key] = intervals
    return result


def isolated_earliest_start(intervals, ready, latest_start, service):


    positive = []
    for left,right,capacity in intervals:
        if capacity <= 0:
            continue
        if positive and abs(positive[-1][1]-left)<1e-7:
            positive[-1] = (positive[-1][0],right)
        else:
            positive.append((left,right))
    for left,right in positive:
        candidate=max(left,ready)
        if candidate <= latest_start+1e-7 and candidate+service <= right+1e-7:
            return candidate
    return None


def integral_capacity(intervals, start, end):
    return sum(max(0,min(right,end)-max(left,start))*capacity for left,right,capacity in intervals)


def precheck(requests, servers, network, availability, cfg, scenario):
    arrivals = utc_column(requests.arrival_time,'arrival_time').astype('int64').to_numpy()/1e9
    deadlines = utc_column(requests.deadline,'deadline').astype('int64').to_numpy()/1e9
    start,end = float(arrivals.min()),float(deadlines.max())
    if np.any(deadlines<arrivals):
        raise ValueError('Completion deadline precedes arrival')
    intervals = capacity_intervals(availability,servers,start,end)
    pool_multipliers = servers.set_index(['region','model_type']).service_time_multiplier.to_dict()
    routes={origin:list(block.itertuples()) for origin,block in network[network.allowed.eq(1)].groupby('origin_region')}
    classes = {c['name']:c for c in cfg['service_classes']}
    router = cfg['precheck']['router_seconds'] + cfg['precheck']['network_control_seconds']
    if not np.isfinite(router) or router < 0:
        raise ValueError('Invalid fixed control overhead')
    records, eligible = [], []
    optimistic_work = []
    for i,r in enumerate(requests.itertuples()):
        wait = classes[r.service_class]['max_queue_wait_seconds']
        if not np.isfinite(wait) or wait < 0:
            raise ValueError('Maximum queue wait must be finite and nonnegative')
        reference_multiplier=1 if r.selected_model=='S' else cfg['large_service_multiplier']
        destinations = routes.get(r.origin_region,[])
        finishes, starts, latests, queue_waits, services, regions = [], [], [], [], [], set()
        for link in destinations:
            key=(link.destination_region,r.selected_model)
            if key not in pool_multipliers:
                continue
            service = service_at_pool(r.estimated_service_time_seconds,reference_multiplier,pool_multipliers[key])
            ready = arrivals[i]+router+link.latency_ms/1000
            latest = min(ready+wait,deadlines[i]-service)
            earliest = isolated_earliest_start(intervals[key],ready,latest,service)
            regions.add(link.destination_region)
            services.append(service)
            latests.append(latest)
            if earliest is not None:
                finishes.append(earliest+service)
                starts.append(earliest)
                queue_waits.append(max(0,earliest-ready))
        possible=bool(finishes)
        eligible.append(regions)
        optimistic_work.append(min(services) if services else r.estimated_service_time_seconds)
        as_utc=lambda value:pd.Timestamp(value,unit='s',tz='UTC').isoformat() if value is not None else None
        records.append({'scenario':scenario,'request_id':r.request_id,'model_type':r.selected_model,
            'max_queue_wait_seconds':wait,'completion_deadline':r.deadline,
            'latest_start_upper_bound_utc':as_utc(max(latests) if latests else None),
            'isolated_earliest_start_utc':as_utc(min(starts) if starts else None),
            'isolated_earliest_finish_utc':as_utc(min(finishes) if finishes else None),
            'minimum_required_queue_wait_seconds':min(queue_waits) if queue_waits else None,
            'isolated_possible':possible,'allowed_regions':'|'.join(sorted(regions)),
            'interpretation':'necessary_only_not_joint_schedule_feasibility'})
    individual = pd.DataFrame(records)




    points = np.unique(np.r_[arrivals,deadlines])
    n=min(int(cfg['precheck']['sampled_time_boundaries']),len(points))
    boundaries=np.unique(np.r_[start,end,np.quantile(points,np.linspace(0,1,n))])
    regions=sorted(servers.region.unique())
    subsets=[set(c) for k in range(1,len(regions)+1) for c in combinations(regions,k)]
    work=np.array(optimistic_work)
    models=requests.selected_model.to_numpy()
    bounds=[]
    for model in ['S','L']:
        for subset in subsets:
            relevant=np.array([bool(e) and e.issubset(subset) for e in eligible])&(models==model)
            for left,right in combinations(boundaries,2):
                mask=relevant&(arrivals>=left-1e-7)&(deadlines<=right+1e-7)
                if not mask.any():
                    continue
                demand=float(work[mask].sum())
                capacity=sum(integral_capacity(intervals[(region,model)],left,right) for region in subset)
                bounds.append({'scenario':scenario,'model_type':model,'region_subset':'|'.join(sorted(subset)),
                    'window_start_utc':pd.Timestamp(left,unit='s',tz='UTC').isoformat(),
                    'window_end_utc':pd.Timestamp(right,unit='s',tz='UTC').isoformat(),
                    'contained_requests':int(mask.sum()),'required_server_seconds_lower_bound':demand,
                    'available_server_seconds':capacity,'necessary_bound_satisfied':bool(demand<=capacity+1e-7)})
    bound_frame=pd.DataFrame(bounds)
    violations=int((~bound_frame.necessary_bound_satisfied).sum()) if len(bound_frame) else 0
    summary={'scenario':scenario,'requests':len(requests),'individually_impossible_under_estimates':int((~individual.isolated_possible).sum()),
        'sampled_capacity_bounds':len(bounds),'violated_capacity_bounds':violations,
        'necessary_checks_satisfied':bool(individual.isolated_possible.all() and violations==0),
        'joint_feasibility':'not_established_no_scheduler_or_optimization_run',
        'uncertainty':'deterministic prior estimates only; stochastic overruns not guaranteed',
        'queue_wait_definition':'time from arrival+router+network_control+one_way_transfer until service starts',
        'capacity_semantics':'positive capacity must persist through isolated nonpreemptive service; no assumption about actual in-flight cancellation',
        'sample_scope':'all region subsets, sampled windows only; passing does not prove all capacity inequalities'}
    return individual,bound_frame,summary


def write_prechecks(output, requests, servers, network, cfg, scenarios):
    individual,bounds,summaries=[],[],[]
    for name,availability in scenarios.items():
        i,b,s=precheck(requests,servers,network,availability,cfg,name)
        individual.append(i);bounds.append(b);summaries.append(s)
    pd.concat(individual,ignore_index=True).to_csv(output/'request_feasibility_precheck.csv',index=False)
    pd.concat(bounds,ignore_index=True).to_csv(output/'capacity_precheck.csv',index=False)
    save_json(output/'feasibility_precheck.json',{'scope':'necessary_conditions_only','scenarios':summaries,
        'no_MILP_MPC_DES_models_built':True})
    return summaries
