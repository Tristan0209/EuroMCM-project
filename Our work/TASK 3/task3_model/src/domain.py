from dataclasses import dataclass,replace
import hashlib
import numpy as np
import pandas as pd
from .util import DATA,PREP,read
from .power import incremental_w


@dataclass(frozen=True)
class Request:
    request_id:str
    model:str
    origin:str
    task_type:str
    service_class:str
    arrival:float
    deadline:float
    wait:float
    service:float
    priority:int


def load_inputs(n=None,load=1.,delay_scale=1.,defer_fraction=1.,network_scale=1.):
    frame=pd.read_csv(DATA/'requests.csv')
    if n is not None and n<len(frame):

        groups=list(frame.groupby(['task_type','selected_model','service_class'],sort=True))
        counts=np.array([len(g) for _,g in groups],float)*n/len(frame)
        quotas=np.floor(counts).astype(int)
        for j in np.argsort(-(counts-quotas),kind='stable')[:n-quotas.sum()]:quotas[j]+=1
        indices=[]
        for (_,g),q in zip(groups,quotas):
            if q:indices.extend(g.index[np.linspace(0,len(g)-1,q).round().astype(int)])
        frame=frame.loc[sorted(indices)].copy()
    anchor=pd.Timestamp(read(DATA/'alignment.json')['anchor_utc'])
    rules={c['name']:c for c in read(PREP/'config.json')['service_classes']}
    requests=[]
    for r in frame.itertuples():
        arrival=(pd.Timestamp(r.arrival_time)-anchor).total_seconds()/load
        budget=r.max_delay_seconds*(delay_scale if r.service_class!='realtime' else 1.)
        can_defer=int(hashlib.sha256(r.request_id.encode()).hexdigest()[:8],16)/2**32<defer_fraction
        wait=rules[r.service_class]['max_queue_wait_seconds']*delay_scale
        if not can_defer:wait=min(wait,5.)
        requests.append(Request(r.request_id,r.selected_model,r.origin_region,r.task_type,r.service_class,
            arrival,arrival+budget,wait,r.estimated_service_time_seconds,r.priority))
    servers={}
    config=pd.read_csv(DATA/'server_config.csv')
    hosts=pd.read_csv(DATA/'physical_host_config.csv').set_index('host_id')
    if config.host_id.duplicated().any() or hosts.index.duplicated().any():raise ValueError('Hosts must not be double charged across pools')
    if set(config.host_id)!=set(hosts.index):raise ValueError('Host/pool mapping is incomplete')
    for s in config.itertuples():
        host=hosts.loc[s.host_id]
        base=host.shared_w+host.gpu_count_per_host*host.gpu_idle_w
        delta=host.gpu_busy_w-host.gpu_idle_w
        if not (0<s.capacity<=host.gpu_count_per_host and int(s.capacity)==s.capacity):raise ValueError('Invalid serving slot capacity')
        if not np.isclose(s.p_idle_w,base) or not np.isclose(s.p_busy_w,base+delta):raise ValueError('Host power alias disagrees with explicit GPU profile')
        if base+host.gpu_count_per_host*delta>host.server_max_w:raise ValueError('Host exceeds IT power ceiling')
        servers[(s.region,s.model_type)]={'capacity':int(s.capacity),'idle':s.p_idle_w,'busy':s.p_busy_w,
            'pue':s.pue,'idle_base_w':base,'host_id':s.host_id,'installed_gpus':int(host.gpu_count_per_host)}
    network={}
    for s in pd.read_csv(DATA/'network_config.csv').itertuples():
        network[(s.origin_region,s.destination_region)]={'delay':.05+s.latency_ms/1000*network_scale,
            'energy':s.transfer_energy_kwh,'allowed':bool(s.allowed)}
    return requests,servers,network,anchor


class Availability:
    def __init__(self,scenario,anchor):
        f=pd.read_csv(DATA/('server_availability_'+scenario+'.csv'))
        self.events=sorted(((pd.Timestamp(r.datetime_utc)-anchor).total_seconds(),r.region,r.model_type,int(r.available_capacity)) for r in f.itertuples())
        self.series={}
        for t,r,m,c in self.events:self.series.setdefault((r,m),[]).append((t,c))

    def at(self,region,model,t):
        value=None
        for when,c in self.series[(region,model)]:
            if when<=t+1e-9:value=c
            else:break
        if value is None:raise ValueError('Availability does not cover start')
        return value


def shocks_for(rep,ids):
    f=pd.read_csv(DATA/'service_random_conditions.csv')
    series=f[f.rep.eq(rep)].set_index('request_id').service_multiplier_shock
    if not set(ids).issubset(series.index):raise ValueError('Missing common random conditions')
    return series.loc[ids].to_dict()
