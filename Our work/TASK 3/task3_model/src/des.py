import heapq
import itertools
import math
import time
from collections import defaultdict
import pandas as pd

from .carbon import ForecastView,OracleView
from .milp import solve,gap_start

DEFAULTS={'router_seconds':.01,'solve_delay_seconds':.01,'measured_wall_fraction':0.,'service_guard':1.35,
    'horizon_seconds':10800.,'reoptimization_seconds':300.,'arrival_batch_seconds':.5,'candidate_steps':5,
    'max_batch':64,'oracle_batch':96,'solver_time_limit':.2,'forecast_method':'persistence','forecast_noise':0.,
    'seed':20261017,'controller_w':10.,'controller_region':'DE','record_events':True}


class Simulation:
    priorities={'ServiceCompletion':0,'CapacityChange':1,'QueueReady':2,'RequestArrival':3,'RoutingDecision':4,
        'Commit':5,'MPCReoptimization':5,'Dispatch':6,'Expire':8,'DeadlineMiss':9}

    def __init__(self,requests,servers,network,availability,carbon,shocks,strategy,cfg=None):
        self.requests=list(requests);self.q={q.request_id:q for q in requests}
        self.servers=servers;self.network=network;self.availability=availability;self.actual=carbon
        self._shocks=shocks;self.strategy=strategy;self.cfg={**DEFAULTS,**(cfg or {})}
        self.events=[];self.counter=itertools.count();self.now=0.
        self.rec={q.request_id:{'request_id':q.request_id,'selected_model':q.model,'task_type':q.task_type,
            'service_class':q.service_class,'origin_region':q.origin,'arrival_s':q.arrival,'deadline_s':q.deadline,
            'max_wait_s':q.wait,'estimated_service_s':q.service,'status':'not_arrived','version':0,
            'failure_reason':'','fallback_reason':'','region':None,'replica':None} for q in requests}
        self.running={};self.queues=defaultdict(list);self.pending=set();self.log=[];self.solver_log=[]
        self.controls=[];self.forecasts=[];self.forecast_audit=[];self.busy_until=0.
        self.opt_token=0;self.opt_scheduled=math.inf;self.fallbacks=0;self.actual_starts=0
        self.current_capacity={(r,m):availability.at(r,m,0.) for r,m in servers}
        minimum_end=max(q.deadline for q in requests)+max(q.service for q in requests)*20+60
        self.end=self.cfg.get('observation_horizon_seconds',minimum_end)
        if self.end<max(q.deadline for q in requests):raise ValueError('Common observation horizon precedes input deadlines')
        self.oracle_plans={}

    def emit(self,kind,rid=None,**extra):
        if self.cfg['record_events']:self.log.append({'time_s':self.now,'event':kind,'request_id':rid,**extra})

    def push(self,t,kind,payload=None):
        if t<self.now-1e-5:raise ValueError('Event scheduled in the past')
        heapq.heappush(self.events,(max(float(t),self.now),self.priorities[kind],next(self.counter),kind,payload))

    def link(self,q,r):return self.network[(q.origin,r)]

    def allowed(self,q):return [r for r,m in self.servers if m==q.model and self.link(q,r)['allowed']]

    def forecast(self,t):
        view=ForecastView(self.actual,t,self.cfg['horizon_seconds'],self.cfg['forecast_method'],self.cfg['forecast_noise'],self.cfg['seed'])
        self.forecast_audit.append({'issued_s':t,'observed_end_s':view.cutoff,'method':view.method})
        if self.cfg.get('export_forecasts',False):self.forecasts.extend(view.records())
        return view

    def reservations(self):
        result=defaultdict(list);guard=self.cfg['service_guard']
        for key,rid in self.running.items():
            rec=self.rec[rid];q=self.q[rid]

            end=max(rec['service_start_s']+q.service*guard,self.now+max(.01,q.service*.1))
            result[key].append((self.now,end))
        for (r,m),ids in self.queues.items():
            for rid in ids:
                rec=self.rec[rid]
                if rec['status']!='queued':continue
                candidates=[]
                for j in range(self.servers[(r,m)]['capacity']):
                    key=(r,m,j);start=gap_start(self.now,self.q[rid].service*guard,result[key])
                    candidates.append((start,j))
                start,j=min(candidates);result[(r,m,j)].append((start,start+self.q[rid].service*guard))
        for rid,rec in self.rec.items():
            if rec['status']=='planned':
                result[(rec['region'],self.q[rid].model,rec['replica'])].append((max(self.now,rec['planned_start_s']),
                    max(self.now,rec['planned_start_s'])+self.q[rid].service*guard))
        return result

    def latest_dispatch(self,q):
        return max((min(q.arrival+self.cfg['router_seconds']+q.wait,
            q.deadline-q.service*self.cfg['service_guard']-self.link(q,r)['delay']) for r in self.allowed(q)),default=q.arrival)

    def fail(self,rid,reason):
        rec=self.rec[rid]
        if rec['status'] in {'running','completed','failed'}:return
        rec.update(status='failed',failure_reason=reason,terminal_s=self.now)
        self.pending.discard(rid);rec['version']+=1

    def dispatch(self,rid,r,replica=None):
        rec=self.rec[rid];q=self.q[rid]
        if rec['status'] in {'running','completed','failed','inflight','queued'}:return
        if r not in self.allowed(q):self.fail(rid,'no_allowed_region');return
        rec.update(status='inflight',region=r,replica=replica,dispatch_s=self.now)
        self.pending.discard(rid)
        rec['ready_s']=self.now+self.link(q,r)['delay']
        self.emit('Dispatch',rid,region=r)
        self.push(rec['ready_s'],'QueueReady',rid)

    def drain(self,r,m):
        queue=self.queues[(r,m)]
        queue[:]=[rid for rid in queue if self.rec[rid]['status']=='queued']
        queue.sort(key=lambda rid:(self.rec[rid]['ready_s'],self.rec[rid]['sequence']))
        while queue:
            free=[j for j in range(self.current_capacity[(r,m)]) if (r,m,j) not in self.running]
            if not free:return
            rid=queue[0];rec=self.rec[rid];q=self.q[rid]
            latest=min(q.arrival+self.cfg['router_seconds']+self.link(q,r)['delay']+q.wait,
                q.deadline-q.service*self.cfg['service_guard'])
            if self.now>latest+1e-6:
                queue.pop(0);self.fail(rid,'waiting_or_predicted_completion_limit');continue

            replica=rec['replica'] if rec['replica'] in free else free[0]
            queue.pop(0)
            rec.update(status='running',replica=replica,service_start_s=self.now,
                actual_service_s=q.service*self._shocks[rid])
            rec['queue_wait_s']=self.now-rec['ready_s']
            rec['intentional_delay_s']=rec['dispatch_s']-q.arrival-self.cfg['router_seconds']
            rec['network_delay_s']=self.link(q,r)['delay']
            self.running[(r,m,replica)]=rid;self.actual_starts+=1
            self.emit('ServiceStart',rid,region=r,model=m,replica=replica,
                active=sum(1 for key in self.running if key[:2]==(r,m)),admission_capacity=self.current_capacity[(r,m)])
            self.push(self.now+rec['actual_service_s'],'ServiceCompletion',rid)

    def greedy_region(self,q,earliest=False):
        started=time.perf_counter();booked=self.reservations();candidates=[]
        view=None if earliest else self.forecast(self.now)
        for r in self.allowed(q):
            p=self.servers[(r,q.model)];net=self.link(q,r)['delay'];release=self.now+net
            starts=[gap_start(release,q.service*self.cfg['service_guard'],booked.get((r,q.model,j),[]))
                for j in range(self.current_capacity[(r,q.model)])]
            if not starts:continue
            start=min(starts)
            if (start>q.arrival+self.cfg['router_seconds']+net+q.wait+1e-7 or
                    start+q.service*self.cfg['service_guard']>q.deadline+1e-7):continue
            immediate=start<=release+1e-6
            cost=0 if earliest else p['pue']*(p['busy']-p['idle'])/3.6e6*view.integral(start,start+q.service,r)
            if not earliest:cost+=self.link(q,r)['energy']*(view.at(self.now,q.origin)+view.at(self.now,r))/2
            candidates.append((not immediate,start,cost,r))
        if candidates:
            if earliest:region=min(candidates,key=lambda x:(x[1],x[3]))[3]
            else:
                immediate=[v for v in candidates if not v[0]]
                region=min(immediate or candidates,key=lambda x:(x[2],x[1],x[3]))[3]
        else:region=q.origin if q.origin in self.allowed(q) else next(iter(self.allowed(q)),None)
        return region,time.perf_counter()-started

    def fallback(self,rid,reason):
        self.fallbacks+=1;self.rec[rid]['fallback_reason']=reason
        region,_=self.greedy_region(self.q[rid],earliest=True)
        if region is None:self.fail(rid,'no_allowed_compatible_pool')
        else:self.dispatch(rid,region)

    def request_optimization(self,t):
        t=max(t,self.busy_until,self.now)
        if t<self.opt_scheduled-1e-8:
            self.opt_token+=1;self.opt_scheduled=t
            self.push(t,'MPCReoptimization',self.opt_token)

    def optimize(self):
        ids=sorted((rid for rid in self.pending if self.rec[rid]['status']=='pending'),
            key=lambda rid:(self.q[rid].deadline,self.q[rid].arrival,rid))[:self.cfg['max_batch']]
        if not ids:return
        jobs=[self.q[rid] for rid in ids]
        if any(q.arrival>self.now+1e-8 for q in jobs):raise AssertionError('MPC read a future arrival')
        view=self.forecast(self.now);observed=dict(self.current_capacity)
        plans,log=solve(jobs,self.now,self.servers,self.network,self.reservations(),
            lambda r,m,t:observed[(r,m)],view,self.cfg)
        duration=self.cfg['solve_delay_seconds']+self.cfg['measured_wall_fraction']*log['wall_seconds']
        log.update({'decision_s':self.now,'observed_carbon_end_s':view.cutoff,'max_request_arrival_s':max(q.arrival for q in jobs),
            'request_ids':ids,'modelled_controller_s':duration})
        self.solver_log.append(log);self.controls.append((self.now,self.now+duration,self.cfg['controller_region'],'MILP_MPC'))
        self.busy_until=self.now+duration
        self.push(self.busy_until,'Commit',(ids,plans,log['status']))

    def prepare_oracle(self):
        booked=defaultdict(list);jobs=sorted(self.requests,key=lambda q:(q.arrival,q.deadline,q.request_id))
        planning_time=0.;batch=self.cfg['oracle_batch'];view=OracleView(self.actual)

        for offset in range(0,len(jobs),batch):
            group=jobs[offset:offset+batch]
            earliest=min(q.arrival for q in group)


            for key in booked:booked[key]=[(left,right) for left,right in booked[key] if right>earliest-1e-8]
            cfg={**self.cfg,'horizon_seconds':self.end+1}
            plans,log=solve(group,planning_time,self.servers,self.network,booked,self.availability.at,view,cfg,
                extra_candidates=[t for t,_,_,c in self.availability.events if c>0])
            duration=self.cfg['solve_delay_seconds']+self.cfg['measured_wall_fraction']*log['wall_seconds']
            log.update({'decision_s':planning_time,'request_ids':[q.request_id for q in group],
                'modelled_controller_s':duration,'oracle_batch_restricted':True})
            self.solver_log.append(log);self.controls.append((planning_time,planning_time+duration,self.cfg['controller_region'],'MILP_ORACLE'))
            if log['status']==0:
                for plan in plans:
                    self.oracle_plans[plan['request_id']]=plan
                    booked[(plan['region'],plan['model'],plan['replica'])].append((plan['start'],plan['end']))
            planning_time+=duration

    def run(self):
        if self.strategy=='Oracle':self.prepare_oracle()
        for q in self.requests:self.push(q.arrival,'RequestArrival',q.request_id)
        for event in self.availability.events:
            if 0<event[0]<=self.end:self.push(event[0],'CapacityChange',event[1:])
        if self.strategy=='MPC':self.request_optimization(0.)
        while self.events:
            t,_,_,kind,payload=heapq.heappop(self.events)
            if t>self.end+1e-6:raise RuntimeError('Observation window fails to cover drain; no silent truncation')
            self.now=t
            if kind=='RequestArrival':
                rid=payload;rec=self.rec[rid];q=self.q[rid]
                rec.update(status='routing',sequence=next(self.counter));self.emit(kind,rid)
                self.controls.append((t,t+self.cfg['router_seconds'],q.origin,'frozen_router'))
                self.push(t+self.cfg['router_seconds'],'RoutingDecision',rid)
                self.push(q.deadline,'DeadlineMiss',rid)
            elif kind=='RoutingDecision':
                rid=payload;q=self.q[rid];rec=self.rec[rid]
                if rec['status']=='failed':continue
                rec['status']='pending';self.emit(kind,rid,selected_model=q.model)
                self.push(max(t,self.latest_dispatch(q))+1e-6,'Expire',(rid,'pending'))
                if self.strategy=='Fixed':self.dispatch(rid,q.origin)
                elif self.strategy=='Greedy':
                    region,wall=self.greedy_region(q)
                    if region is None:self.fail(rid,'no_allowed_pool')
                    else:self.dispatch(rid,region)
                elif self.strategy=='Oracle':
                    plan=self.oracle_plans.get(rid)
                    if plan is None:self.fallback(rid,'oracle_restricted_plan_unassigned_or_timeout')
                    else:
                        rec.update(status='planned',region=plan['region'],replica=plan['replica'],planned_start_s=plan['start'])
                        self.push(max(t,plan['dispatch']),'Dispatch',(rid,rec['version']))
                else:
                    self.pending.add(rid)
                    next_tick=(math.floor(t/self.cfg['reoptimization_seconds'])+1)*self.cfg['reoptimization_seconds']
                    urgency=self.latest_dispatch(q)-self.cfg['solve_delay_seconds']-.001

                    self.request_optimization(min(next_tick,max(t,min(t+self.cfg['arrival_batch_seconds'],urgency))))
            elif kind=='MPCReoptimization':
                if payload!=self.opt_token:continue
                self.opt_scheduled=math.inf;self.emit(kind)
                self.optimize()
                if self.pending:
                    future=(math.floor(t/self.cfg['reoptimization_seconds'])+1)*self.cfg['reoptimization_seconds']
                    if future<=max(q.deadline for q in self.requests):self.request_optimization(future)
            elif kind=='Commit':
                ids,plans,status=payload
                assigned={p['request_id']:p for p in plans}
                for rid in ids:
                    rec=self.rec[rid]
                    if rec['status']!='pending':continue
                    plan=assigned.get(rid)
                    if status!=0 or plan is None:self.fallback(rid,'solver_status_'+str(status) if status!=0 else 'MILP_unassigned')
                    elif plan['start']<=t+self.cfg['reoptimization_seconds']+1e-8:
                        rec.update(status='planned',region=plan['region'],replica=plan['replica'],planned_start_s=plan['start'])
                        self.pending.discard(rid)
                        self.push(max(t,plan['dispatch']),'Dispatch',(rid,rec['version']))

            elif kind=='Dispatch':
                rid,version=payload;rec=self.rec[rid]
                if rec['status']=='planned' and rec['version']==version:self.dispatch(rid,rec['region'],rec['replica'])
            elif kind=='QueueReady':
                rid=payload;rec=self.rec[rid];q=self.q[rid]
                if rec['status']!='inflight':continue
                rec['status']='queued';self.queues[(rec['region'],q.model)].append(rid)
                latest=min(q.arrival+self.cfg['router_seconds']+self.link(q,rec['region'])['delay']+q.wait,
                    q.deadline-q.service*self.cfg['service_guard'])
                self.push(max(t,latest)+1e-6,'Expire',(rid,'queued'))
                self.drain(rec['region'],q.model)
            elif kind=='ServiceCompletion':
                rid=payload;rec=self.rec[rid];q=self.q[rid]
                if rec['status']!='running':raise AssertionError('Duplicate or nonrunning completion')
                del self.running[(rec['region'],q.model,rec['replica'])]
                rec.update(status='completed',completion_s=t,terminal_s=t,response_s=t-q.arrival)
                self.emit(kind,rid,region=rec['region']);self.drain(rec['region'],q.model)
            elif kind=='CapacityChange':
                r,m,c=payload;self.current_capacity[(r,m)]=c;self.emit(kind,region=r,model=m,capacity=c)
                if self.strategy=='MPC':
                    for rid,rec in self.rec.items():
                        if (rec['status']=='planned' and rec['region']==r and self.q[rid].model==m
                                and rec['replica']>=c):
                            rec.update(status='pending',version=rec['version']+1,replan_reason='observed_capacity_change')
                            self.pending.add(rid)
                self.drain(r,m)
                if self.strategy=='MPC' and self.pending:self.request_optimization(t)
            elif kind=='Expire':
                rid,stage=payload;rec=self.rec[rid]
                if stage=='pending' and rec['status'] in {'pending','planned','routing'}:self.fail(rid,'max_wait_or_predicted_deadline')
                elif stage=='queued' and rec['status']=='queued':
                    region=rec['region'];self.fail(rid,'queue_wait_or_predicted_deadline');self.drain(region,self.q[rid].model)
            elif kind=='DeadlineMiss':
                rid=payload;rec=self.rec[rid]
                if rec['status']!='completed':
                    self.emit(kind,rid,status=rec['status'])
                    if rec['status'] not in {'running','failed'}:self.fail(rid,'completion_deadline')
        if any(rec['status'] not in {'completed','failed'} for rec in self.rec.values()):raise AssertionError('Request disappeared without terminal outcome')
        if self.actual_starts!=sum(rec['status']=='completed' for rec in self.rec.values()):raise AssertionError('Repeated/interrupted service')
        return self
