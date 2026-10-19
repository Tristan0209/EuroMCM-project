import heapq
import math
from collections import Counter
from .sources import ROOT, load_final

ENGINE, SPATIAL, SOURCE_IDENTITIES = load_final(ROOT / 'audit/review_v2/scratch')


class WindowMixin:
    public_policy = None

    def __init__(self, requests, servers, network, availability, carbon, shocks, cfg=None, horizon=43200., forecast_factory=None):
        self._forecast_factory = forecast_factory
        requests = list(requests)
        if not requests or not math.isfinite(horizon) or horizon <= 0:
            raise ValueError('Nonempty window cohort required')
        if len({q.request_id for q in requests}) != len(requests):
            raise ValueError('Duplicate request ID')
        for q in requests:
            if not all(math.isfinite(v) for v in [q.arrival, q.deadline, q.wait, q.service]) or not (
                    0 <= q.arrival < horizon and q.deadline > q.arrival and q.wait >= 0 and q.service > 0):
                raise ValueError('Invalid pre-arrival/request/window contract')
        cfg = dict(cfg or {})
        if cfg.get('measured_wall_fraction', 0.) != 0:
            raise ValueError('Measured wall shift/revalidation adapter has not been validated')


        initial = {**cfg, 'observation_horizon_seconds': max(horizon, max(q.deadline for q in requests))}
        super().__init__(requests, servers, network, availability, carbon, shocks,
                         'MPC' if self.public_policy == 'Spatial_MILP' else self.public_policy, cfg=initial)
        self.end = float(horizon)
        self.cfg['observation_horizon_seconds'] = float(horizon)
        self._observer_has_run = False

    def forecast(self, t):
        if self._forecast_factory is None:
            return super().forecast(t)
        view = self._forecast_factory(self.actual, t, self.cfg['horizon_seconds'])
        self.forecast_audit.append({'issued_s': t, 'observed_end_s': view.cutoff,
                                    'method': view.method, 'error_grid_hour': view.grid_hour})
        if self.cfg.get('export_forecasts', False):
            self.forecasts.extend(view.records())
        return view

    def _close_window(self):
        counts = Counter()
        nominal = actual_running = 0.
        for rid, rec in self.rec.items():
            stage = rec['status']
            if stage in {'completed', 'failed'}:
                continue
            counts[stage] += 1
            q = self.q[rid]
            if stage == 'running':
                elapsed = self.end - rec['service_start_s']
                nominal += max(0., q.service - elapsed)
                actual_running += max(0., rec['actual_service_s'] - elapsed)
            else:
                nominal += q.service
            rec.update(status='censored_at_H', stage_at_H=stage, terminal_s=self.end,
                       failure_reason='observation_window_censoring')
        censored = sum(counts.values())
        completed = sum(r['status'] == 'completed' for r in self.rec.values())
        if self.actual_starts != completed + counts['running']:
            raise AssertionError('A started service disappeared or repeated')
        self.window_summary = {
            'unfinished_at_H': censored, 'running_at_H': counts['running'],
            'queued_at_H': counts['queued'], 'waiting_at_H': censored - counts['running'],
            'unfinished_stages': dict(counts), 'remaining_work_nominal_s': nominal,
            'remaining_running_actual_s_evaluation_only': actual_running,
            'unstarted_remaining_actual_unknown_count': censored - counts['running'],
            'future_events_retained': len(self.events),
        }
        self.now = self.end
        self.strategy = self.public_policy
        return self

    def run(self):
        if self._observer_has_run:raise RuntimeError('Window observer is single-use')
        self._observer_has_run=True
        if self.strategy=='Oracle':self.prepare_oracle()
        for q in self.requests:self.push(q.arrival,'RequestArrival',q.request_id)
        for event in self.availability.events:
            if 0<event[0]<=self.end:self.push(event[0],'CapacityChange',event[1:])
        if self.strategy=='MPC':self.request_optimization(0.)
        while self.events:
            if self.events[0][0]>self.end:break
            t,_,_,kind,payload=heapq.heappop(self.events)
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
            elif kind=='Admission':
                r,m=payload
                self._admission_scheduled.discard((self.now,r,m))
                self.emit(kind,region=r,model=m,capacity=self.current_capacity[(r,m)])
                self._admit(r,m)
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
        return self._close_window()


class WindowedSpatial(WindowMixin, SPATIAL.SpatialSimulation):
    public_policy = 'Spatial_MILP'


class WindowedGreedy(WindowMixin, ENGINE.Simulation):
    public_policy = 'Greedy'


class WindowedFixed(WindowMixin, ENGINE.Simulation):

    public_policy = 'Fixed'
