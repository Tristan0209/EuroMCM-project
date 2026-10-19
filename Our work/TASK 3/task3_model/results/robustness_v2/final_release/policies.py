from context import *
import time,math
from queue_milp import spatial_solve
from v2.des import Simulation

def objective(plans,jobs):

    return (len(jobs)-len(plans),sum(round(p['cost_g']*1e6) for p in plans))

def nested_solve(jobs,now,servers,network,booked,observed,view,cfg,full):
    began=time.perf_counter();branches=[];all_plans=[]
    holds=cfg.get('holding_seconds',[0.,30.,120.,240.])
    assert holds[0]==0 and len(holds)==len(set(holds))
    for hold in holds:
        local={**cfg,'solver_time_limit':cfg['solver_time_limit']/len(holds)}
        if cfg.get('force_timeout',False):
            plans=[];log={'status':1,'wall_seconds':0.,'message':'injected timeout','candidates':[],'engine':'test'}
        else:
            plans,log=spatial_solve(jobs,now+hold,servers,network,booked,observed,view,local)


        log['holding_s']=hold;log['plans']=plans
        log['nominal_objective']=objective(plans,jobs)
        branches.append(log);all_plans.append(plans)
    allowed=range(len(holds)) if full else [0]
    valid=[j for j in allowed if branches[j]['status']==0]
    chosen=min(valid,key=lambda j:(*branches[j]['nominal_objective'],holds[j])) if valid else 0
    plans=all_plans[chosen] if valid else []
    log={'engine':'HiGHS nested queue union','status':0 if valid else branches[chosen]['status'],
        'wall_seconds':time.perf_counter()-began,'mip_gap':max((b.get('mip_gap') or 0. for b in branches),default=0.),
        'mip_node_count':sum((b.get('mip_node_count') or 0) for b in branches),
        'branches':branches,'selected_branch':chosen,'holding_s':holds[chosen],
        'full':full,'unassigned_count':len(jobs)-len(plans),'zero_branch_objective':branches[0]['nominal_objective'],
        'selected_objective':objective(plans,jobs),'planning_carbon_proxy_g':sum(p['cost_g'] for p in plans),
        'all_branch_optimal':all(b['status'] in [0,2] for b in branches),
        'subproblem_timeout_count':sum(b['status']==1 for b in branches),
        'forecast_kind':view.kind,'candidate_restricted':True,
        'nominal_time_semantics':'continuous FCFS queue feasibility; dispatch at commit plus uniform optional holding'}
    if full and branches[0]['status']==0 and valid:
        assert tuple(log['selected_objective'])<=tuple(log['zero_branch_objective'])
    return plans,log

def shift_and_validate(plans,jobs,network,cfg,extra):






    byid={q.request_id:q for q in jobs};valid=[];rejected=[]
    for raw in plans:
        p=dict(raw);q=byid[p['request_id']];net=network[(q.origin,p['region'])]['delay']
        for key in ['dispatch','start','end','predicted_queue_start']:p[key]+=extra
        predicted=p['predicted_queue_start']
        latest=min(q.deadline-cfg['service_guard']*q.service,q.arrival+cfg['router_seconds']+net+q.wait)
        if predicted>latest+1e-7:rejected.append(p['request_id'])
        else:valid.append(p)
    return valid,rejected

class NestedSimulation(Simulation):
    def __init__(self,*args,policy='E-Spatial',delay_scale=0.,**kwargs):
        super().__init__(*args,**kwargs);self.strategy='MPC';self.policy=policy;self.delay_scale=delay_scale
    def optimize(self):
        ids=sorted((rid for rid in self.pending if self.rec[rid]['status']=='pending'),key=lambda rid:(self.q[rid].deadline,self.q[rid].arrival,rid))[:self.cfg['max_batch']]
        if not ids:return
        jobs=[self.q[rid] for rid in ids];assert all(q.arrival<=self.now+1e-8 for q in jobs)
        began=time.perf_counter();view=self.forecast(self.now)
        plans,log=nested_solve(jobs,self.now,self.servers,self.network,self.reservations(),dict(self.current_capacity),view,self.cfg,self.policy=='E-Full')
        wall=time.perf_counter()-began
        extra=self.delay_scale*wall
        plans,rejected=shift_and_validate(plans,jobs,self.network,self.cfg,extra)
        duration=self.cfg['solve_delay_seconds']+extra
        log.update(decision_s=self.now,observed_carbon_end_s=view.cutoff,max_request_arrival_s=max(q.arrival for q in jobs),
            request_ids=ids,modelled_controller_s=duration,raw_solver_wall_seconds=log['wall_seconds'],wall_seconds=wall,
            extra_compute_s=extra,controller_queue_s=0.,post_compute_rejected_ids=rejected,
            issued_s=view.issued,policy=self.policy,delay_scale=self.delay_scale,committed_plans=plans)
        self.solver_log.append(log)
        self.controls.append((self.now,self.now+duration,self.cfg['controller_region'],self.policy))
        self.busy_until=self.now+duration;self.push(self.busy_until,'Commit',(ids,plans,log['status']))
    def run(self):
        super().run();self.strategy=self.policy;return self
