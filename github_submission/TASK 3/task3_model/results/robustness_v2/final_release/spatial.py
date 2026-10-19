from context import *
from v2.des import Simulation as FullSimulation
from queue_milp import spatial_solve
class SpatialSimulation(FullSimulation):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.strategy='MPC'

    def optimize(self):
        ids=sorted((rid for rid in self.pending if self.rec[rid]['status']=='pending'),key=lambda rid:(self.q[rid].deadline,self.q[rid].arrival,rid))[:self.cfg['max_batch']]
        if not ids:return
        jobs=[self.q[rid] for rid in ids];assert all(q.arrival<=self.now+1e-8 for q in jobs)
        view=self.forecast(self.now)
        plans,log=spatial_solve(jobs,self.now,self.servers,self.network,self.reservations(),dict(self.current_capacity),view,self.cfg)
        duration=self.cfg['solve_delay_seconds']+self.cfg['measured_wall_fraction']*log['wall_seconds']
        for p in plans:
            p['dispatch']=self.now+duration
            p['start']=p['dispatch']+self.network[(self.q[p['request_id']].origin,p['region'])]['delay']
        log.update(decision_s=self.now,observed_carbon_end_s=view.cutoff,max_request_arrival_s=max(q.arrival for q in jobs),request_ids=ids,modelled_controller_s=duration)
        self.solver_log.append(log);self.controls.append((self.now,self.now+duration,self.cfg['controller_region'],'Spatial_MILP'))
        self.busy_until=self.now+duration;self.push(self.busy_until,'Commit',(ids,plans,log['status']))

    def run(self):
        super().run();self.strategy='Spatial_MILP';return self
