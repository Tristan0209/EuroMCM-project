from context import *
import time
import numpy as np
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from v2.des import Simulation as FullSimulation
from recovery import restore,TOL


def spatial_solve(jobs,now,servers,network,booked,observed,view,cfg):
    started=time.perf_counter();guard=cfg['service_guard'];N=len(jobs);choices=[];byjob=[]
    for i,q in enumerate(jobs):
        indexes=[]
        for (r,m),pool in servers.items():
            if m!=q.model or not network[(q.origin,r)]['allowed']:continue
            net=network[(q.origin,r)]
            ready=now+cfg['solve_delay_seconds']+net['delay']
            latest=min(q.deadline-guard*q.service,q.arrival+cfg['router_seconds']+net['delay']+q.wait)
            for j in range(observed[(r,m)]):
                tail=max([ready]+[end for _,end in booked.get((r,m,j),[])])
                if tail>latest+1e-8:continue
                cost=pool['pue']*(pool['busy']-pool['idle'])/3.6e6*q.service*view.at(now,r)
                cost+=net['energy']*(view.at(now,q.origin)+view.at(now,r))/2
                indexes.append(len(choices))
                choices.append({'i':i,'request_id':q.request_id,'region':r,'model':m,'replica':j,'ready':ready,'earliest':tail,
                    'latest':latest,'duration':guard*q.service,'cost_g':cost,'queue_order':(net['delay'],q.arrival,q.request_id)})
        byjob.append(indexes)
    K=len(choices)
    log={'requests':N,'variables':2*K+N,'engine':'HiGHS spatial queue assignment','status':0,'mip_gap':None,
        'candidate_restricted':True,'forecast_kind':view.kind,'spatial_no_policy_holding':True,'candidates':choices,'request_ids':[q.request_id for q in jobs]}
    def plan(c,predicted):
        return {**{k:c[k] for k in ['i','request_id','region','model','replica','cost_g']},
            'dispatch':now+cfg['solve_delay_seconds'],'start':c['ready'],'end':c['ready']+c['duration'],'predicted_queue_start':predicted}
    if N==1:
        choice=min(choices,key=lambda c:(round(c['cost_g']*1e6),c['earliest'],c['region'],c['replica'])) if choices else None
        log.update(status=0 if choice else 2,engine='singleton_enumeration_spatial',constraints=1,mip_gap=0. if choice else None,wall_seconds=time.perf_counter()-started)
        return ([plan(choice,choice['earliest'])] if choice else []),log
    rr=[];cc=[];vv=[];lower=[];upper=[]
    def add(items,lb=-np.inf,ub=np.inf):
        idx=len(lower);lower.append(lb);upper.append(ub)
        for col,value in items:rr.append(idx);cc.append(col);vv.append(value)
    for i,indexes in enumerate(byjob):add([(x,1.) for x in indexes]+[(K+i,1.)],1.,1.)
    end=cfg.get('observation_horizon_seconds',max(q.deadline for q in jobs));M=end+max(q.service*guard for q in jobs)+1
    for k,c in enumerate(choices):
        u=K+N+k
        add([(u,1.),(k,-c['latest'])],ub=0.)
        add([(u,-1.),(k,c['earliest'])],ub=0.)
    pools={}
    for k,c in enumerate(choices):pools.setdefault((c['region'],c['model'],c['replica']),[]).append(k)
    for items in pools.values():
        items.sort(key=lambda k:choices[k]['queue_order'])
        for a_pos,a in enumerate(items):
            for b in items[a_pos+1:]:
                add([(K+N+a,1.),(K+N+b,-1.),(a,M),(b,M)],ub=2*M-choices[a]['duration'])
    matrix=coo_matrix((vv,(rr,cc)),shape=(len(lower),2*K+N)).tocsc()
    costs=np.zeros(2*K+N)
    penalty=sum(max((round(choices[k]['cost_g']*1e6) for k in ids),default=0) for ids in byjob)+2
    for k,c in enumerate(choices):costs[k]=round(c['cost_g']*1e6);costs[K+N+k]=1/max(1,(K+1)*(end+1))
    costs[K:K+N]=penalty
    bounds_upper=np.r_[np.ones(K+N),np.full(K,end)]
    try:
        result=milp(costs,integrality=np.r_[np.ones(K+N),np.zeros(K)],bounds=Bounds(0,bounds_upper),
            constraints=LinearConstraint(matrix,np.array(lower),np.array(upper)),options={'time_limit':cfg['solver_time_limit'],'presolve':True})
        log.update(status=int(result.status),message=result.message,mip_gap=getattr(result,'mip_gap',None),mip_node_count=getattr(result,'mip_node_count',None),
            constraints=len(lower),wall_seconds=time.perf_counter()-started)
        if result.status!=0 or result.x is None:return [],log
        integer=np.r_[np.ones(K+N,dtype=bool),np.zeros(K,dtype=bool)]
        x,recovery=restore(result.x,matrix,lower,upper,0,bounds_upper,integer,costs,
            cfg['solver_time_limit']-(time.perf_counter()-started))
        log['recovery']=recovery
        if recovery['illegal_rounded_candidate']:
            import hashlib
            evidence=ROOT/'audit/recovery_candidates';evidence.mkdir(parents=True,exist_ok=True)
            key=hashlib.sha256(np.asarray(result.x).tobytes()+matrix.data.tobytes()).hexdigest()[:24]
            path=evidence/(key+'.npz')
            if not path.exists():
                np.savez_compressed(path,raw=result.x,recovered=np.array([]) if x is None else x,A=matrix.toarray(),
                    lower=lower,upper=upper,bounds_lower=np.zeros(len(costs)),bounds_upper=bounds_upper,
                    integer=integer,objective=costs)
            log['recovery_evidence']=str(path)
        if x is None:
            log.update(status=4,message='Spatial solver constraint violation; full solution recovery rejected: '+recovery.get('rejection_reason','unknown'),wall_seconds=time.perf_counter()-started)
            return [],log

        selected=[k for k in range(K) if x[k]>.5]
        for k in selected:
            c=choices[k];u=x[K+N+k];q=jobs[c['i']]
            assert c['earliest']-TOL<=u<=c['latest']+TOL
            assert u+c['duration']<=q.deadline+TOL and u>=q.arrival-TOL
            assert c['replica']<observed[(c['region'],c['model'])] and c['model']==q.model
            assert network[(q.origin,c['region'])]['allowed']
        for items in pools.values():
            active=[k for k in items if x[k]>.5]
            for a,b in zip(active,active[1:]):
                assert x[K+N+a]+choices[a]['duration']<=x[K+N+b]+TOL
        log.update(wall_seconds=time.perf_counter()-started,full_solution_validated=True)
        return [plan(choices[k],x[K+N+k]) for k in selected],log
    except Exception as exc:
        log.update(status=4,message=repr(exc),wall_seconds=time.perf_counter()-started,constraints=len(lower))
        return [],log
