import bisect
import time
import numpy as np
import scipy
import warnings
from scipy.optimize import milp,Bounds,LinearConstraint
from scipy.sparse import coo_matrix
from recovery import restore


class IntervalBook:
    def __init__(self,intervals):
        self.intervals=[]
        for left,right in sorted(intervals):
            if self.intervals and left<=self.intervals[-1][1]+1e-9:
                self.intervals[-1]=(self.intervals[-1][0],max(self.intervals[-1][1],right))
            else:self.intervals.append((left,right))
        self.ends=[right for _,right in self.intervals]


def gap_start(start,duration,intervals):
    book=intervals if isinstance(intervals,IntervalBook) else IntervalBook(intervals)
    first=bisect.bisect_right(book.ends,start+1e-9)
    for left,right in book.intervals[first:]:
        if start+duration<=left+1e-9:break
        if start<right-1e-9 and start+duration>left+1e-9:start=right
    return start


def solve(requests,now,servers,network,reservations,capacity_at,forecast,cfg,extra_candidates=()):
    if forecast.kind=='causal_forecast' and forecast.issued>now+1e-8:
        raise ValueError('Unpublished forecast cannot be used at this decision time')
    started=time.perf_counter();candidates=[];by_request=[]
    fixed_books={key:IntervalBook(value) for key,value in reservations.items()}
    delay=cfg['solve_delay_seconds'];guard=cfg['service_guard'];horizon=cfg['horizon_seconds']
    if cfg.get('force_timeout'):
        return [],{'status':1,'message':'forced_test_timeout','wall_seconds':0.,'mip_gap':None,'requests':len(requests),'variables':0,'constraints':0,'engine':'forced_test'}
    for i,q in enumerate(requests):
        columns=[];duration=q.service*guard
        for (r,m),pool in servers.items():
            if m!=q.model or not network[(q.origin,r)]['allowed']:continue
            link=network[(q.origin,r)];net=link['delay']
            release=max(q.arrival+cfg['router_seconds'],now+delay)+net
            latest=min(q.arrival+cfg['router_seconds']+net+q.wait,q.deadline-duration,now+horizon-duration)
            if latest<release-1e-8:continue
            for replica in range(pool['capacity']):
                key=(r,m,replica);fixed=fixed_books.get(key,IntervalBook([]))
                base=gap_start(release,duration,fixed)
                raw=[base+k*max(duration,1.) for k in range(cfg['candidate_steps'])]
                raw+=[latest]
                raw+=[end for _,end in fixed.intervals if release<=end<=latest][:cfg['candidate_steps']]
                first_hour=np.ceil((release-forecast.origin)/3600)*3600+forecast.origin
                raw+=list(np.arange(first_hour,latest+1e-8,3600))
                raw+=list(extra_candidates)
                seen=set()
                for value in raw:
                    u=round(gap_start(max(release,float(value)),duration,fixed),6)
                    if u in seen or not release-1e-6<=u<=latest+1e-6 or replica>=capacity_at(r,m,u):continue
                    seen.add(u)
                    dispatch=u-net
                    cost=pool['pue']*(pool['busy']-pool['idle'])/3.6e6*forecast.integral(u,u+q.service,r)
                    cost+=link['energy']*(forecast.at(dispatch,q.origin)+forecast.at(dispatch,r))/2
                    columns.append(len(candidates))
                    candidates.append({'i':i,'request_id':q.request_id,'region':r,'model':m,'replica':replica,
                        'start':u,'dispatch':dispatch,'end':u+duration,'cost_g':cost})
        by_request.append(columns)
    n=len(requests);K=len(candidates)
    log={'requests':n,'variables':K+n,'engine':'HiGHS via scipy '+scipy.__version__,'forecast_kind':forecast.kind,
        'candidate_restricted':True,'carbon_quantum_g':1e-6,'mip_gap':None,'status':None}
    if n==1:
        chosen=min(candidates,key=lambda x:(round(x['cost_g']*1e6),x['start'],x['region'])) if candidates else None
        log.update({'status':0 if chosen else 2,'message':'singleton exact candidate enumeration' if chosen else 'no feasible candidate',
            'engine':'singleton_enumeration','constraints':1,'wall_seconds':time.perf_counter()-started,'mip_gap':0. if chosen else None})
        return ([chosen] if chosen else []),log
    if not n:return [],{**log,'status':0,'message':'empty','constraints':0,'wall_seconds':0.}
    rows=[];cols=[];vals=[];lower=[1.]*n;upper=[1.]*n
    for i,indexes in enumerate(by_request):
        for col in indexes:rows.append(i);cols.append(col);vals.append(1.)
        rows.append(i);cols.append(K+i);vals.append(1.)
    pools={}
    for col,c in enumerate(candidates):pools.setdefault((c['region'],c['model'],c['replica']),[]).append((col,c))
    for key,items in pools.items():
        points=sorted({x for _,c in items for x in [c['start'],c['end']]})
        offset=len(lower)
        lower.extend([-np.inf]*(len(points)-1));upper.extend([1.]*(len(points)-1))
        for col,c in items:
            left=bisect.bisect_left(points,c['start']);right=bisect.bisect_left(points,c['end'])
            for interval in range(left,right):rows.append(offset+interval);cols.append(col);vals.append(1.)
    A=coo_matrix((np.array(vals), (np.array(rows,dtype=np.int32),np.array(cols,dtype=np.int32))),shape=(len(lower),K+n)).tocsc()
    costs=np.zeros(K+n)
    max_carbon=sum(max((round(candidates[k]['cost_g']*1e6) for k in indices),default=0) for indices in by_request)
    penalty=max_carbon+2
    for i,indices in enumerate(by_request):
        rank={k:j for j,k in enumerate(sorted(indices,key=lambda k:(candidates[k]['start'],candidates[k]['region'])))}
        for k in indices:costs[k]=round(candidates[k]['cost_g']*1e6)+rank[k]/max(1,n*(K+1))
    costs[K:]=penalty
    try:
        options={'time_limit':cfg['solver_time_limit'],'presolve':True}


        extra=cfg.get('highs_options',{})
        if set(extra)-{'time_limit','presolve','disp'}:
            raise ValueError('Native work/thread/seed options are unverified in this installed interface; use frozen ample-budget protocol')
        options.update(extra)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result=milp(costs,integrality=np.ones(K+n),bounds=Bounds(np.zeros(K+n),np.ones(K+n)),
                constraints=LinearConstraint(A,np.array(lower),np.array(upper)),options=options)
        log['option_warnings']=[str(w.message) for w in caught]
        log['solver_options']=options
        log['mip_node_count']=getattr(result,'mip_node_count',None)
        log['simplex_iterations']=getattr(result,'nit',None)
        log['iteration_visibility']='scipy.optimize.milp 1.9.1 exposes node count, not MIP simplex iterations'
        log['has_incumbent']=result.x is not None

        log.update({'status':int(result.status),'message':result.message,'constraints':len(lower),
            'wall_seconds':time.perf_counter()-started,'mip_gap':getattr(result,'mip_gap',None),
            'mip_dual_bound':getattr(result,'mip_dual_bound',None),'objective_solver':getattr(result,'fun',None)})
        if result.status!=0 or result.x is None:return [],log
        x,recovery=restore(result.x,A,lower,upper,0,1,np.ones(K+n,dtype=bool),costs,
            cfg['solver_time_limit']-(time.perf_counter()-started))
        log['recovery']=recovery
        if x is None:raise ValueError('Full binary solution rejected: '+recovery.get('rejection_reason','unknown'))
        log.update(full_solution_validated=True,wall_seconds=time.perf_counter()-started)
        plans=[candidates[k] for k in range(K) if x[k]>.5]
        log['unassigned']=int(x[K:].sum())
        return plans,log
    except Exception as exc:
        log.update({'status':4,'message':repr(exc),'constraints':len(lower),'wall_seconds':time.perf_counter()-started})
        return [],log
