import time
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csc_matrix
TOL=1e-5

def violations(x,A,lower,upper,lb,ub,integer):
    x=np.asarray(x,float)
    if not np.isfinite(x).all():return {'max_violation':float('inf'),'finite':False}
    activity=np.asarray(A@x).ravel()
    def maximum(v):return max(0.,float(np.max(v))) if len(v) else 0.
    values={'row_lower':maximum(np.asarray(lower)-activity),'row_upper':maximum(activity-np.asarray(upper)),
        'bound_lower':maximum(np.asarray(lb)-x),'bound_upper':maximum(x-np.asarray(ub)),
        'integrality':maximum(np.abs(x[integer]-np.rint(x[integer]))),'finite':True}
    values['max_violation']=max(v for k,v in values.items() if k!='finite');return values

def restore(raw,A,lower,upper,lb,ub,integer,costs,budget_seconds):
    start=time.perf_counter();raw=np.asarray(raw,float);A=csc_matrix(A)
    lower=np.asarray(lower,float);upper=np.asarray(upper,float)
    lb=np.broadcast_to(np.asarray(lb,float),raw.shape).copy();ub=np.broadcast_to(np.asarray(ub,float),raw.shape).copy()
    integer=np.asarray(integer,bool);continuous=~integer
    info={'illegal_rounded_candidate':False,'attempted':False,'recovered':False,'unrecoverable':False,
        'feasibility_tolerance':TOL,'lp_status':None,'lp_iterations':None,'lp_wall_seconds':0.}
    if not np.isfinite(raw).all():
        info.update(unrecoverable=True,rejection_reason='nonfinite incumbent');return None,info
    info['raw_integrality_distance']=float(np.max(abs(raw[integer]-np.rint(raw[integer])))) if integer.any() else 0.
    x=raw.copy();x[integer]=np.rint(x[integer])
    info['rounded_validation']=violations(x,A,lower,upper,lb,ub,integer)
    if info['raw_integrality_distance']>TOL:
        info.update(illegal_rounded_candidate=True,unrecoverable=True,rejection_reason='integer distance exceeds unchanged threshold');return None,info
    if info['rounded_validation']['max_violation']<=TOL:
        info['final_validation']=info['rounded_validation'];return x,info
    info['illegal_rounded_candidate']=True
    if not continuous.any() or budget_seconds<=0:
        info.update(unrecoverable=True,rejection_reason='no continuous freedom or remaining work budget');return None,info
    info['attempted']=True;lp_start=time.perf_counter()
    fixed=x[integer].copy();shift=np.asarray(A[:,integer]@fixed).ravel();Ac=A[:,continuous]
    equality=np.isfinite(lower)&np.isfinite(upper)&(lower==upper)
    high=np.isfinite(upper)&~equality;low=np.isfinite(lower)&~equality
    from scipy.sparse import vstack
    Au=vstack([Ac[high],-Ac[low]],format='csc');bu=np.r_[upper[high]-shift[high],shift[low]-lower[low]]
    ce=np.asarray(costs,float)[continuous];scale=float(np.max(abs(ce))) if len(ce) else 0.
    if scale:ce=ce/scale
    info['lp_objective_positive_rescale']=scale
    try:
        result=linprog(ce,A_ub=Au if len(bu) else None,b_ub=bu if len(bu) else None,
            A_eq=Ac[equality] if equality.any() else None,b_eq=lower[equality]-shift[equality] if equality.any() else None,
            bounds=list(zip(lb[continuous],ub[continuous])),method='highs',
            options={'time_limit':float(budget_seconds),'presolve':True,
                'primal_feasibility_tolerance':1e-7,'dual_feasibility_tolerance':1e-7})
        info.update(lp_status=int(result.status),lp_message=result.message,lp_iterations=getattr(result,'nit',None))
        if result.status!=0 or result.x is None:
            info.update(unrecoverable=True,rejection_reason='fixed-integer continuous LP did not succeed');return None,info
        x[continuous]=result.x
        assert np.array_equal(x[integer],fixed),'Recovery changed an integer assignment'
        info['final_validation']=violations(x,A,lower,upper,lb,ub,integer)
        if info['final_validation']['max_violation']>TOL:
            info.update(unrecoverable=True,rejection_reason='full recovered solution still violates original model');return None,info
        info.update(recovered=True,original_objective=float(np.dot(costs,raw)),recovered_objective=float(np.dot(costs,x)),
            maximum_continuous_change=float(np.max(abs(x[continuous]-raw[continuous]))))
        return x,info
    except Exception as exc:
        info.update(unrecoverable=True,rejection_reason=repr(exc));return None,info
    finally:
        info['lp_wall_seconds']=time.perf_counter()-lp_start
        info['total_recovery_wall_seconds']=time.perf_counter()-start
