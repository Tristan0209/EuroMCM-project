from context import *
import numpy as np
import recovery
from recovery import restore,violations,TOL

def counterexample():
    return np.load(PREVIOUS/'audit/numerical_diagnostic/rejected_3621.npz')

def test_saved_real_counterexample_is_unrecoverable_and_rejected():
    f=counterexample();raw=f['x'];integer=f['integer'].astype(bool)
    args=[f[k] for k in ['A','lower','upper','bounds_lower','bounds_upper']]
    naive=raw.copy();naive[integer]=np.rint(naive[integer])
    before=violations(naive,*args,integer)
    assert before['max_violation']>1e-3
    x,info=restore(raw,*args,integer,f['objective'],10.)


    assert x is None and info['illegal_rounded_candidate'] and info['unrecoverable']
    assert info['lp_status']==2
    save(ROOT/'audit/saved_counterexample.json',{'naive':before,'recovery':info})

def simple(raw,budget=10.):

    return restore(raw,np.array([[-10000.,1.]]),[-np.inf],[0.],[0.,0.],[1.,10.],
        [True,False],[1.,1e-6],budget)

def test_valid_unchanged():
    raw=np.array([1.,3.]);x,info=simple(raw)
    assert np.array_equal(raw,x) and not info['attempted']

def test_switch_repair():
    x,info=simple([1e-6,.005]);assert info['recovered'] and np.array_equal(x,[0.,0.])

def test_budget_exhaustion_rejects():
    x,info=simple([1e-6,.005],0);assert x is None and info['unrecoverable']

def test_fractionality_rejects_without_lp():
    x,info=simple([.1,.005]);assert x is None and not info['attempted']

def test_nonfinite_rejected():
    x,info=simple([np.nan,.005]);assert x is None and info['unrecoverable']

def test_fixed_integer_infeasible_never_changes_assignment():
    x,info=restore([1e-6,.005],[[1.,0.]],[1.],[1.],[0.,0.],[1.,10.],[True,False],[1.,0.],10.)
    assert x is None and info['unrecoverable'] and info['lp_status']==2

def test_pure_binary_no_continuous_repair():
    x,info=restore([0.],[[1.]],[1.],[1.],[0.],[1.],[True],[1.],10.)
    assert x is None and not info['attempted']

def test_lp_timeout_rejected(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(recovery,'linprog',lambda *a,**k:SimpleNamespace(status=1,x=None,message='timeout',nit=0))
    x,info=simple([1e-6,.005]);assert x is None and info['lp_status']==1

def test_invalid_lp_solution_rejected(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(recovery,'linprog',lambda *a,**k:SimpleNamespace(status=0,x=np.array([11.]),message='bad fake',nit=0))
    x,info=simple([1e-6,.005]);assert x is None and info['final_validation']['bound_upper']>0

def test_lower_bound_and_equality_recovered():
    x,info=restore([1e-6,.005],[[10000.,-1.],[0.,1.]],[-np.inf,0.],[0.,0.],
        [0.,0.],[1.,10.],[True,False],[1.,1e-6],10.)
    assert info['recovered'] and np.array_equal(x,[0.,0.])

def test_a2_des_byte_identical():
    assert sha(ROOT/'v2/des.py')==sha(A2/'v2/des.py')

def test_original_tolerance_preserved():assert TOL==1e-5
