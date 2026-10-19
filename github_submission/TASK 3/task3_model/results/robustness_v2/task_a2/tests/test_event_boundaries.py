from dataclasses import replace
import copy
import pytest
import numpy as np
from test_system import fixtures,Availability
from src.domain import Request
from v2.des import Simulation
from src.accounting import evaluate
from src.experiments import audit_run


def isolated():
    servers,net,carbon,cfg,q=fixtures()


    for (a,b),v in net.items():v['allowed']=b==a
    cfg.update(observation_horizon_seconds=200.,oracle_batch=64)
    return servers,net,carbon,cfg,q


def assert_physics(sim):
    v,rec,_=evaluate(sim,200)
    audit_run(sim,rec)
    assert v['completed']+v['failed']==len(sim.requests)
    starts=[x for x in sim.log if x['event']=='ServiceStart']
    assert len(starts)==len({x['request_id'] for x in starts})
    assert all(x['replica']<sim.availability.at(x['region'],x['model'],x['time_s']) for x in starts)
    return v,rec


@pytest.mark.parametrize('strategy',['Fixed','Greedy','MPC','Oracle'])
def test_audit_10_30_regression(strategy):
    s,n,c,cfg,q=isolated()


    jobs=[replace(q,deadline=10),replace(q,request_id='two',arrival=1,service=5)]
    sim=Simulation(jobs,s,n,Availability([(10.,'DE','L',0),(30.,'DE','L',1)]),c,{'one':1.,'two':1.},strategy,cfg).run()
    assert sim.rec['one']['completion_s']==10
    assert sim.rec['two']['service_start_s']==30
    assert sim.rec['two']['completion_s']==35
    assert_physics(sim)


@pytest.mark.parametrize('ready_at',[9.999,10.,10.001])
def test_network_ready_at_or_around_drop(ready_at):
    s,n,c,cfg,q=isolated()
    n[('DE','DE')]['delay']=ready_at
    sim=Simulation([q],s,n,Availability([(10.,'DE','L',0),(30.,'DE','L',1)]),c,{'one':1.},'Fixed',cfg).run()
    expected=ready_at if ready_at<10 else 30
    assert sim.rec['one']['service_start_s']==pytest.approx(expected)
    assert_physics(sim)


@pytest.mark.parametrize('strategy',['Fixed','Greedy','MPC','Oracle'])
def test_completion_restore_arrival_and_zero_network_same_timestamp(strategy):
    s,n,c,cfg,q=isolated()
    jobs=[replace(q,service=20),replace(q,request_id='two',arrival=20,service=5)]
    sim=Simulation(jobs,s,n,Availability([(5.,'DE','L',0),(20.,'DE','L',1)]),c,{'one':1.,'two':1.},strategy,cfg).run()
    assert sim.rec['one']['completion_s']==20
    assert sim.rec['two']['service_start_s']==20
    assert_physics(sim)


def test_partial_drop_grandfathers_disabled_replica_without_duplication():
    s,n,c,cfg,q=isolated();s[('DE','L')]['capacity']=2
    jobs=[q,replace(q,request_id='two',service=20),replace(q,request_id='three',arrival=10,service=5)]
    av=Availability([(0.,'DE','L',2),(10.,'DE','L',1)])
    sim=Simulation(jobs,s,n,av,c,dict.fromkeys(['one','two','three'],1.),'Fixed',cfg).run()
    assert sim.rec['two']['completion_s']==20
    assert sim.rec['three']['service_start_s']==10
    assert sim.rec['three']['replica']==0
    assert_physics(sim)


@pytest.mark.parametrize('reverse',[False,True])
def test_multiple_pool_updates_are_all_visible_before_admission(reverse):
    s,n,c,cfg,q=isolated()
    jobs=[q,replace(q,request_id='two',arrival=1,service=5),
        replace(q,request_id='fr1',origin='FR'),replace(q,request_id='fr2',origin='FR',arrival=1,service=5)]
    events=[(10.,'DE','L',0),(10.,'FR','L',0),(30.,'DE','L',1),(40.,'FR','L',1)]
    if reverse:events.reverse()
    sim=Simulation(jobs,s,n,Availability(events),c,{q.request_id:1. for q in jobs},'Fixed',cfg).run()
    assert sim.rec['two']['service_start_s']==30
    assert sim.rec['fr2']['service_start_s']==40
    at10=[e for e in sim.log if e['time_s']==10]
    changes=[i for i,e in enumerate(at10) if e['event']=='CapacityChange']
    starts=[i for i,e in enumerate(at10) if e['event']=='ServiceStart']
    assert not starts or max(changes)<min(starts)
    assert_physics(sim)


def test_zero_capacity_until_deadline_retains_failure_and_idle_energy():
    s,n,c,cfg,q=isolated()
    sim=Simulation([q],s,n,Availability([(0.,'DE','L',0)]),c,{'one':1.},'Fixed',cfg).run()
    v,rec=assert_physics(sim)
    assert v['failed']==1 and v['active_increment_kwh']==0
    assert v['idle_kwh']==pytest.approx(6*175*1.2*200/3.6e6)


def test_conflicting_duplicate_capacity_events_rejected_not_order_dependent():
    s,n,c,cfg,q=isolated()
    with pytest.raises(ValueError,match='duplicate'):
        Simulation([q],s,n,Availability([(10.,'DE','L',0),(10.,'DE','L',1)]),c,{'one':1.},'Fixed',cfg)


def test_zero_duration_same_timestamp_completions_do_not_duplicate():
    s,n,c,cfg,q=isolated()
    jobs=[replace(q,request_id=str(i),service=0.) for i in range(5)]
    sim=Simulation(jobs,s,n,Availability(),c,{q.request_id:1. for q in jobs},'Fixed',cfg).run()
    v,rec=assert_physics(sim)
    assert v['completed']==5 and rec.service_start_s.eq(0.).all()


def test_v2_solver_node_count_and_option_log():
    from v2.milp import solve
    from src.carbon import OracleView
    s,n,c,cfg,q=isolated()
    cfg['highs_options']={}
    plans,log=solve([q,replace(q,request_id='two')],0,s,n,{},lambda r,m,t:1,OracleView(c),cfg)
    assert log['status']==0 and log['mip_node_count'] is not None
    assert log['solver_options']=={'time_limit':cfg['solver_time_limit'],'presolve':True}
    assert len(plans)==2


def test_exact_original_fixed_10_30_fixture():
    s,n,c,cfg,q=isolated()
    jobs=[q,replace(q,request_id='two',arrival=1,service=5)]
    sim=Simulation(jobs,s,n,Availability([(10.,'DE','L',0),(30.,'DE','L',1)]),c,{'one':1.,'two':1.},'Fixed',cfg).run()
    assert sim.rec['one']['completion_s']==10
    assert sim.rec['two']['service_start_s']==30
    assert_physics(sim)


def test_unsupported_native_options_are_explicitly_rejected_for_safe_fallback():
    from v2.milp import solve
    from src.carbon import OracleView
    s,n,c,cfg,q=isolated()
    cfg['highs_options']={'max_threads':1}
    plans,log=solve([q,replace(q,request_id='two')],0,s,n,{},lambda r,m,t:1,OracleView(c),cfg)
    assert log['status']==4 and not plans and 'unverified' in log['message']
