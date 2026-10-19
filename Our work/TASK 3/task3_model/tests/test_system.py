import copy
from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from src.domain import Request,load_inputs,shocks_for
from src.carbon import Carbon,ForecastView,OracleView
from src.milp import solve,gap_start,IntervalBook
from src.des import Simulation,DEFAULTS
from src.accounting import evaluate

ANCHOR=pd.Timestamp('2025-01-02T00:00:00Z')


class Availability:
    def __init__(self,events=()):self.events=list(events)
    def at(self,r,m,t):
        c=1
        for when,rr,mm,value in sorted(self.events):
            if rr==r and mm==m and when<=t:c=value
        return c


def fixtures():
    servers={(r,m):{'capacity':1,'idle':175.,'busy':250.,'pue':1.2} for r in ['DE','FR','ES'] for m in ['S','L']}
    network={(r,s):{'delay':0.,'energy':0.,'allowed':True} for r in ['DE','FR','ES'] for s in ['DE','FR','ES']}
    frame=pd.DataFrame([{'datetime_utc':t,'region':r,'carbon_intensity_gco2e_kwh':{'DE':300,'FR':30,'ES':200}[r]}
        for t in pd.date_range('2025-01-01T00:00:00Z',periods=72,freq='h') for r in ['DE','FR','ES']])
    carbon=Carbon(frame,ANCHOR)
    cfg={**DEFAULTS,'router_seconds':0.,'solve_delay_seconds':0.,'service_guard':1.,'arrival_batch_seconds':0.,
        'horizon_seconds':7200,'solver_time_limit':2.,'record_events':True}
    q=Request('one','L','DE','Math','batch',0,100,90,10,1)
    return servers,network,carbon,cfg,q


def test_frozen_router_and_common_conditions():
    requests,servers,network,anchor=load_inputs()
    assert len(requests)==7335
    assert all(q.model=='L' for q in requests if q.task_type in ['Code','Math'])
    assert sum(q.model=='S' for q in requests)==431
    for rep in [0,1,2]:
        a=shocks_for(rep,[q.request_id for q in requests[:3]])
        b=shocks_for(rep,[q.request_id for q in reversed(requests[:3])])
        assert a==b


def test_hand_calculated_MILP_optimum_and_DES_energy():
    servers,net,carbon,cfg,q=fixtures();jobs=[q,replace(q,request_id='two')]
    plans,log=solve(jobs,0,servers,net,{},lambda r,m,t:1,OracleView(carbon),cfg)
    assert log['status']==0
    assert all(p['region']=='FR' for p in plans)
    assert sum(p['cost_g'] for p in plans)==pytest.approx(20*90/3.6e6*30)
    a,b=sorted(plans,key=lambda p:p['start']);assert b['start']>=a['end']-1e-7
    sim=Simulation(jobs,servers,net,Availability(),carbon,{'one':1,'two':1},'Oracle',cfg).run()
    summary,records,ledger=evaluate(sim,200,{'one':.8,'two':.8})
    expected_idle=6*175*1.2*200/3.6e6
    assert summary['idle_kwh']==pytest.approx(expected_idle)
    assert summary['active_increment_kwh']==pytest.approx(20*90/3.6e6)
    assert summary['active_increment_co2e_g']==pytest.approx(.015)
    assert records.status.eq('completed').all()


def test_cross_hour_carbon_integration():
    servers,net,carbon,cfg,q=fixtures()
    i=carbon.index(0);carbon.values[i,0]=100;carbon.values[i+1,0]=200
    assert carbon.integral(3590,3610,'DE')==pytest.approx(3000)
    q=replace(q,arrival=3590,deadline=3700,service=20)
    sim=Simulation([q],servers,net,Availability(),carbon,{'one':1},'Fixed',cfg).run()
    summary,_,_=evaluate(sim,3800)
    assert summary['active_increment_co2e_g']==pytest.approx(90/3.6e6*3000)


def test_no_future_actual_or_future_shock_in_MPC():
    servers,net,actual,cfg,q=fixtures()
    forecast=ForecastView(actual,0,7200)
    before=forecast.at(3600,'DE')
    changed=copy.deepcopy(actual);changed.values[changed.index(0):]=10000
    assert ForecastView(changed,0,7200).at(3600,'DE')==before
    with pytest.raises(ValueError):forecast.at(-1,'DE')
    jobs=[q,replace(q,request_id='two',arrival=5000,deadline=5100)]
    sim=Simulation(jobs,servers,net,Availability(),actual,{'one':1,'two':1},'MPC',cfg).run()
    assert all(log['max_request_arrival_s']<=log['decision_s'] for log in sim.solver_log)
    assert all(log['observed_carbon_end_s']<=log['decision_s'] for log in sim.solver_log)
    altered=Simulation([q],servers,net,Availability(),actual,{'one':2},'MPC',cfg).run()
    assert sim.rec['one']['dispatch_s']==altered.rec['one']['dispatch_s']
    assert sim.rec['one']['region']==altered.rec['one']['region']


def test_capacity_loss_grandfathers_running_and_blocks_new_starts():
    servers,net,carbon,cfg,q=fixtures()
    jobs=[replace(q,service=20),replace(q,request_id='two',arrival=6,deadline=100)]
    availability=Availability([(5,'DE','L',0),(30,'DE','L',1)])
    sim=Simulation(jobs,servers,net,availability,carbon,{'one':1,'two':1},'Fixed',cfg).run()
    assert sim.rec['one']['completion_s']==20
    assert sim.rec['two']['service_start_s']==30
    assert sum(e['event']=='ServiceStart' for e in sim.log)==2


def test_region_prohibition_and_model_compatibility():
    servers,net,carbon,cfg,q=fixtures()
    net[('DE','FR')]['allowed']=False
    sim=Simulation([q],servers,net,Availability(),carbon,{'one':1},'MPC',cfg).run()
    assert sim.rec['one']['region']!='FR'
    assert sim.rec['one']['selected_model']=='L'


def test_timeout_fallback_records_all_requests():
    servers,net,carbon,cfg,q=fixtures();cfg['force_timeout']=True
    sim=Simulation([q],servers,net,Availability(),carbon,{'one':1},'MPC',cfg).run()
    assert sim.fallbacks==1 and sim.rec['one']['status']=='completed'
    assert sim.solver_log[0]['status']==1


def test_infeasible_or_unavailable_request_never_disappears():
    servers,net,carbon,cfg,q=fixtures();q=replace(q,deadline=1,wait=0)
    sim=Simulation([q],servers,net,Availability(),carbon,{'one':1},'MPC',cfg).run()
    summary,records,_=evaluate(sim,100)
    assert len(records)==1 and summary['failed']==1
    assert summary['deadline_violation_rate']==1
    assert sim.solver_log[0]['status']==2 and sim.fallbacks==1


def test_service_multiplier_network_and_time_identity():
    servers,net,carbon,cfg,q=fixtures();cfg['router_seconds']=.01
    net[('DE','DE')]['delay']=.07;net[('DE','DE')]['energy']=.000001
    sim=Simulation([q],servers,net,Availability(),carbon,{'one':1.1},'Fixed',cfg).run()
    rec=sim.rec['one'];assert rec['actual_service_s']==pytest.approx(11)
    assert rec['response_s']==pytest.approx(.01+rec['intentional_delay_s']+.07+rec['queue_wait_s']+11)
    metrics,_,_=evaluate(sim,100)
    assert metrics['network_kwh']==pytest.approx(.000001)
    assert metrics['controller_kwh']==pytest.approx(.01*12/3.6e6)


def test_missing_or_nonfinite_carbon_fails_explicitly():
    with pytest.raises(FileNotFoundError):Carbon.load(ANCHOR,'missing_fixture')
    servers,net,carbon,cfg,q=fixtures()
    with pytest.raises(ValueError):carbon.at(10**10,'DE')


def test_unpublished_forecast_is_rejected():
    servers,net,carbon,cfg,q=fixtures()
    view=ForecastView(carbon,10,7200)
    with pytest.raises(ValueError,match='Unpublished'):
        solve([q],0,servers,net,{},lambda r,m,t:1,view,cfg)


def test_estimation_overrun_is_queued_not_double_allocated():
    servers,net,carbon,cfg,q=fixtures()
    jobs=[q,replace(q,request_id='two',arrival=1)]
    sim=Simulation(jobs,servers,net,Availability(),carbon,{'one':2,'two':1},'Fixed',cfg).run()
    assert sim.rec['one']['completion_s']==20
    assert sim.rec['two']['service_start_s']==20
    assert all(e.get('active',0)<=1 for e in sim.log if e['event']=='ServiceStart')


def test_reservation_index_and_history_pruning_preserve_feasible_gaps():
    intervals=[(i*4.,i*4.+2) for i in range(1000)]+[(4100,4110),(4105,4120)]
    book=IntervalBook(intervals)
    for start in [0,2,100,3990,4099,4100,4115,4120]:
        expected=start
        for left,right in sorted(intervals):
            if expected+3<=left+1e-9:break
            if expected<right-1e-9 and expected+3>left+1e-9:expected=right
        assert gap_start(start,3,book)==expected
        assert gap_start(start,3,[v for v in intervals if v[1]>start-1e-8])==expected


def test_input_fingerprint_invalidates_after_data_change():
    import tempfile
    from pathlib import Path
    from src.util import ROOT,signature_for_files
    with tempfile.TemporaryDirectory(prefix='synthetic_test_only_',dir=ROOT/'results') as temp:
        path=Path(temp)/'input.csv';path.write_text('capacity\n1\n',encoding='utf-8')
        first=signature_for_files([path])
        path.write_text('capacity\n2\n',encoding='utf-8')
        second=signature_for_files([path])
        assert first['sha256']!=second['sha256']


@pytest.mark.parametrize('strategy',['Fixed','Greedy','Oracle','MPC'])
def test_unique_nonpreemptive_assignment_and_physical_capacity(strategy):
    servers,net,carbon,cfg,q=fixtures()
    jobs=[replace(q,request_id=str(i),arrival=i*.1,deadline=200) for i in range(12)]
    sim=Simulation(jobs,servers,net,Availability(),carbon,{str(i):1 for i in range(12)},strategy,cfg).run()
    assert len(sim.rec)==12
    starts=[e for e in sim.log if e['event']=='ServiceStart']
    assert len({e['request_id'] for e in starts})==len(starts)
    assert all(e['active']<=1 and e['admission_capacity']>=1 for e in starts)
    for rec in sim.rec.values():
        assert rec['status'] in ['completed','failed']
        if rec['status']=='completed':assert rec['service_start_s']>=rec['arrival_s']
