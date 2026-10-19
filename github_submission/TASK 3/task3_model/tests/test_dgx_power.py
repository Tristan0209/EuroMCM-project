from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from src.power import idle_base_w,incremental_w,profile,upstream_paths
from src.domain import load_inputs
from src.util import DATA,ROOT,PROJECT,read,input_signature
from src.milp import solve
from src.carbon import OracleView
from src.des import Simulation
from src.accounting import evaluate
from test_system import fixtures,Availability


def dgx_servers():
    return {(r,m):{'capacity':1,'idle':2900.,'busy':3485.,'pue':1.2,'idle_base_w':2900.}
            for r in ['DE','FR','ES'] for m in ['S','L']}


def test_current_DGX_topology_preserves_six_slots_and_request_fields():
    requests,servers,network,anchor=load_inputs()
    assert len(servers)==6 and sum(s['capacity'] for s in servers.values())==6
    assert sum(s['installed_gpus'] for s in servers.values())==48
    assert sum(idle_base_w(s)*s['pue']*43200/3.6e6 for s in servers.values())==pytest.approx(250.56)
    assert all(incremental_w(s)==585 for s in servers.values())
    old=pd.read_csv(ROOT/read(ROOT/'configs/upstream.json')['legacy_archive']/'results/data/requests.csv')
    new=pd.read_csv(DATA/'requests.csv')
    pd.testing.assert_frame_equal(old.drop(columns='estimated_energy_kwh'),new.drop(columns='estimated_energy_kwh'),check_exact=False,rtol=1e-12)
    np.testing.assert_allclose(new.estimated_energy_kwh,old.estimated_energy_kwh*7.8)


def test_DGX_hand_optimum_and_energy_are_not_legacy_75W():
    _,net,carbon,cfg,q=fixtures();servers=dgx_servers()
    jobs=[q,replace(q,request_id='two')]
    plans,log=solve(jobs,0,servers,net,{},lambda r,m,t:1,OracleView(carbon),cfg)
    assert log['status']==0 and all(p['region']=='FR' for p in plans)
    assert sum(p['cost_g'] for p in plans)==pytest.approx(.117)
    sim=Simulation(jobs,servers,net,Availability(),carbon,{'one':1,'two':1},'Oracle',cfg).run()
    result,records,ledger=evaluate(sim,200)
    assert result['active_increment_co2e_g']==pytest.approx(.117)
    assert result['active_increment_kwh']==pytest.approx(.0039)
    assert result['idle_kwh']==pytest.approx(1.16)
    assert records.status.eq('completed').all()


def test_extra_admitted_slot_does_not_duplicate_host_idle_power():
    _,net,carbon,cfg,q=fixtures();servers=dgx_servers()

    servers[('DE','L')]['capacity']=2
    jobs=[q,replace(q,request_id='two')]
    sim=Simulation(jobs,servers,net,Availability([(0,'DE','L',2)]),carbon,{'one':1,'two':1},'Fixed',cfg).run()
    result,records,_=evaluate(sim,200)
    assert result['idle_kwh']==pytest.approx(1.16)
    assert result['active_increment_kwh']==pytest.approx(.0039)
    assert records.service_start_s.eq(0).all()


def test_unavailability_does_not_switch_off_host():
    _,net,carbon,cfg,q=fixtures();servers=dgx_servers()
    unavailable=Availability([(0,'DE','L',0)])
    sim=Simulation([q],servers,net,unavailable,carbon,{'one':1},'Fixed',cfg).run()
    result,records,_=evaluate(sim,200)
    assert result['idle_kwh']==pytest.approx(1.16)
    assert result['failed']==1 and result['active_increment_kwh']==0


def test_DGX_cross_hour_increment_integral():
    _,net,carbon,cfg,q=fixtures();servers=dgx_servers()
    i=carbon.index(0);carbon.values[i,0]=100;carbon.values[i+1,0]=200
    q=replace(q,arrival=3590,deadline=3700,service=20)
    sim=Simulation([q],servers,net,Availability(),carbon,{'one':1},'Fixed',cfg).run()
    result,_,_=evaluate(sim,3800)
    assert result['active_increment_co2e_g']==pytest.approx(.585)


def test_cache_fingerprint_includes_current_upstream_and_power_contract():
    sig=input_signature();paths=upstream_paths()
    for p in [paths['resource_config'],paths['resource_core'],paths['run']/'resource_assumptions.json',
              DATA/'resource_accounting_contract.json',DATA/'physical_host_config.csv',ROOT/'configs/upstream.json']:
        assert str(p) in sig['files']
    assert profile()['gpu_busy_w']==760


def test_current_saved_Pre_provenance_is_disjoint_from_test():
    paths=upstream_paths();run=paths['run']
    splits=pd.read_csv(run/'splits.csv')
    provenance=read(run/'pre_provenance.json')['Pre_full_train']
    trained=set(provenance['training_groups'])
    assert trained==set(splits[splits.phase.eq('train')].group)
    assert trained.isdisjoint(set(splits[splits.phase.eq('test')].group))
