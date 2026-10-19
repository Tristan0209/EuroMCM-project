import numpy as np
import pandas as pd
from .power import idle_base_w,incremental_w


def evaluate(sim,horizon,quality=None):
    if horizon<max(rec.get('completion_s',0.) for rec in sim.rec.values()):raise ValueError('Common window misses completions')
    carbon=sim.actual;components=[]
    def add(component,region,start,end,watts=None,kwh=None,carbon_g=None,rid=None):
        if start<0 or end>horizon+1e-6:raise ValueError('Energy interval outside common window')
        if watts is not None:
            energy=watts*(end-start)/3.6e6
            emissions=watts*carbon.integral(start,end,region)/3.6e6
        else:energy=kwh;emissions=carbon_g
        components.append({'component':component,'region':region,'start_s':start,'end_s':end,'energy_kwh':energy,'co2e_g':emissions,'request_id':rid})
    for (r,m),pool in sim.servers.items():
        add('idle',r,0,horizon,watts=idle_base_w(pool)*pool['pue'])
    rows=[]
    for rid,rec in sim.rec.items():
        q=sim.q[rid];row={**rec};row['historical_quality']=quality.get(rid) if quality else None
        complete=rec['status']=='completed'
        row['deadline_violation']=not complete or rec.get('completion_s',np.inf)>q.deadline+1e-6
        row['wait_violation']=complete and rec['intentional_delay_s']+rec['queue_wait_s']>q.wait+1e-5
        row['service_constraint_violation']=row['deadline_violation'] or row['wait_violation']
        row['quality_is_evaluation_only']=True
        if complete:
            r=rec['region'];pool=sim.servers[(r,q.model)]
            add('active_increment',r,rec['service_start_s'],rec['completion_s'],watts=incremental_w(pool)*pool['pue'],rid=rid)
        if 'dispatch_s' in rec:
            r=rec['region'];kwh=sim.link(q,r)['energy']/2
            for region in [q.origin,r]:add('network',region,rec['dispatch_s'],rec['dispatch_s'],kwh=kwh,
                carbon_g=kwh*carbon.at(rec['dispatch_s'],region),rid=rid)
        rows.append(row)
    for start,end,r,kind in sim.controls:

        add('controller',r,start,end,watts=sim.cfg['controller_w']*1.2)
    ledger=pd.DataFrame(components);records=pd.DataFrame(rows)
    completed=records[records.status.eq('completed')]
    total=ledger.groupby('component')[['energy_kwh','co2e_g']].sum()
    response=completed.response_s if len(completed) else pd.Series(dtype=float)
    def quantile(q):return float(response.quantile(q)) if len(response) else None
    summary={'strategy':sim.strategy,'requests':len(records),'completed':len(completed),'failed':int(records.status.eq('failed').sum()),
        'completion_rate':len(completed)/len(records),'deadline_violation_rate':float(records.deadline_violation.mean()),
        'wait_violation_rate':float(records.wait_violation.mean()),'service_violation_rate':float(records.service_constraint_violation.mean()),
        'all_service_constraints_met':bool(not records.service_constraint_violation.any()),
        'mean_response_s':float(response.mean()) if len(response) else None,'p50_response_s':quantile(.5),'p95_response_s':quantile(.95),'p99_response_s':quantile(.99),
        'mean_queue_wait_s':float(completed.queue_wait_s.mean()) if len(completed) else None,
        'mean_intentional_delay_s':float(completed.intentional_delay_s.mean()) if len(completed) else None,
        'energy_kwh':float(ledger.energy_kwh.sum()),'co2e_g':float(ledger.co2e_g.sum()),'horizon_s':horizon,
        'historical_quality_fixed_cohort':float(records.historical_quality.mean()) if quality else None,
        'delivered_quality_utility_per_input':float(completed.historical_quality.sum()/len(records)) if quality else None,
        'fallback_count':sim.fallbacks,'solver_calls':len(sim.solver_log),
        'highs_calls':sum(log['engine'].startswith('HiGHS') for log in sim.solver_log),
        'solver_wall_seconds':sum(log['wall_seconds'] for log in sim.solver_log),
        'solver_timeout_count':sum(log['status']==1 for log in sim.solver_log),
        'solver_nonoptimal_count':sum(log['status']!=0 for log in sim.solver_log),
        'modelled_optimizer_seconds':sum(log['modelled_controller_s'] for log in sim.solver_log),
        'forecast_kind':'perfect_future_actual' if sim.strategy=='Oracle' else sim.cfg['forecast_method'],
        'optimality_scope':'offline batch-restricted finite MILPs, not global stochastic optimum' if sim.strategy=='Oracle' else 'online finite-candidate decisions'}
    for component in ['idle','active_increment','network','controller']:
        summary[component+'_kwh']=float(total.loc[component,'energy_kwh']) if component in total.index else 0.
        summary[component+'_co2e_g']=float(total.loc[component,'co2e_g']) if component in total.index else 0.
    if not np.isclose(summary['energy_kwh'],sum(summary[x+'_kwh'] for x in ['idle','active_increment','network','controller']),rtol=1e-12):raise AssertionError('Energy decomposition failed')
    return summary,records,ledger
