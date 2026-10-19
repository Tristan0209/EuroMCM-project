import gzip
import json
import numpy as np
import pandas as pd
from .sources import read, sha
from .resources import host_for_replica, pools_from_hosts
from .world import ExogenousWorld


def require(condition, message):
    if not bool(condition):
        raise AssertionError(message)


def validate_execution(folder, world=None):
    folder = folder.resolve()
    world = world or ExogenousWorld.load(folder.parents[2] / 'worlds' / read(folder/'execution.json')['world_id'])
    world.verify()
    records = pd.read_csv(folder/'requests.csv.gz', dtype={'request_id': str}, keep_default_na=True)
    ledger = pd.read_csv(folder/'energy_ledger.csv.gz')
    events = pd.read_csv(folder/'events.csv.gz', dtype={'request_id': str})
    controls = pd.read_csv(folder/'controls.csv.gz')
    audit = pd.read_csv(folder/'forecast_audit.csv.gz')
    summary = read(folder/'metrics.json')
    with gzip.open(folder/'solver_log.jsonl.gz', 'rt', encoding='utf-8') as stream:
        logs = [json.loads(line) for line in stream]
    require(records.request_id.is_unique and set(records.request_id) == set(world.instances.instance_id), 'Input ID mass')
    require(records.status.isin(['completed', 'failed', 'censored_at_H']).all(), 'Invalid terminal stage')
    for stage, key in [('completed','completed'), ('failed','failed'), ('censored_at_H','censored')]:
        require(int(records.status.eq(stage).sum()) == summary[key], 'Terminal summary mismatch')
    require(sum(summary[k] for k in ['completed','failed','censored']) == len(records), 'Terminal conservation')
    require(summary['horizon_s'] == 43200 and (events.time_s <= 43200).all(), 'Fixed window')
    require((ledger.start_s >= 0).all() and (ledger.end_s <= 43200).all() and
            (ledger.end_s >= ledger.start_s).all(), 'Energy intervals')
    require(np.isfinite(ledger[['energy_kwh','co2e_g']]).all().all() and
            (ledger[['energy_kwh','co2e_g']] >= 0).all().all(), 'Finite nonnegative accounting')
    pools, net, availability, carbon, shocks, _ = world.environment()
    capacity = {(r,m): c for t,r,m,c in availability.events if t == 0}
    running, started, completed = {}, set(), set()
    for event in events.to_dict('records'):
        kind, rid = event['event'], event.get('request_id')
        if kind == 'CapacityChange':
            capacity[(event['region'], event['model'])] = int(event['capacity'])
        elif kind == 'ServiceStart':
            key = (event['region'], event['model'], int(event['replica']))
            require(rid not in started and key not in running, 'Duplicate service / occupied replica')
            require(key[2] < capacity[key[:2]] and event['admission_capacity'] == capacity[key[:2]], 'Admission barrier')
            running[key], started = rid, started | {rid}
        elif kind == 'ServiceCompletion':
            keys = [key for key,value in running.items() if value == rid]
            require(len(keys) == 1 and rid not in completed, 'Completion consistency')
            del running[keys[0]]
            completed.add(rid)
    require(len(started) == summary['completed']+summary['running_at_H'], 'Start/completion/backlog conservation')
    require(summary['unfinished_at_H'] == summary['censored'], 'Censor/backlog mismatch')
    expected_idle = sum((h['shared_w']+h['gpus']*h['gpu_idle_w'])*h['pue']*43200/3.6e6 for h in world.hosts)
    require(np.isclose(summary['idle_kwh'], expected_idle, rtol=0, atol=1e-9), 'Physical idle once per host')
    require(ledger.loc[ledger.component.eq('idle'), 'host_id'].nunique() == len(world.hosts), 'Physical host ledger')
    expected_active = expected_net = 0.
    selected = records.set_index('request_id')
    active_ledger = {rid:part for rid,part in ledger[ledger.component.eq('active_increment')].groupby('request_id')}
    for rid in started:
        rec = selected.loc[rid]
        require(np.isclose(rec.actual_service_s, rec.estimated_service_s*shocks[rid], rtol=1e-12), 'Action service shock')
        pool = pools[(rec.region, rec.selected_model)]
        expected_active += max(0., min(rec.actual_service_s, 43200-rec.service_start_s))*(pool['busy']-pool['idle'])*pool['pue']/3.6e6
        part = active_ledger[rid]
        require(len(part) == 1 and part.host_id.iloc[0] == host_for_replica(pool,int(rec.replica)), 'Started task ledger')
    for rec in records.to_dict('records'):
        if pd.notna(rec.get('dispatch_s')):
            expected_net += net[(rec['origin_region'],rec['region'])]['energy']
    require(np.isclose(summary['active_increment_kwh'], expected_active, rtol=0, atol=1e-9), 'Partial active conservation')
    require(np.isclose(summary['network_kwh'], expected_net, rtol=0, atol=1e-9), 'Network once, including failed')
    execution = read(folder/'execution.json')
    cfg = execution['config']
    expected_control = sum(max(0., min(43200,row.end_s)-max(0.,row.start_s))*cfg['controller_w']*cfg['controller_pue']/3.6e6
                           for row in controls.itertuples())
    require(np.isclose(summary['controller_kwh'], expected_control, rtol=0, atol=1e-9), 'Partial controller conservation')

    integrated = 0.
    for region, part in ledger.groupby('region'):
        left, right, energy = [part[c].to_numpy(float) for c in ['start_s','end_s','energy_kwh']]
        duration = right-left
        continuous = duration > 0
        for h in range(12):
            overlap = np.maximum(0, np.minimum(right, (h+1)*3600)-np.maximum(left,h*3600))
            integrated += float(np.sum(energy[continuous]*overlap[continuous]/duration[continuous]))*carbon.at(h*3600,region)
        for t, e in zip(left[~continuous], energy[~continuous]):
            integrated += e*carbon.at(t,region)
    require(np.isclose(integrated, summary['co2e_g'], rtol=0, atol=1e-7), 'Independent carbon integration')
    for unit,key in [('energy_kwh','energy_kwh'),('co2e_g','co2e_g')]:
        require(np.isclose(ledger[unit].sum(), summary[key], rtol=0, atol=1e-8), 'Total ledger conservation')
    completed_rows = records.status.eq('completed')
    deadline_bad = ~completed_rows | (records.get('completion_s', pd.Series(np.nan,index=records.index)) > records.deadline_s+1e-6)
    wait_bad = completed_rows & ((records.get('intentional_delay_s',0)+records.get('queue_wait_s',0)) > records.max_wait_s+1e-5)
    require(np.array_equal(records.service_constraint_violation, deadline_bad | wait_bad), 'All-input SLA denominator')
    require(all(l['max_request_arrival_s'] <= l['decision_s']+1e-8 and l['observed_carbon_end_s'] <= l['decision_s']+1e-8 for l in logs), 'Solver causal metadata')
    require(len(audit) > 0 and (audit.observed_end_s <= audit.issued_s+1e-8).all(), 'Forecast past-only cutoff')
    quality = pd.read_csv(folder/'quality_harms.csv.gz')
    require(set(quality.request_id) == set(records.request_id), 'Posthoc quality completeness')
    require(np.allclose(quality.end_to_end_harm, quality.model_selection_harm+quality.delivery_incremental_harm, atol=1e-12), 'Quality additivity')
    failure = pd.read_csv(folder/'failure_causes.csv.gz')
    require(len(failure) == len(records) and failure.primary_service_outcome.notna().all(), 'Failure decomposition')
    return {'status': 'PASS', 'requests': len(records), 'checks': ['IDs/terminals', 'H', 'admission/queue lifecycle',
            'partial busy/controller', 'host/PUE/network', 'independent carbon quadrature', 'causal metadata', 'quality/causes'],
            'world_id': world.world_id, 'scope': 'software conservation; not empirical validity of assumptions'}
