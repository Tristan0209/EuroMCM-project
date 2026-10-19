from pathlib import Path
import sys
import json
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SIMPLE = HERE.parent
TASK4 = SIMPLE.parent
sys.path.insert(0,str(TASK4))
import numpy as np
import pandas as pd
from task4_core.sources import read,sha,TASK2
from task4_core.serialization import save
from task4_core.registry import reusable

SEEDS = [2029101800,2029101801,2029101802]
SCENARIOS = ['Baseline','Difficult','Prediction Error','Demand Growth']
POLICIES = ['Greedy','Spatial_MILP']
TASKS = ['General','Math','Code']
PASS = .5
MATERIAL_DROP = .10
TABLES = {}


def metric_unit(metric):
    if metric=='same_actual_region_hour_carbon':
        return 'boolean'
    if metric.endswith('_multiplier'):
        return 'ratio'
    if metric in ['c_incremental_baseline','c_incremental_growth','incremental_carbon_per_input_g']:
        return 'gCO2e/input'
    if 'kwh' in metric:
        return 'kWh'
    if metric.endswith('_g') or metric.startswith('C_'):
        return 'gCO2e'
    if 'intensity' in metric:
        return 'gCO2e/kWh'
    if metric.endswith('_s'):
        return 'seconds'
    if (metric.startswith('n_') or metric.startswith('unique_') or metric.endswith('_count') or
        metric in ['L_to_S','S_to_L','S_count','L_count','N0','Ng','dispatched','started','completed',
                   'q_S_greater_than_q_L_instances','lost_L_pass_due_to_new_S']):
        return 'requests_or_unique_IDs_as_named'
    if metric in ['changed_quality_decrease_sum','changed_quality_increase_sum']:
        return 'sum_of_scores'
    if metric=='material_drop_threshold':
        return 'absolute_score'
    if any(w in metric for w in ['fraction','rate','pass','share','weight']) or metric in ['r_linear_reference','growth_actual']:
        return 'fraction'
    if 'quality' in metric or 'score' in metric:
        return 'score'
    raise ValueError('Explicit unit required for '+metric)


def record_table(name, rows):
    frame = pd.DataFrame(rows)
    TABLES[name] = frame
    frame.to_csv(HERE/(name+'.csv'),index=False,encoding='utf-8-sig')
    return frame


def quality(frame):
    return np.where(frame.selected_model.eq('L'),frame.q_l,frame.q_s)


def main():
    assert (HERE/'before.json').is_file(),'Save preservation snapshot first'
    rows = read(SIMPLE/'runs/completion.json')['rows']
    assert len(rows)==24
    frames,ledgers,routes,metrics,sources,worlds = {},{},{},{},{},{}
    original = pd.read_csv(SIMPLE/'Task4_Simple_Results.csv')
    audits = []

    wanted = set()
    for row in rows:
        record = read(row['record'])
        assert reusable(record,record['config']),row['record']
        key = (row['scenario'],row['seed'],row['policy'])
        folder = Path(record['folder'])
        f = pd.read_csv(folder/'quality_harms.csv.gz')
        route = pd.read_csv(folder/'routing.csv.gz')
        assert f.request_id.is_unique and route.request_id.is_unique
        f = f.set_index('request_id',drop=False)
        route = route.set_index('request_id',drop=False).loc[f.index]
        assert f.selected_model.equals(route.selected_model)
        assert f.base_request_id.equals(route.base_request_id)
        assert set(f.index)==set(pd.read_csv(folder/'requests.csv.gz').request_id)
        frames[key], routes[key] = f,route
        ledgers[key] = pd.read_csv(folder/'energy_ledger.csv.gz')
        metrics[key] = read(folder/'metrics.json')
        sources[key] = str(folder)
        world = read(SIMPLE/'runs/worlds'/row['world_id']/'world.json')
        worlds[(row['scenario'],row['seed'])] = world
        assert len(f)==len(world['instances'])
        wanted.update(f.base_request_id)
    archive = []
    with (TASK2/'inputs/paired/outcomes.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            r = json.loads(line)
            if r['request_id'] in wanted:
                archive.append({k:r[k] for k in ['request_id','action_id','quality_true']})
    truth = pd.DataFrame(archive).pivot(index='request_id',columns='action_id',values='quality_true')
    assert set(truth.index)==wanted
    for key,f in frames.items():
        for m in ['S','L']:
            np.testing.assert_allclose(f['q_'+m.lower()],f.base_request_id.map(truth[m]),atol=1e-14,rtol=0)
        qm = quality(f)
        done = f.status.eq('completed')
        valid = done & ~f.service_constraint_violation
        saved = original[(original.scenario==key[0])&(original.seed==key[1])&(original.policy==key[2])].iloc[0]
        calculated = {'mean_historical_quality':qm.mean(),'answer_pass_rate':(qm>=PASS).mean(),
            'valid_answer_pass_rate':(valid & (qm>=PASS)).mean(),'completion_rate':done.mean(),
            'service_violation_rate':f.service_constraint_violation.mean(),
            'p95_response_s':f.loc[done,'response_s'].quantile(.95),'requests':len(f),
            'completed':done.sum(),'failed':f.status.eq('failed').sum(),'censored':f.status.eq('censored_at_H').sum()}
        errors = {k:float(v-saved[k]) for k,v in calculated.items()}
        assert all(abs(x)<1e-10 for x in errors.values()),(key,errors)
        ledger = ledgers[key]
        assert np.isclose(ledger.energy_kwh.sum(),saved.energy_kwh,atol=1e-9,rtol=0)
        assert np.isclose(ledger.co2e_g.sum(),saved.co2e_g,atol=1e-8,rtol=0)
        assert len(f)==calculated['completed']+calculated['failed']+calculated['censored']
        audits.append({'scenario':key[0],'seed':key[1],'policy':key[2],
                       'maximum_summary_difference':max(abs(x) for x in errors.values()),
                       'historical_quality_legacy_column_all_null':bool(f.historical_quality.isna().all()),
                       'authoritative_truth_columns':'q_s/q_l checked directly against paired/outcomes.jsonl',
                       'status':'PASS'})

    for s in SCENARIOS:
        for seed in SEEDS:
            a,b = [routes[(s,seed,p)] for p in POLICIES]
            pd.testing.assert_frame_equal(a,b,check_exact=True)
            assert worlds[(s,seed)]['world_id']==read(Path(sources[(s,seed,'Greedy')])/'execution.json')['world_id']
    record_table('original_summary_audit',audits)

    flip_rows,flip_details = [],[]
    for seed in SEEDS:
        b = frames[('Baseline',seed,'Greedy')]
        e = frames[('Prediction Error',seed,'Greedy')].loc[b.index]
        assert b.base_request_id.equals(e.base_request_id)
        np.testing.assert_array_equal(b.q_s,e.q_s)
        np.testing.assert_array_equal(b.q_l,e.q_l)
        q0,q1 = quality(b),quality(e)
        changed = b.selected_model.ne(e.selected_model).to_numpy()
        lost_s = b.selected_model.eq('L').to_numpy() & e.selected_model.eq('S').to_numpy() & (b.q_l>=PASS).to_numpy() & (b.q_s<PASS).to_numpy()
        for task in ['All']+TASKS:
            mask = np.ones(len(b),bool) if task=='All' else b.task_type.eq(task).to_numpy()
            c = mask & changed
            n = int(c.sum())
            delta = q1[c]-q0[c]
            vals = {'scenario':'Prediction Error','seed':seed,'policy':'Shared routing','task_type':task,
                'n_input':int(mask.sum()),'n_changed':n,'unique_changed_groups':b.loc[c,'group'].nunique(),
                'L_to_S':int((c & b.selected_model.eq('L').to_numpy()).sum()),
                'S_to_L':int((c & b.selected_model.eq('S').to_numpy()).sum()),'change_fraction':float(c.sum()/mask.sum()),
                'quality_before_changed':float(q0[c].mean()) if n else None,
                'quality_after_changed':float(q1[c].mean()) if n else None,
                'quality_delta_changed':float(delta.mean()) if n else None,
                'pass_before_changed':float((q0[c]>=PASS).mean()) if n else None,
                'pass_after_changed':float((q1[c]>=PASS).mean()) if n else None,
                'pass_delta_changed':float(((q1[c]>=PASS).astype(int)-(q0[c]>=PASS).astype(int)).mean()) if n else None,
                'quality_drop_count':int((delta < -1e-12).sum()),'quality_gain_count':int((delta>1e-12).sum()),
                'quality_unchanged_count':int(np.isclose(delta,0,atol=1e-12,rtol=0).sum()),
                'material_drop_count':int((delta<=-MATERIAL_DROP+1e-12).sum()),
                'material_drop_fraction_changed':float((delta<=-MATERIAL_DROP+1e-12).mean()) if n else None,
                'lost_L_pass_due_to_new_S':int((lost_s & mask).sum()),
                'changed_quality_abs_delta_mean':float(abs(delta).mean()) if n else None,
                'changed_quality_decrease_sum':float(-delta[delta<0].sum()),
                'changed_quality_increase_sum':float(delta[delta>0].sum()),
                'population_quality_delta':float((q1[mask]-q0[mask]).mean()),
                'population_pass_delta':float(((q1[mask]>=PASS).astype(int)-(q0[mask]>=PASS).astype(int)).mean()),
                'material_drop_threshold':MATERIAL_DROP}
            assert vals['n_changed']==vals['L_to_S']+vals['S_to_L']
            assert vals['n_changed']==vals['quality_drop_count']+vals['quality_gain_count']+vals['quality_unchanged_count']
            assert np.isclose(vals['population_quality_delta'],(n/vals['n_input'])*vals['quality_delta_changed'] if n else 0,atol=1e-14)
            flip_rows.append(vals)
        for i in np.flatnonzero(changed):
            rid = b.index[i]
            br,er = b.iloc[i],e.iloc[i]
            flip_details.append({'seed':seed,'instance_id':rid,'base_request_id':br.base_request_id,'prompt_group':br['group'],
                'task_type':br.task_type,'before_model':br.selected_model,'after_model':er.selected_model,
                'q_s':br.q_s,'q_l':br.q_l,'q_before':q0[i],'q_after':q1[i],'quality_delta':q1[i]-q0[i],
                'pass_before':q0[i]>=PASS,'pass_after':q1[i]>=PASS,'lost_L_pass_due_to_new_S':bool(lost_s[i]),
                'pre_S':routes[('Baseline',seed,'Greedy')].loc[rid,'pre_S'],
                'pre_L':routes[('Baseline',seed,'Greedy')].loc[rid,'pre_L'],
                'perturbed_S':routes[('Prediction Error',seed,'Greedy')].loc[rid,'perturbed_S'],
                'perturbed_L':routes[('Prediction Error',seed,'Greedy')].loc[rid,'perturbed_L']})
    record_table('prediction_flips',flip_rows)
    record_table('flipped_request_pairs',flip_details)

    router_rows,service_rows,energy_rows,region_rows,hour_rows = [],[],[],[],[]
    for s in SCENARIOS:
        for seed in SEEDS:
            f = frames[(s,seed,'Greedy')]
            for task in TASKS:
                part = f[f.task_type.eq(task)]
                qm = quality(part)
                ql = part.q_l.to_numpy()
                router_rows.append({'scenario':s,'seed':seed,'policy':'Shared routing vs offline All-L','task_type':task,
                    'n_input':len(part),'unique_base_ids':part.base_request_id.nunique(),'unique_prompt_groups':part['group'].nunique(),
                    'router_quality':qm.mean(),'AllL_quality':ql.mean(),'delta_quality':(qm-ql).mean(),
                    'router_pass':(qm>=PASS).mean(),'AllL_pass':(ql>=PASS).mean(),
                    'delta_pass':((qm>=PASS).astype(int)-(ql>=PASS).astype(int)).mean(),
                    'S_count':part.selected_model.eq('S').sum(),'L_count':part.selected_model.eq('L').sum(),
                    'S_fraction':part.selected_model.eq('S').mean(),'L_fraction':part.selected_model.eq('L').mean(),
                    'q_S_greater_than_q_L_instances':int((part.q_s>part.q_l).sum()),
                    'selected_S_better_than_L_count':int((part.selected_model.eq('S') & (part.q_s>part.q_l)).sum()),
                    'selected_S_worse_than_L_count':int((part.selected_model.eq('S') & (part.q_s<part.q_l)).sum()),
                    'selected_S_loses_L_pass_count':int((part.selected_model.eq('S') & (part.q_l>=PASS) & (part.q_s<PASS)).sum())})
                if task in ['Code','Math']:
                    assert router_rows[-1]['S_count']==0 and router_rows[-1]['delta_quality']==router_rows[-1]['delta_pass']==0
            for p in POLICIES:
                key = (s,seed,p)
                f = frames[key]
                ledger = ledgers[key]
                raw = metrics[key]
                er = {'scenario':s,'seed':seed,'policy':p,'task_type':'All','n_input':len(f),
                      'energy_kwh':ledger.energy_kwh.sum(),'carbon_g':ledger.co2e_g.sum()}
                for component in ['idle','active_increment','network','controller']:
                    part = ledger[ledger.component.eq(component)]
                    er[component+'_energy_kwh'] = part.energy_kwh.sum()
                    er[component+'_carbon_g'] = part.co2e_g.sum()
                er['operational_energy_kwh'] = er['energy_kwh']-er['idle_energy_kwh']
                er['operational_carbon_g'] = er['carbon_g']-er['idle_carbon_g']
                er['idle_carbon_fraction'] = er['idle_carbon_g']/er['carbon_g']
                er['active_effective_intensity'] = er['active_increment_carbon_g']/er['active_increment_energy_kwh']
                er['incremental_carbon_per_input_g'] = er['operational_carbon_g']/len(f)
                energy_rows.append(er)
                for task in TASKS:
                    part = f[f.task_type.eq(task)]
                    qm = quality(part)
                    done = part.status.eq('completed')
                    valid = done & ~part.service_constraint_violation
                    service_rows.append({'scenario':s,'seed':seed,'policy':p,'task_type':task,'n_input':len(part),
                        'unique_prompt_groups':part['group'].nunique(),'violation_rate':part.service_constraint_violation.mean(),
                        'completion_rate':done.mean(),'valid_completion_rate':valid.mean(),
                        'completed_p95_s':part.loc[done,'response_s'].quantile(.95),
                        'valid_pass_rate':(valid & (qm>=PASS)).mean(),'mean_queue_wait_completed_s':part.loc[done,'queue_wait_s'].mean()})
                active = ledger[ledger.component.eq('active_increment')]
                for region in ['DE','FR','ES']:
                    part = f[f.region.eq(region)]
                    a = active[active.region.eq(region)]
                    region_rows.append({'scenario':s,'seed':seed,'policy':p,'task_type':'All','region':region,
                        'dispatched':int(part.dispatch_s.notna().sum()),'started':int(part.service_start_s.notna().sum()),
                        'completed':int(part.status.eq('completed').sum()),
                        'active_energy_kwh':a.energy_kwh.sum(),'active_carbon_g':a.co2e_g.sum(),
                        'active_energy_share':a.energy_kwh.sum()/active.energy_kwh.sum(),
                        'effective_active_intensity':a.co2e_g.sum()/a.energy_kwh.sum() if a.energy_kwh.sum()>0 else None})
                carbon = pd.DataFrame(worlds[(s,seed)]['carbon_frame'])
                anchor = pd.Timestamp(worlds[(s,seed)]['anchor_utc'])
                carbon['hour'] = (pd.to_datetime(carbon.datetime_utc,utc=True)-anchor).dt.total_seconds()/3600
                ci = carbon.set_index(['hour','region']).carbon_intensity_gco2e_kwh.to_dict()

                for region,a in active.groupby('region'):
                    left,right,e = [a[c].to_numpy(float) for c in ['start_s','end_s','energy_kwh']]
                    duration = right-left
                    assert (duration>0).all()
                    for h in range(12):
                        overlap = np.maximum(0,np.minimum(right,(h+1)*3600)-np.maximum(left,h*3600))
                        energy = float(np.sum(e*overlap/duration))
                        hour_rows.append({'scenario':s,'seed':seed,'policy':p,'task_type':'All','region':region,'hour':h,
                            'active_energy_kwh':energy,'actual_intensity':ci[(float(h),region)],
                            'active_carbon_g':energy*ci[(float(h),region)]})
    router = record_table('router_vs_AllL',router_rows)
    service = record_table('group_service',service_rows)
    energy = record_table('energy_comparison',energy_rows)
    regions = record_table('region_execution',region_rows)
    hourly = record_table('hourly_active_ledger',hour_rows)
    differences = []
    for s in SCENARIOS:
        for seed in SEEDS:
            a,b = [energy[(energy.scenario==s)&(energy.seed==seed)&(energy.policy==p)].iloc[0] for p in POLICIES]
            differences.append({'scenario':s,'seed':seed,'policy':'Greedy minus Spatial','task_type':'All',
                'energy_saving_kwh':a.energy_kwh-b.energy_kwh,'energy_saving_fraction':(a.energy_kwh-b.energy_kwh)/a.energy_kwh,
                'carbon_saving_g':a.carbon_g-b.carbon_g,'carbon_saving_fraction':(a.carbon_g-b.carbon_g)/a.carbon_g,
                'operational_carbon_saving_fraction':(a.carbon_g-b.carbon_g)/a.operational_carbon_g})
    record_table('energy_paired',differences)

    rb = []
    for seed in SEEDS:
        a,b,g = [energy[(energy.scenario==s)&(energy.seed==seed)&(energy.policy==p)].iloc[0]
                 for s,p in [('Baseline','Greedy'),('Baseline','Spatial_MILP'),('Demand Growth','Spatial_MILP')]]
        n0,ng = int(b.n_input),int(g.n_input)
        assert np.isclose(a.idle_carbon_g,b.idle_carbon_g,atol=1e-9) and np.isclose(g.idle_carbon_g,b.idle_carbon_g,atol=1e-9)
        c0,cg = b.operational_carbon_g/n0,g.operational_carbon_g/ng
        rlin = (a.carbon_g-b.carbon_g)/b.operational_carbon_g
        crossing = b.idle_carbon_g+(1+rlin)*n0*c0
        assert np.isclose(crossing,a.carbon_g,atol=1e-9,rtol=0)
        predicted = b.idle_carbon_g+ng*c0
        rb.append({'scenario':'Demand Growth','seed':seed,'policy':'Spatial across demand','task_type':'All',
            'N0':n0,'Ng':ng,'growth_actual':ng/n0-1,'C_Greedy0':a.carbon_g,'C_Spatial0':b.carbon_g,'C_Spatial_growth':g.carbon_g,
            'C_idle':b.idle_carbon_g,'c_incremental_baseline':c0,'c_incremental_growth':cg,
            'incremental_per_input_multiplier':cg/c0,'r_linear_reference':rlin,
            'linear_crossing_residual_g':crossing-a.carbon_g,'growth_linear_prediction_g':predicted,
            'DES_minus_linear_g':g.carbon_g-predicted,'cross_demand_delta_g':g.carbon_g-a.carbon_g,
            'cross_demand_delta_fraction':(g.carbon_g-a.carbon_g)/a.carbon_g,
            'active_energy_multiplier':g.active_increment_energy_kwh/b.active_increment_energy_kwh,
            'active_intensity_multiplier':g.active_effective_intensity/b.active_effective_intensity})
    record_table('rebound_audit',rb)


    attribution = []
    for seed in SEEDS:
        b,g = [energy[(energy.scenario==s)&(energy.seed==seed)&(energy.policy=='Spatial_MILP')].iloc[0]
               for s in ['Baseline','Demand Growth']]
        e0,e1 = b.active_increment_energy_kwh,g.active_increment_energy_kwh
        i0,i1 = b.active_effective_intensity,g.active_effective_intensity
        increase_energy = (e1-e0)*i0
        increase_intensity = e1*(i1-i0)
        residual = (g.active_increment_carbon_g-b.active_increment_carbon_g)-increase_energy-increase_intensity
        assert abs(residual)<1e-10

        parts = [hourly[(hourly.scenario==s)&(hourly.seed==seed)&(hourly.policy=='Spatial_MILP')]
                 for s in ['Baseline','Demand Growth']]
        x = parts[0].merge(parts[1],on=['region','hour'],suffixes=('_b','_g'),validate='one_to_one')
        np.testing.assert_allclose(x.actual_intensity_b,x.actual_intensity_g,rtol=0,atol=0)
        attributed = ((x.active_energy_kwh_g-x.active_energy_kwh_b)*x.actual_intensity_b).sum()
        assert abs(attributed-(g.active_increment_carbon_g-b.active_increment_carbon_g))<1e-9
        attribution.append({'scenario':'Demand Growth','seed':seed,'policy':'Spatial across demand','task_type':'All',
            'active_carbon_change_g':g.active_increment_carbon_g-b.active_increment_carbon_g,
            'energy_volume_term_g':increase_energy,'execution_mix_intensity_term_g':increase_intensity,
            'identity_residual_g':residual,'same_actual_region_hour_carbon':True})
    record_table('active_carbon_identity',attribution)

    composition = []
    for seed in SEEDS:
        b = frames[('Baseline',seed,'Greedy')]
        d = frames[('Difficult',seed,'Greedy')]
        for task in TASKS:
            bp,dp = b[b.task_type.eq(task)],d[d.task_type.eq(task)]
            wb,wd = len(bp)/len(b),len(dp)/len(d)
            qb,qd = quality(bp),quality(dp)
            composition.append({'scenario':'Difficult','seed':seed,'policy':'Shared routing','task_type':task,
                'weight_baseline':wb,'weight_difficult':wd,'score_baseline':qb.mean(),'score_difficult':qd.mean(),
                'pass_baseline':(qb>=PASS).mean(),'pass_difficult':(qd>=PASS).mean(),
                'score_composition_component':(wd-wb)*qb.mean(),'score_within_component':wd*(qd.mean()-qb.mean()),
                'pass_composition_component':(wd-wb)*(qb>=PASS).mean(),
                'pass_within_component':wd*((qd>=PASS).mean()-(qb>=PASS).mean())})
    comp = record_table('task_composition',composition)
    for seed in SEEDS:
        c = comp[comp.seed==seed]
        b,d = [frames[(s,seed,'Greedy')] for s in ['Baseline','Difficult']]
        assert np.isclose(c.score_composition_component.sum()+c.score_within_component.sum(),quality(d).mean()-quality(b).mean(),atol=1e-14)
        assert np.isclose(c.pass_composition_component.sum()+c.pass_within_component.sum(),(quality(d)>=PASS).mean()-(quality(b)>=PASS).mean(),atol=1e-14)

    identifiers = {'scenario','seed','policy','task_type','region','hour'}
    long = []
    for section,table in TABLES.items():
        if section in ['flipped_request_pairs','original_summary_audit']:
            continue
        for row in table.to_dict('records'):
            for metric,value in row.items():
                if metric in identifiers:
                    continue
                unit = metric_unit(metric)
                long.append({'analysis':section,**{k:row.get(k,'') for k in ['scenario','seed','policy','task_type','region','hour']},
                    'metric':metric,'value':int(value) if isinstance(value,(bool,np.bool_)) else value,'unit':unit,'n_input':row.get('n_input',''),
                    'denominator_note':'changed subset where *_changed or changed fraction; all-input for population; seed-specific; see final_review data dictionary',
                    'provenance':'saved 24 executions; no new simulation'})
    dest = SIMPLE/'Task4_Additional_Analysis.csv'
    if dest.exists():
        draft = HERE/'additional_analysis_draft_before_unit_audit.csv'
        assert not draft.exists(),'Preserve previous draft'
        draft.write_bytes(dest.read_bytes())
    pd.DataFrame(long).to_csv(dest,index=False,encoding='utf-8-sig')
    save(HERE/'calculation_audit.json',{'status':'PASS','new_DES':0,'new_predictions':0,'new_training':0,
        'saved_executions_reused_read_only':24,'quality_archive_ids_checked':len(truth),
        'unique_flip_instances_across_seeds':pd.DataFrame(flip_details).base_request_id.nunique(),
        'long_statistic_rows':len(long),'pass_threshold':PASS,'material_drop_threshold':MATERIAL_DROP,
        'source_hashes':{str(SIMPLE/'Task4_Simple_Results.csv'):sha(SIMPLE/'Task4_Simple_Results.csv'),
                       str(TASK2/'inputs/paired/outcomes.jsonl'):sha(TASK2/'inputs/paired/outcomes.jsonl')}})
    print('PASS offline calculations: 24 saved executions, no new simulations; rows',len(long))
    print(pd.DataFrame(flip_rows).query("task_type=='All'").to_string(index=False))
    print(pd.DataFrame(rb).to_string(index=False))


if __name__=='__main__':main()
