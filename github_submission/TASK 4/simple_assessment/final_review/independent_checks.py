from pathlib import Path
import math
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SIMPLE = HERE.parent
sys.path.insert(0,str(SIMPLE.parent))
import numpy as np
import pandas as pd
from task4_core.sources import read,sha
from task4_core.serialization import save
from task4_core.registry import reusable


def near(a,b,tol=1e-10):
    if not math.isclose(float(a),float(b),rel_tol=0,abs_tol=tol):
        raise AssertionError((a,b))


if __name__=='__main__':
    completion = read(SIMPLE/'runs/completion.json')['rows']
    lookup = {(r['scenario'],r['seed'],r['policy']):Path(r['record']).parent for r in completion}
    flips = pd.read_csv(HERE/'prediction_flips.csv')
    router = pd.read_csv(HERE/'router_vs_AllL.csv')
    hourly = pd.read_csv(HERE/'hourly_active_ledger.csv')
    region = pd.read_csv(HERE/'region_execution.csv')
    energy = pd.read_csv(HERE/'energy_comparison.csv')
    rb = pd.read_csv(HERE/'rebound_audit.csv')
    results = []

    for seed in [2029101800,2029101801,2029101802]:
        a = pd.read_csv(lookup[('Baseline',seed,'Greedy')]/'quality_harms.csv.gz').set_index('base_request_id').to_dict('index')
        b = pd.read_csv(lookup[('Prediction Error',seed,'Greedy')]/'quality_harms.csv.gz').set_index('base_request_id').to_dict('index')
        pairs = [(rid,a[rid],b[rid]) for rid in a if a[rid]['selected_model']!=b[rid]['selected_model']]
        d = flips[(flips.seed==seed)&(flips.task_type=='All')].iloc[0]
        assert len(pairs)==d.n_changed
        nls = sum(v0['selected_model']=='L' for _,v0,v1 in pairs)
        assert nls==d.L_to_S and len(pairs)-nls==d.S_to_L
        values = [(v0['q_'+v0['selected_model'].lower()],v1['q_'+v1['selected_model'].lower()]) for _,v0,v1 in pairs]
        near(math.fsum(q0 for q0,q1 in values)/len(values),d.quality_before_changed)
        near(math.fsum(q1 for q0,q1 in values)/len(values),d.quality_after_changed)
        near(math.fsum(q1-q0 for q0,q1 in values)/len(values),d.quality_delta_changed)
        near(sum(q0>=.5 for q0,q1 in values)/len(values),d.pass_before_changed)
        near(sum(q1>=.5 for q0,q1 in values)/len(values),d.pass_after_changed)
        assert sum(q1-q0<=-.1+1e-12 for q0,q1 in values)==d.material_drop_count
        assert sum(v0['selected_model']=='L' and v1['selected_model']=='S' and v0['q_l']>=.5 and v0['q_s']<.5 for _,v0,v1 in pairs)==d.lost_L_pass_due_to_new_S
        near(math.fsum(q1-q0 for q0,q1 in values)/len(a),d.population_quality_delta)
        results.append({'check':'scalar paired flips','seed':seed,'status':'PASS'})
    for key,folder in lookup.items():
        r = read(folder/'record.json')
        assert reusable(r,r['config'])
        f = pd.read_csv(folder/'quality_harms.csv.gz')
        q = pd.read_csv(folder/'routing.csv.gz')
        decision = q.fallback_reason.isna() | q.fallback_reason.eq('')
        proposed = np.where(q.loc[decision,'perturbed_L']-q.loc[decision,'perturbed_S']-.5*q.loc[decision,'cost_delta']>=q.loc[decision,'threshold'],'L','S')
        assert np.array_equal(proposed,q.loc[decision,'selected_model'])
        assert q.loc[~decision,'selected_model'].eq('L').all()
        e = energy[(energy.scenario==key[0])&(energy.seed==key[1])&(energy.policy==key[2])].iloc[0]
        hs = hourly[(hourly.scenario==key[0])&(hourly.seed==key[1])&(hourly.policy==key[2])]
        near(math.fsum(hs.active_energy_kwh),e.active_increment_energy_kwh)
        near(math.fsum(hs.active_carbon_g),e.active_increment_carbon_g,1e-8)
        rs = region[(region.scenario==key[0])&(region.seed==key[1])&(region.policy==key[2])]
        near(math.fsum(rs.active_energy_share),1.)
        for task in ['General','Math','Code']:
            fp = f[f.task_type==task].to_dict('records')
            count = len(fp)
            quality = [x['q_'+x['selected_model'].lower()] for x in fp]
            ql = [x['q_l'] for x in fp]
            saved = router[(router.scenario==key[0])&(router.seed==key[1])&(router.task_type==task)].iloc[0]
            near(math.fsum(a-b for a,b in zip(quality,ql))/count,saved.delta_quality)
            near(sum(int(a>=.5)-int(b>=.5) for a,b in zip(quality,ql))/count,saved.delta_pass)
            if task in ['Code','Math']:
                assert saved.delta_quality==saved.delta_pass==0
        results.append({'check':'routing formula, All-L paired quality, independent hour accounting','scenario':key[0],
                        'seed':key[1],'policy':key[2],'status':'PASS'})
    for row in rb.itertuples():
        near(row.c_incremental_baseline,(row.C_Spatial0-row.C_idle)/row.N0)
        near(row.c_incremental_growth,(row.C_Spatial_growth-row.C_idle)/row.Ng)
        near(row.r_linear_reference,(row.C_Greedy0-row.C_idle)/(row.N0*row.c_incremental_baseline)-1.)
        near(row.C_idle+(1+row.r_linear_reference)*row.N0*row.c_incremental_baseline,row.C_Greedy0,1e-8)
        near(row.cross_demand_delta_fraction,(row.C_Spatial_growth-row.C_Greedy0)/row.C_Greedy0)
        assert row.DES_minus_linear_g>0
    long = pd.read_csv(SIMPLE/'Task4_Additional_Analysis.csv')
    assert pd.api.types.is_numeric_dtype(long.value)
    assert not long[['analysis','scenario','seed','policy','task_type','region','hour','metric']].duplicated().any()
    nulls = long[long.value.isna()]
    assert set(nulls.analysis)=={'prediction_flips'}
    assert set(nulls.task_type)<=set(['Math','Code'])
    assert long[long.metric.eq('execution_mix_intensity_term_g')].unit.eq('gCO2e').all()
    assert long[long.metric.eq('c_incremental_growth')].unit.eq('gCO2e/input').all()
    for (section,source_path),part in long.groupby(['analysis','source_table']):
        table = pd.read_csv(SIMPLE/source_path)
        identifiers = [k for k in ['scenario','seed','policy','task_type','region','hour'] if k in table]
        expected_rows = len(table)*(len(table.columns)-len(identifiers))
        assert len(part)==expected_rows,(section,expected_rows,len(part))
        for row in part.to_dict('records'):
            match = table
            for k in identifiers:
                match = match[match[k].eq(row[k])]
            assert len(match)==1
            value = match.iloc[0][row['metric']]
            if pd.isna(value):
                assert pd.isna(row['value'])
            else:
                near(value,row['value'],1e-9)
    save(HERE/'independent_checks.json',{'status':'PASS','checks':results,'new_DES_runs':0,'new_model_inference':0,
        'rebound_formula_checks':3,'long_csv_values_checked':len(long),'preserved_empty_subset_NA':len(nulls),
        'original_result_numeric_mismatch':0,'new_statistics_units_checked':True,
        'comment':'Independent scalar paired comparisons and saved-ledger hourly reconstruction; no new simulation.'})
    print('PASS independent calculations; tidy values',len(long),'NA',len(nulls))
