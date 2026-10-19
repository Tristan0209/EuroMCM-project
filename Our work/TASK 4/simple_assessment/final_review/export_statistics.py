from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
sys.path.insert(0,str(HERE.parents[1]))
import numpy as np
import pandas as pd
from analyze_saved import metric_unit
from task4_core.serialization import save

NAMES = ['prediction_flips','router_vs_AllL','group_service','energy_comparison','region_execution',
         'hourly_active_ledger','energy_paired','rebound_audit','active_carbon_identity','task_composition']
ID = ['scenario','seed','policy','task_type','region','hour']

if __name__=='__main__':
    rows = []
    for section in NAMES:
        path = HERE/(section+'.csv')
        frame = pd.read_csv(path)
        for row in frame.to_dict('records'):
            for metric,value in row.items():
                if metric in ID:
                    continue
                if isinstance(value,(bool,np.bool_)):
                    value = int(value)
                rows.append({'analysis':section,**{k:row.get(k,'') for k in ID},'metric':metric,'value':value,
                    'unit':metric_unit(metric),'n_input':row.get('n_input',''),
                    'source_table':'final_review/'+path.name,
                    'scope':'one saved world/seed, or shared routing once per world; no new simulation'})
    target = HERE.parent/'Task4_Additional_Analysis.csv'
    pd.DataFrame(rows).to_csv(target,index=False,encoding='utf-8-sig')
    save(HERE/'statistics_schema.json',{'format':'tidy numeric statistics; one metric per row',
        'rows':len(rows),'scenario_count':4,'independent_seeds_per_scenario':3,
        'aggregation':'Each row is per seed. Report summaries are arithmetic means of the three seed-level quantities, never pooled requests.',
        'fraction_units':'Pass/violation/share/growth/relative savings stored as fractions. Reports multiply by 100 for % or percentage points.',
        'changed_quality_metrics':'Denominator n_changed; absent changed requests produce NA, not quality 0.',
        'material_drop':'q_after-q_before <= -0.10 absolute score; declared reporting convention, not clinical or calibrated quality standard.',
        'lost_L_pass_due_to_new_S':'Before L, after S, q_L>=0.5 and q_S<0.5; subset of L_to_S.',
        'all_l':'Offline paired quality reference. No All-L DES simulation or actual completion is implied.',
        'operational_boundary':'total minus fixed idle = active_increment + network + controller',
        'region_energy_share':'Share of active_increment electricity, not share of all facility electricity.',
        'effective_active_intensity':'active_increment CO2e divided by active_increment kWh, not full-window unweighted regional carbon.',
        'unique_prompt_groups':'Distinct normalized prompt groups; count is not a proof of statistical independence.',
        'missing_values':'NA retained for an empty changed subset or absent statistic; never replaced by zero',
        'no_new_DES':True})
    print('Exported',len(rows),'per-seed statistics with explicit numeric units')
