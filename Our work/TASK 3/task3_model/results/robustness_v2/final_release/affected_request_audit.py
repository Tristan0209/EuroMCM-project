from context import *
from analyze_release import locate
from functools import lru_cache
import pandas as pd

@lru_cache(maxsize=24)
def records(path):return pd.read_csv(Path(path)/'requests.csv.gz').set_index('request_id')

def main():
    counts=[]
    for phase,oldname in [('core','E_nested_ablation'),('F','F_sla_risk'),('H','H_controller_overhead')]:
        new=pd.read_csv(out(phase)/'raw_runs.csv');old=pd.read_csv(PREVIOUS/'results'/oldname/'raw_runs.csv')
        if phase=='H':old=pd.concat([old,pd.read_csv(PREVIOUS/'results/H_controller_overhead/efficient_spatial_raw_runs.csv')])
        keys=['strategy','seed','gamma','delay_scale']
        cases=new.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one').set_index(keys)
        timing=[]
        for index,case in cases.iterrows():
            base=dict(zip(keys,index))
            for metric in ['solver_wall_seconds','modelled_optimizer_seconds','solver_calls','fallback_count','solver_nonoptimal_count']:
                timing.append({**base,'metric':metric,'old':case[metric+'_old'],'new':case[metric+'_new'],
                    'new_minus_old':case[metric+'_new']-case[metric+'_old']})
            old_seconds=case['controller_kwh_old']*3.6e6/12.
            new_seconds=case['controller_kwh_new']*3.6e6/12.
            timing.append({**base,'metric':'all_controller_computing_seconds_including_upstream_router',
                'old':old_seconds,'new':new_seconds,'new_minus_old':new_seconds-old_seconds})
        pd.DataFrame(timing).to_csv(out(phase)/'before_after_controller_timing.csv',index=False)
        try:affected=pd.read_csv(out(phase)/'numerical_affected_batches.csv')
        except pd.errors.EmptyDataError:affected=pd.DataFrame()
        rows=[]
        for batch in affected.to_dict('records'):
            case=cases.loc[tuple(batch[k] for k in keys)]
            a=records(str(locate(case['artifact_directory_old'])));b=records(case['artifact_directory_new'])
            for rid in str(batch['request_ids']).split(';'):
                row={k:batch[k] for k in keys}
                row.update(decision_s=batch['decision_s'],request_id=rid,branch_index=batch['branch_index'],
                    selected_branch=batch['selected_branch'],affected_branch_selected=batch['branch_index']==batch['selected_branch'],
                    restored=bool(batch['recovered']),unrecoverable=bool(batch['unrecoverable']),
                    recovery_evidence=batch['recovery_evidence'],rejection_reason=batch['rejection_reason'])
                for c in ['region','status','dispatch_s','service_start_s','completion_s','fallback_reason','failure_reason']:
                    row[c+'_old']=a.loc[rid].get(c);row[c+'_new']=b.loc[rid].get(c)
                rows.append(row)
        pd.DataFrame(rows).to_csv(out(phase)/'numerical_affected_request_outcomes.csv.gz',index=False,compression='gzip')
        counts.append({'phase':phase,'rejected_branch_request_incidences':len(rows),'note':'batch/request incidences, not independent samples; unused branches retained'})
    pd.DataFrame(counts).to_csv(ROOT/'audit/affected_request_audit_summary.csv',index=False)
if __name__=='__main__':main()
