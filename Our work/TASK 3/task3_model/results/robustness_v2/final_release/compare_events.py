from context import *
from analyze_release import locate
import gzip
import numpy as np
import pandas as pd

def main():
    summaries=[]
    for phase in ['core','F','H']:
        new=pd.read_csv(out(phase)/'raw_runs.csv')
        oldpath={'core':'E_nested_ablation','F':'F_sla_risk','H':'H_controller_overhead'}[phase]
        old=pd.read_csv(PREVIOUS/'results'/oldpath/'raw_runs.csv')
        if phase=='H':
            old=pd.concat([old,pd.read_csv(PREVIOUS/'results/H_controller_overhead/efficient_spatial_raw_runs.csv')])
            old=old[old.strategy.ne('Greedy')]
        keys=['strategy','seed','gamma','delay_scale'];cases=new.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
        fields=['time_s','selected_model','region','model','capacity','replica','active','admission_capacity','status']
        columns=keys+['event','request_id','occurrence','presence']+[c+s for c in fields for s in ['_old','_new']]
        with gzip.open(out(phase)/'before_after_changed_events.csv.gz','wt',encoding='utf-8',newline='') as stream:
            pd.DataFrame(columns=columns).to_csv(stream,index=False)
            for case in cases.to_dict('records'):
                a=pd.read_csv(locate(case['artifact_directory_old'])/'events.csv.gz')
                b=pd.read_csv(Path(case['artifact_directory_new'])/'events.csv.gz')
                for f in [a,b]:
                    for c in fields:
                        if c not in f:f[c]=np.nan
                    f['occurrence']=f.groupby(['event','request_id'],dropna=False).cumcount()
                index=['event','request_id','occurrence']
                merged=a[index+fields].merge(b[index+fields],on=index,how='outer',suffixes=('_old','_new'),indicator='presence',validate='one_to_one')
                mask=merged.presence.ne('both')
                for c in fields:
                    x=merged[c+'_old'];y=merged[c+'_new']
                    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):mask|=~np.isclose(x,y,rtol=0,atol=1e-8,equal_nan=True)
                    else:mask|=~(x.eq(y)|(x.isna()&y.isna()))
                changed=merged[mask].copy()
                for k in keys:changed[k]=case[k]
                changed[columns].to_csv(stream,index=False,header=False)
                summaries.append({**{k:case[k] for k in keys},'phase':phase,'aligned_events':len(merged),'changed_or_missing_events':int(mask.sum())})
    pd.DataFrame(summaries).to_csv(ROOT/'audit/event_difference_summary.csv',index=False)
    print('Compared',len(summaries),'phase references; duplicate case references do not add statistical samples')
if __name__=='__main__':main()
