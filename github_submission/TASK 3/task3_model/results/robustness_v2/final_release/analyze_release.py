from context import *
import argparse,gzip
import numpy as np
import pandas as pd
from analyze import aggregate,paired,figure,METRICS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def locate(value):
    p=Path(value)
    if p.exists():return p
    normalized=str(value).replace('\\','/');marker='/EuroMCM-project/'
    if marker in normalized:
        candidate=PROJECT/normalized.split(marker,1)[1]
        if candidate.exists():return candidate
    raise FileNotFoundError('Preserved before-repair artifact not found: '+str(value))

def logs(folder):
    with gzip.open(Path(folder)/'solver_log.jsonl.gz','rt',encoding='utf-8') as f:
        for line in f:yield json.loads(line)

def equal(x,y):
    if pd.isna(x) and pd.isna(y):return True
    if isinstance(x,(float,int,np.number)) and isinstance(y,(float,int,np.number)):
        return bool(np.isclose(x,y,rtol=0,atol=1e-8,equal_nan=True))
    return x==y

def before_after(f,old,target):
    keys=['strategy','seed','gamma','delay_scale'];joined=f.merge(old,on=keys,suffixes=('_new','_old'),validate='one_to_one')
    assert len(joined)==len(f)
    joined['artifact_directory_old']=joined.artifact_directory_old.map(lambda x:str(locate(x)))
    delta=[];counts=[];changes=[];event_rows=[]
    for row in joined.to_dict('records'):
        base={k:row[k] for k in keys}
        for metric in METRICS+['solver_wall_seconds','solver_nonoptimal_count']:
            if metric+'_old' in row and metric+'_new' in row:
                delta.append({**base,'metric':metric,'old':row[metric+'_old'],'new':row[metric+'_new'],
                    'new_minus_old':row[metric+'_new']-row[metric+'_old']})
        a=pd.read_csv(Path(row['artifact_directory_old'])/'requests.csv.gz').set_index('request_id')
        b=pd.read_csv(Path(row['artifact_directory_new'])/'requests.csv.gz').set_index('request_id')
        assert set(a.index)==set(b.index) and len(a)==7335
        b=b.reindex(a.index)
        def differs(col):
            if pd.api.types.is_numeric_dtype(a[col]) and pd.api.types.is_numeric_dtype(b[col]):
                return pd.Series(~np.isclose(a[col].astype(float),b[col].astype(float),rtol=0,atol=1e-8,equal_nan=True),index=a.index)
            return ~(a[col].eq(b[col])|(a[col].isna()&b[col].isna()))
        for col in ['model','historical_quality','arrival_s','deadline_s']:
            if col in a and col in b:assert not differs(col).any(),col
        cols=[c for c in ['region','status','dispatch_s','service_start_s','completion_s','intentional_delay_s','queue_wait_s','service_constraint_violation'] if c in a and c in b]
        mask=pd.DataFrame({c:differs(c) for c in cols})
        for rid in mask.index[mask.any(axis=1)]:
            entry={**base,'request_id':rid,'changed_fields':';'.join(mask.columns[mask.loc[rid]]),
                'interpretation':'fresh measured wall and recovery protocol jointly changed; not an isolated LP causal effect' if row['delay_scale']>0 else 'ideal-control solver/recovery comparison; consult numerical_affected_batches'}
            for c in cols:entry[c+'_old']=a.loc[rid,c];entry[c+'_new']=b.loc[rid,c]
            changes.append(entry)
        counts.append({**base,'changed_requests':int(mask.any(axis=1).sum()),'changed_region':int(mask.region.sum()),'changed_service_start':int(mask.service_start_s.sum()),'changed_terminal_status':int(mask.status.sum())})

        ea=pd.read_csv(Path(row['artifact_directory_old'])/'events.csv.gz')
        eb=pd.read_csv(Path(row['artifact_directory_new'])/'events.csv.gz')
        typecol='event' if 'event' in ea else 'kind'
        if typecol in ea and typecol in eb:
            for event in sorted(set(ea[typecol])|set(eb[typecol])):
                event_rows.append({**base,'event':event,'old_count':int(ea[typecol].eq(event).sum()),'new_count':int(eb[typecol].eq(event).sum())})
    pd.DataFrame(delta).to_csv(target/'before_after_metrics.csv',index=False)
    pd.DataFrame(counts).to_csv(target/'before_after_request_counts.csv',index=False)
    change_columns=keys+['request_id','changed_fields','interpretation']+[c+suffix for c in cols for suffix in ['_old','_new']]
    pd.DataFrame(changes,columns=change_columns).to_csv(target/'before_after_changed_requests.csv.gz',index=False,compression='gzip')
    pd.DataFrame(event_rows).to_csv(target/'before_after_event_counts.csv',index=False)
    old_solver=[]
    for row in joined.to_dict('records'):
        counters={'subproblems':0,'status4':0,'timeout':0,'numeric_round_rejections':0}
        for log in logs(row['artifact_directory_old']):
            for b in log.get('branches',[log]):
                counters['subproblems']+=1;counters['status4']+=b['status']==4;counters['timeout']+=b['status']==1
                message=b.get('message','').lower()
                counters['numeric_round_rejections']+=b['status']==4 and ('constraint violation' in message or 'violates integer/linear' in message)
        old_solver.append({**{k:row[k] for k in keys},**counters})
    pd.DataFrame(old_solver).to_csv(target/'before_solver_status_counts.csv',index=False)

def numerical(f,target):
    import csv
    affected=[];counts=[];nonzero=[];writer=None
    stream=gzip.open(target/'solver_subproblems.csv.gz','wt',encoding='utf-8',newline='')
    for case in f.itertuples():
        for index,log in enumerate(logs(case.artifact_directory)):
            if log.get('holding_s',0)>0:nonzero.append({'strategy':case.strategy,'seed':case.seed,'decision_s':log['decision_s'],'holding_s':log['holding_s'],'zero_status':log['branches'][0]['status'],'selected_branch':log['selected_branch']})
            for branch_index,b in enumerate(log.get('branches',[log])):
                r=b.get('recovery',{})
                entry={'strategy':case.strategy,'seed':case.seed,'gamma':case.gamma,'delay_scale':case.delay_scale,
                    'decision_index':index,'decision_s':log['decision_s'],'branch_index':branch_index,
                    'holding_s':b.get('holding_s',0),'status':b['status'],'engine':b['engine'],
                    'wall_seconds':b['wall_seconds'],'mip_node_count':b.get('mip_node_count'),'mip_gap':b.get('mip_gap'),
                    **{k:r.get(k) for k in ['illegal_rounded_candidate','attempted','recovered','unrecoverable','lp_status','lp_iterations','lp_wall_seconds']}}
                if writer is None:writer=csv.DictWriter(stream,fieldnames=list(entry));writer.writeheader()
                writer.writerow(entry)
                if r.get('illegal_rounded_candidate'):
                    affected.append({**entry,'request_ids':';'.join(log.get('request_ids',[])),
                        'selected_branch':log.get('selected_branch',0),'recovery_evidence':b.get('recovery_evidence'),
                        'rejection_reason':r.get('rejection_reason'),
                        'rounded_max_violation':r['rounded_validation']['max_violation'],
                        'recovered_max_violation':r.get('final_validation',{}).get('max_violation')})
    stream.close()
    pd.DataFrame(affected).to_csv(target/'numerical_affected_batches.csv',index=False)
    pd.DataFrame(nonzero,columns=['strategy','seed','decision_s','holding_s','zero_status','selected_branch']).to_csv(target/'nonzero_hold_decisions.csv',index=False)

def core():
    target=out('core');f=pd.read_csv(target/'raw_runs.csv');assert len(f)==120
    assert set(f.seed)==set(SEEDS) and all(f.groupby('seed').size()==5)
    for _,g in f.groupby('seed'):
        assert g.input_ids_digest.nunique()==1 and g.common_shocks_digest.nunique()==1
    aggregate(f,['strategy']).to_csv(target/'aggregate.csv',index=False)
    comparisons=[('Greedy','Spatial_MILP'),('MPC','Spatial_MILP'),('Greedy','MPC'),('Greedy','E-Spatial'),('E-Spatial','E-Full'),('Spatial_MILP','E-Spatial')]
    raw=[];stats=[]
    for a,b in comparisons:
        x,y=paired(f,a,b);raw.append(x);stats.append(y)
    raw=pd.concat(raw);stats=pd.concat(stats)
    raw.to_csv(target/'paired_raw.csv',index=False);stats.to_csv(target/'paired_summary.csv',index=False)
    f.to_csv(ROOT/'FINAL_COMPARISON.csv',index=False)
    old=pd.read_csv(PREVIOUS/'results/E_nested_ablation/raw_runs.csv')
    before_after(f,old,target);numerical(f,target)
    equal_service=[]
    for seed,g in f.groupby('seed'):
        cases={r.strategy:Path(r.artifact_directory) for r in g.itertuples()}
        records={s:pd.read_csv(p/'requests.csv.gz').set_index('request_id') for s,p in cases.items()}
        common=set.intersection(*[set(r.index[r.status.eq('completed')]) for r in records.values()])
        for s,p in cases.items():
            ledger=pd.read_csv(p/'energy_ledger.csv.gz');r=records[s]
            attributable=ledger[ledger.request_id.notna()].groupby('request_id')[['energy_kwh','co2e_g']].sum().reindex(r.index,fill_value=0)
            completed=set(r.index[r.status.eq('completed')]);extra=completed-common
            utility=float(r.loc[r.status.eq('completed'),'historical_quality'].sum())
            valid=r.status.eq('completed')&~r.service_constraint_violation
            metric=g[g.strategy.eq(s)].iloc[0]
            equal_service.append({'seed':seed,'strategy':s,'common_completed_n':len(common),'completed_n':len(completed),
                'common_completed_carbon_g':attributable.reindex(list(common)).co2e_g.sum(),
                'noncommon_completed_carbon_g':attributable.reindex(list(extra)).co2e_g.sum(),
                'failed_attributable_carbon_g':attributable.reindex(list(set(r.index)-completed)).co2e_g.sum(),
                'delivered_quality_utility':utility,'valid_quality_utility':float(r.loc[valid,'historical_quality'].sum()),
                'incremental_carbon_per_completed_g':metric.incremental_co2e_g/len(completed),
                'total_carbon_per_quality_unit_g':metric.co2e_g/utility})
    pd.DataFrame(equal_service).to_csv(target/'equal_service_by_seed.csv',index=False)

    fig,ax=plt.subplots(1,4,figsize=(15,3.6))
    labels=list(f.strategy.unique())
    for axis,metric,label in zip(ax,['co2e_g','incremental_co2e_g','p95_response_s','service_violation_rate'],['Total CO2e (g)','Incremental CO2e (g)','P95 response (s)','Service violation rate']):
        groups=[f.loc[f.strategy.eq(s),metric] for s in labels]
        axis.bar(range(len(labels)),[g.mean() for g in groups],yerr=[g.std(ddof=1) for g in groups],capsize=3)
        axis.set_xticks(range(len(labels)));axis.set_xticklabels(labels,rotation=32,ha='right');axis.set_ylabel(label)
    figure(fig,target,'strategy_comparison')
    p=stats[stats.metric.eq('co2e_g')].iloc[:5];fig,ax=plt.subplots(figsize=(8,4))
    ax.errorbar(p['mean'],np.arange(len(p)),xerr=[p['mean']-p.ci95_lower,p.ci95_upper-p['mean']],fmt='o',capsize=4)
    ax.set_yticks(np.arange(len(p)));ax.set_yticklabels(p.baseline+' minus '+p.proposed);ax.axvline(0,color='grey',lw=1);ax.set_xlabel('Paired net reduction (g CO2e), 95% t interval')
    figure(fig,target,'paired_net_reduction')

def sensitivity(phase):
    target=out(phase);f=pd.read_csv(target/'raw_runs.csv');col='gamma' if phase=='F' else 'delay_scale'
    assert len(f)==(72 if phase=='F' else 72)
    aggregate(f,['strategy',col]).to_csv(target/'aggregate.csv',index=False)
    baseline=pd.read_csv(out('core')/'raw_runs.csv');base=baseline[baseline.strategy.eq('Greedy')&baseline.seed.isin(SEEDS[:6])]
    pairs=[];stats=[]
    if phase=='F':
        x,y=paired(f,'E-Spatial','E-Full',['gamma']);pairs.append(x);stats.append(y)
        for s in ['E-Spatial','E-Full']:
            x,y=paired(f,'Greedy',s,['gamma']);pairs.append(x);stats.append(y)
        risk=[]
        for (s,g),v in f.groupby(['strategy','gamma']):
            for budget in [.01,.005,.001,0.]:
                risk.append({'strategy':s,'gamma':g,'risk_budget':budget,'mean_rate':v.service_violation_rate.mean(),'max_seed_rate':v.service_violation_rate.max(),'all_six_seeds_qualify':bool((v.service_violation_rate<=budget+1e-14).all()),'n_independent_seeds':len(v)})
        pd.DataFrame(risk).to_csv(target/'risk_qualification.csv',index=False)
        old=pd.read_csv(PREVIOUS/'results/F_sla_risk/raw_runs.csv')
    else:
        for k,g in f.groupby(col):
            for s in g.strategy.unique():
                x,y=paired(pd.concat([base,g[g.strategy.eq(s)]]),'Greedy',s)
                x[col]=k;y[col]=k;pairs.append(x);stats.append(y)
        old=pd.concat([pd.read_csv(PREVIOUS/'results/H_controller_overhead/raw_runs.csv'),pd.read_csv(PREVIOUS/'results/H_controller_overhead/efficient_spatial_raw_runs.csv')])
        old=old[old.strategy.ne('Greedy')]
    pd.concat(pairs).to_csv(target/'paired_raw.csv',index=False);pd.concat(stats).to_csv(target/'paired_summary.csv',index=False)
    before_after(f,old,target);numerical(f,target)
    fig,ax=plt.subplots(1,2,figsize=(10,3.5))
    for s,g in f.groupby('strategy'):
        v=g.groupby(col).mean(numeric_only=True)
        ax[0].plot(v.index,v.co2e_g,marker='o',label=s);ax[1].plot(v.index,100*v.service_violation_rate,marker='o',label=s)
    for a in ax:a.set_xlabel(col);a.legend(fontsize=8)
    ax[0].set_ylabel('CO2e (g)');ax[1].set_ylabel('Service violation (%)');figure(fig,target,phase+'_sensitivity')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['core','F','H']);a=p.parse_args()
    if a.phase=='core':core()
    else:sensitivity(a.phase)
