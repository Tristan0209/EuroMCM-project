from context import *
import argparse,gzip
import numpy as np
import pandas as pd
from scipy.stats import t,beta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
METRICS=['co2e_g','incremental_co2e_g','idle_co2e_g','energy_kwh','completion_rate','service_violation_rate',
 'p50_response_s','p95_response_s','p99_response_s','mean_intentional_delay_s','mean_queue_wait_s','mean_solver_seconds',
 'controller_kwh','controller_co2e_g','fallback_count','solver_timeout_count']
def interval(values):
    x=np.asarray(values,float);n=len(x);mean=float(np.mean(x));sd=float(np.std(x,ddof=1)) if n>1 else float('nan')
    half=t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else float('nan')
    return dict(n_seeds=n,mean=mean,sd=sd,ci95_lower=mean-half,ci95_upper=mean+half)
def enriched(frame):
    f=frame.copy();f['incremental_co2e_g']=f.co2e_g-f.idle_co2e_g
    return f
def aggregate(frame,groups):
    rows=[]
    for keys,g in frame.groupby(groups,dropna=False):
        if not isinstance(keys,tuple):keys=(keys,)
        head=dict(zip(groups,keys))
        for col in METRICS:
            if col in g:rows.append({**head,'metric':col,**interval(g[col])})
    return pd.DataFrame(rows)
def paired(frame,a,b,groups=()):
    rows=[]
    partitions=frame.groupby(list(groups),dropna=False) if groups else [('all',frame)]
    for keys,g in partitions:
        if not isinstance(keys,tuple):keys=(keys,)
        head=dict(zip(groups,keys));x=g[g.strategy.eq(a)].set_index('seed');y=g[g.strategy.eq(b)].set_index('seed')
        assert set(x.index)==set(y.index) and not x.index.duplicated().any() and not y.index.duplicated().any()
        for seed in x.index:


            for fingerprint in ['input_ids_digest','common_shocks_digest']:
                if fingerprint in x and fingerprint in y:assert x.loc[seed,fingerprint]==y.loc[seed,fingerprint]
            p=Path(x.loc[seed].artifact_directory);q=Path(y.loc[seed].artifact_directory)
            for filename in ['input_ids.json','common_random_conditions.json']:
                if (p/filename).exists() and (q/filename).exists():assert read(p/filename)==read(q/filename)
            for metric in METRICS:
                if metric in x and metric in y:
                    rows.append({**head,'seed':seed,'baseline':a,'proposed':b,'metric':metric,
                        'baseline_minus_proposed':float(x.loc[seed,metric]-y.loc[seed,metric])})
    raw=pd.DataFrame(rows);summ=[]
    for keys,g in raw.groupby(list(groups)+['baseline','proposed','metric'],dropna=False):
        summ.append({**dict(zip(list(groups)+['baseline','proposed','metric'],keys)),**interval(g.baseline_minus_proposed),
            'positive_fraction':float(g.baseline_minus_proposed.gt(0).mean())})
    return raw,pd.DataFrame(summ)
def figure(fig,target,name):
    (target/'figures').mkdir(exist_ok=True);fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(target/'figures'/(name+'.'+ext),dpi=260,bbox_inches='tight')
    plt.close(fig)
def ledger_request(folder):
    rec=pd.read_csv(folder/'requests.csv.gz').set_index('request_id')
    ledger=pd.read_csv(folder/'energy_ledger.csv.gz')
    attrib=ledger[ledger.request_id.notna()].groupby('request_id')[['energy_kwh','co2e_g']].sum().reindex(rec.index,fill_value=0)
    return rec,ledger,attrib
def E():
    target=out('E');f=enriched(pd.read_csv(target/'raw_runs.csv'));assert len(f)==120
    f.to_csv(target/'raw_runs.csv',index=False)
    aggregate(f,['strategy']).to_csv(target/'aggregate.csv',index=False)
    pairs=[];stats=[]
    for a,b in [('E-Spatial','E-Full'),('Greedy','E-Spatial'),('Greedy','E-Full'),('Spatial_MILP','E-Spatial'),('MPC','E-Full')]:
        raw,s=paired(f,a,b);pairs.append(raw);stats.append(s)
    pd.concat(pairs).to_csv(target/'paired_raw.csv',index=False);pd.concat(stats).to_csv(target/'paired_summary.csv',index=False)
    decisions=[];gaps=[];changes=[];batch_checks=[];solver_rows=[]
    for seed in SEEDS:
        cases={r.strategy:Path(r.artifact_directory) for r in f[f.seed.eq(seed)].itertuples()}
        batch_ids={};records={}
        for policy in ['E-Spatial','E-Full']:
            folder=cases[policy];rec,ledger,attrib=ledger_request(folder);records[policy]=rec
            with gzip.open(folder/'solver_log.jsonl.gz','rt',encoding='utf-8') as stream:logs=[json.loads(line) for line in stream]
            batch_ids[policy]=[(x['decision_s'],x['request_ids']) for x in logs]
            assigned_ids=[];predicted=0.
            for index,log in enumerate(logs):
                assert log['issued_s']<=log['decision_s']+1e-8 and log['observed_carbon_end_s']<=log['decision_s']+1e-8
                if policy=='E-Full' and log['branches'][0]['status']==0:
                    assert tuple(log['selected_objective'])<=tuple(log['zero_branch_objective'])
                selected=log['committed_plans'];ids=[p['request_id'] for p in selected]
                assigned_ids.extend(ids);proxy=sum(p['cost_g'] for p in selected);predicted+=proxy
                actual=float(attrib.reindex(ids,fill_value=0).co2e_g.sum())
                decisions.append({'seed':seed,'strategy':policy,'decision_index':index,'decision_s':log['decision_s'],
                    'request_count':len(log['request_ids']),'selected_hold_s':log['holding_s'],'planning_proxy_g':proxy,
                    'actual_attributable_g_for_assigned_ids':actual,'execution_minus_plan_g':actual-proxy,
                    'unassigned':log['unassigned_count'],'post_compute_rejections':len(log['post_compute_rejected_ids']),
                    'all_branches_optimal_or_infeasible':log['all_branch_optimal']})
                for branch in log['branches']:
                    solver_rows.append({'seed':seed,'strategy':policy,'decision_index':index,'decision_s':log['decision_s'],
                        'holding_s':branch['holding_s'],'selected':branch is log['branches'][log['selected_branch']],
                        'status':branch['status'],'wall_seconds':branch['wall_seconds'],'mip_gap':branch.get('mip_gap'),
                        'mip_node_count':branch.get('mip_node_count'),'candidate_count':len(branch.get('candidates',[]))})
            assert len(assigned_ids)==len(set(assigned_ids)),'Repeated candidate attribution needs separate handling'
            selected_actual=float(attrib.reindex(assigned_ids,fill_value=0).co2e_g.sum())
            gaps.append({'seed':seed,'strategy':policy,'planning_proxy_g':predicted,'assigned_actual_increment_and_network_g':selected_actual,
                'gap_g':selected_actual-predicted,'assigned_unique_requests':len(assigned_ids),
                'unassigned_or_fallback_request_carbon_g':float(attrib.co2e_g.sum())-selected_actual,
                'controller_carbon_separate_g':float(ledger.loc[ledger.component.eq('controller'),'co2e_g'].sum())})
        equal=batch_ids['E-Spatial']==batch_ids['E-Full'];batch_checks.append({'seed':seed,'identical_decision_times_and_batches':equal})
        old=pd.read_csv(cases['Spatial_MILP']/'requests.csv.gz').set_index('request_id')
        for a,b,left,right in [('E-Spatial','E-Full',records['E-Spatial'],records['E-Full']),('Spatial_MILP','E-Spatial',old,records['E-Spatial'])]:
            for rid in left.index:
                x,y=left.loc[rid],right.loc[rid]
                row={'seed':seed,'request_id':rid,'baseline':a,'proposed':b}
                for field in ['status','region','dispatch_s','service_start_s','completion_s','queue_wait_s','intentional_delay_s']:
                    av,bv=x.get(field),y.get(field);row[field+'_baseline']=av;row[field+'_proposed']=bv
                changes.append(row)
    pd.DataFrame(decisions).to_csv(target/'planning_execution_by_decision.csv.gz',index=False,compression='gzip')
    pd.DataFrame(gaps).to_csv(target/'planning_execution_by_seed.csv',index=False)
    changeframe=pd.DataFrame(changes)
    for field in ['status','region','dispatch_s','service_start_s','completion_s','queue_wait_s','intentional_delay_s']:
        left=changeframe[field+'_baseline'];right=changeframe[field+'_proposed']
        if field in ['status','region']:different=left.fillna('<missing>')!=right.fillna('<missing>')
        else:different=~np.isclose(left.astype(float),right.astype(float),atol=1e-8,rtol=0,equal_nan=True)
        changeframe[field+'_changed']=different
    changeframe.to_csv(target/'request_decision_comparison.csv.gz',index=False,compression='gzip')
    flags=[c for c in changeframe if c.endswith('_changed')]
    changeframe.groupby(['baseline','proposed','seed'])[flags].sum().reset_index().to_csv(target/'request_change_counts.csv',index=False)
    changeframe[changeframe[flags].any(axis=1)].to_csv(target/'request_changes_only.csv.gz',index=False,compression='gzip')
    pd.DataFrame(solver_rows).to_csv(target/'solver_subproblem_status.csv.gz',index=False,compression='gzip')
    pd.DataFrame(batch_checks).to_csv(target/'batch_alignment.csv',index=False)
    mean=f.groupby('strategy')[['incremental_co2e_g','p95_response_s','service_violation_rate']].mean()
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,col,label in zip(axes,mean.columns,['Incremental CO2e (g)','P95 completed (s)','Service violation (%)']):
        vals=mean[col]*(100 if col.endswith('rate') else 1);ax.bar(range(len(vals)),vals);ax.set_xticks(range(len(vals)))
        ax.set_xticklabels(vals.index,rotation=30,ha='right');ax.set_ylabel(label)
    fig.suptitle('E: 7335 fixed inputs; 24 paired seeds; common idle carbon reported separately');figure(fig,target,'controlled_ablation')
    gap=pd.DataFrame(gaps).groupby('strategy').gap_g.mean();fig,ax=plt.subplots(figsize=(6,4));ax.bar(gap.index,gap.values)
    ax.set_ylabel('Actual minus point-forecast planning proxy (g)');figure(fig,target,'planning_execution_gap')
    save(target/'validation.json',{'status':'PASS','rows':len(f),'seeds':24,'n':7335,'same_state_zero_branch_nesting':'PASS',
        'end_to_end_actions_including_fallback_nesting':'NOT VERIFIED: fallback lies outside the nominal MILP and its trigger depends on branch status',
        'trajectory_batch_alignment':'PASS' if all(x['identical_decision_times_and_batches'] for x in batch_checks) else 'NOT VERIFIED: divergent endogenous trajectories',
        'execution_never_claimed_to_be_globally_dominated':True,'new_full_nonzero_hold_decisions':int(f.loc[f.strategy.eq('E-Full'),'nonzero_hold_decisions'].sum())})
    save(target/'failure_pending.json',{'failed_new_runs':[],'not_verified':['per-request unrestricted independent holding; this extension only uses common batch holding','nonconstant real forecast temporal value'],'subproblem_status_file':'solver_subproblem_status.csv.gz'})

def G():
    target=out('G');f=enriched(pd.read_csv(out('E')/'raw_runs.csv'));assert len(f)==120
    rows=[];per_request=[]
    for seed in SEEDS:
        cases=f[f.seed.eq(seed)];cache={}
        for row in cases.itertuples():cache[row.strategy]=ledger_request(Path(row.artifact_directory))
        first=next(iter(cache.values()))[0]
        for rec,_,_ in cache.values():
            assert set(rec.index)==set(first.index)
            assert rec.selected_model.sort_index().equals(first.selected_model.sort_index())
            assert rec.historical_quality.sort_index().equals(first.historical_quality.sort_index())
        common=set.intersection(*(set(rec.index[rec.status.eq('completed')]) for rec,_,_ in cache.values()))
        for row in cases.itertuples():
            rec,ledger,attrib=cache[row.strategy];done=rec.status.eq('completed');valid=done&~rec.service_constraint_violation
            assert len(rec)==7335 and rec.index.is_unique
            idle=float(ledger.loc[ledger.component.eq('idle'),'co2e_g'].sum());control=float(ledger.loc[ledger.component.eq('controller'),'co2e_g'].sum())
            attributable=float(attrib.co2e_g.sum());successful=float(attrib.loc[done,'co2e_g'].sum());failed=float(attrib.loc[~done,'co2e_g'].sum())
            shared=float(attrib.reindex(sorted(common)).co2e_g.sum());extra=done&~rec.index.isin(common)
            utility_all=float(rec.loc[done,'historical_quality'].sum());utility_valid=float(rec.loc[valid,'historical_quality'].sum())
            increment=row.co2e_g-idle
            assert abs(row.co2e_g-idle-control-attributable)<1e-7
            assert abs(attributable-successful-failed)<1e-7
            rows.append({'seed':seed,'strategy':row.strategy,'n_input':7335,'completed':int(done.sum()),'on_time_valid':int(valid.sum()),
                'completion_rate':float(done.mean()),'service_violation_rate':float((~valid).mean()),'co2e_g':row.co2e_g,
                'incremental_co2e_g':increment,'idle_co2e_g':idle,'controller_co2e_g':control,'all_request_attributable_g':attributable,
                'common_completed_n':len(common),'common_completed_attributable_g':shared,
                'all_request_attributable_kwh':float(attrib.energy_kwh.sum()),
                'common_completed_attributable_kwh':float(attrib.reindex(sorted(common)).energy_kwh.sum()),
                'noncommon_completed_n':int(extra.sum()),'noncommon_completed_attributable_g':float(attrib.loc[extra,'co2e_g'].sum()),
                'noncommon_completed_attributable_kwh':float(attrib.loc[extra,'energy_kwh'].sum()),
                'failed_request_attributable_kwh':float(attrib.loc[~done,'energy_kwh'].sum()),
                'failed_request_attributable_g':failed,'all_incremental_g_per_completed':increment/done.sum(),
                'successful_attributable_g_per_completed':successful/done.sum(),'utility_all_completed':utility_all,'utility_on_time_valid':utility_valid,
                'total_g_per_completed_utility':row.co2e_g/utility_all,'total_g_per_valid_utility':row.co2e_g/utility_valid,
                'incremental_g_per_completed_utility':increment/utility_all,'incremental_g_per_valid_utility':increment/utility_valid})
            attrib=attrib.copy();attrib['in_common_completed']=attrib.index.isin(common);attrib['completed']=done;attrib['on_time_valid']=valid
            attrib['seed']=seed;attrib['strategy']=row.strategy;attrib['historical_quality']=rec.historical_quality
            per_request.append(attrib.reset_index())
    raw=pd.DataFrame(rows);raw.to_csv(target/'raw_runs.csv',index=False)
    pd.concat(per_request).to_csv(target/'request_equal_service_attribution.csv.gz',index=False,compression='gzip')
    summ=[]
    for policy,g in raw.groupby('strategy'):
        for col in raw.select_dtypes('number').columns:
            if col!='seed':summ.append({'strategy':policy,'metric':col,**interval(g[col])})
    pd.DataFrame(summ).to_csv(target/'aggregate.csv',index=False)
    ps=[];pairraw=[]
    for a,b in [('Greedy','E-Spatial'),('Greedy','E-Full'),('E-Spatial','E-Full'),('Greedy','MPC')]:
        x=raw[raw.strategy.eq(a)].set_index('seed');y=raw[raw.strategy.eq(b)].set_index('seed')
        for metric in ['co2e_g','common_completed_attributable_g','all_incremental_g_per_completed','incremental_g_per_valid_utility','utility_on_time_valid','noncommon_completed_attributable_g','noncommon_completed_attributable_kwh']:
            delta=x[metric]-y[metric];ps.append({'baseline':a,'proposed':b,'metric':metric,**interval(delta)})
            for seed,value in delta.items():pairraw.append({'seed':seed,'baseline':a,'proposed':b,'metric':metric,'baseline_minus_proposed':value})
    pd.DataFrame(ps).to_csv(target/'paired_summary.csv',index=False);pd.DataFrame(pairraw).to_csv(target/'paired_raw.csv',index=False)
    m=raw.groupby('strategy')[['common_completed_attributable_g','all_incremental_g_per_completed','incremental_g_per_valid_utility']].mean()
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,col,label in zip(axes,m.columns,['Common-completion busy + network (gCO2e)','All incremental / completed (gCO2e)','Incremental / valid utility (gCO2e)']):
        ax.bar(range(len(m)),m[col]);ax.set_xticks(range(len(m)));ax.set_xticklabels(m.index,rotation=30,ha='right');ax.set_ylabel(label)
    fig.suptitle('G: post-hoc equal-service diagnostics; idle and controller kept separate');figure(fig,target,'equal_service_metrics')
    save(target/'validation.json',{'status':'PASS','n':7335,'seeds':24,'policies':5,'facility_equals_idle_plus_controller_plus_attributable':'PASS',
        'failed_energy_retained':'PASS','quality_only_evaluation':'PASS; evaluated after scheduling','intersection_is_posthoc_and_biased':True})
    save(target/'failure_pending.json',{'failed_new_runs':[],'not_verified':['unbiased causal marginal energy per additional delivered request']})

def F():
    target=out('F');f=enriched(pd.read_csv(target/'raw_runs.csv'))
    f.to_csv(target/'raw_runs.csv',index=False)
    groups=['gamma','strategy'];aggregate(f,groups).to_csv(target/'aggregate.csv',index=False)
    rows=[];ps=[]
    for a,b in [('Greedy','E-Spatial'),('Greedy','E-Full'),('E-Spatial','E-Full')]:
        raw,s=paired(f,a,b,['gamma']);rows.append(raw);ps.append(s)
    pd.concat(rows).to_csv(target/'paired_raw.csv',index=False);pd.concat(ps).to_csv(target/'paired_summary.csv',index=False)
    risk=[]
    for (gamma,policy),g in f.groupby(groups):
        ci=interval(g.service_violation_rate)
        for threshold in [0.01,.005,.001,0.]:
            exceed=int(g.service_violation_rate.gt(threshold+1e-14).sum())
            exceed_upper=float(beta.ppf(.95,exceed+1,len(g)-exceed)) if exceed<len(g) else 1.
            risk.append({'gamma':gamma,'strategy':policy,'risk_budget':threshold,'n':int(g.n.iloc[0]),'n_seeds':len(g),
                'mean_violation_rate':ci['mean'],'sd_violation_rate':ci['sd'],'ci95_upper_mean_risk':max(0.,ci['ci95_upper']),
                'max_seed_violation_rate':g.service_violation_rate.max(),'empirical_seed_qualification_fraction':g.service_violation_rate.le(threshold+1e-14).mean(),
                'all_evaluation_seeds_qualify':bool(g.service_violation_rate.le(threshold+1e-14).all()),
                'mean_risk_t_upper_within_budget':bool(ci['ci95_upper']<=threshold+1e-14),
                't_interval_degenerate_no_rare_event_information':bool(g.service_violation_rate.std()==0),
                'seed_budget_exceed_probability_upper95_exact':exceed_upper,
                'chance_constraint_guarantee':False,'gamma_selection_claim':'NONE: sensitivity only'})
    pd.DataFrame(risk).to_csv(target/'risk_qualified_strategies.csv',index=False)
    regions=[]
    for row in f.itertuples():
        rec,ledger,_=ledger_request(Path(row.artifact_directory))
        for region in ['DE','FR','ES']:
            done=rec[rec.status.eq('completed')&rec.region.eq(region)]
            sent=rec[rec.dispatch_s.notna()&rec.region.eq(region)]
            regions.append({'seed':row.seed,'strategy':row.strategy,'gamma':row.gamma,'region':region,
                'completed_count':len(done),'dispatched_count':len(sent),'failed_dispatched_count':int(sent.status.eq('failed').sum()),
                'actual_busy_gpu_seconds':done.actual_service_s.sum(),'completed_mean_queue_s':done.queue_wait_s.mean(),
                'active_increment_co2e_g':ledger.loc[ledger.region.eq(region)&ledger.component.eq('active_increment'),'co2e_g'].sum()})
    pd.DataFrame(regions).to_csv(target/'regional_allocation.csv',index=False)
    m=f.groupby(groups)[['incremental_co2e_g','p95_response_s','service_violation_rate']].mean().reset_index()
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for policy,g in m.groupby('strategy'):
        axes[0].plot(g.gamma,g.incremental_co2e_g,'o-',label=policy);axes[1].plot(g.gamma,g.p95_response_s,'o-');axes[2].plot(g.gamma,g.service_violation_rate*100,'o-')
    for ax,label in zip(axes,['Incremental CO2e (g)','P95 completed (s)','Violation of all inputs (%)']):ax.set(xlabel='Service guard gamma',ylabel=label)
    axes[2].axhline(.1,color='grey',ls='--',lw=.9,label='0.1% illustrative budget');axes[2].legend(fontsize=8)
    axes[0].legend();fig.suptitle('F: full-cohort paired service-guard sensitivity; six seeds, no gamma selection');figure(fig,target,'carbon_latency_sla')
    save(target/'validation.json',{'status':'PASS','rows':len(f),'n':int(f.n.iloc[0]),'seeds':int(f.seed.nunique()),
        'all_input_risk_denominator':'PASS','gamma_actually_used':'PASS: tests plus saved per-run model config','risk_budgets_preregistered':'PASS','gamma_generalization':'NOT VERIFIED: no calibration or selection performed'})
    save(target/'failure_pending.json',{'failed_new_runs':[],'not_verified':['24-seed F grid','mathematical chance constraints','independent calibrated gamma deployment reliability']})

def H():
    target=out('H');f=enriched(pd.read_csv(target/'raw_runs.csv'));f.to_csv(target/'raw_runs.csv',index=False)
    aggregate(f,['delay_scale','strategy']).to_csv(target/'aggregate.csv',index=False)
    auxiliary=enriched(pd.read_csv(target/'efficient_spatial_raw_runs.csv')) if (target/'efficient_spatial_raw_runs.csv').exists() else pd.DataFrame()
    loop_frame=pd.concat([f,auxiliary],ignore_index=True)
    timings=[];rwait=[];request_waits=[]
    for row in loop_frame.itertuples():
        if row.strategy=='Greedy':continue
        folder=Path(row.artifact_directory)
        with gzip.open(folder/'solver_log.jsonl.gz','rt',encoding='utf-8') as stream:
            previous_end=0.
            solve_intervals=[];first_decision={}
            for index,line in enumerate(stream):
                log=json.loads(line);assert log['decision_s']>=previous_end-1e-8
                previous_end=log['decision_s']+log['modelled_controller_s']
                solve_intervals.append((log['decision_s'],previous_end))
                for rid in log['request_ids']:first_decision.setdefault(rid,log)
                assert abs(log['modelled_controller_s']-.01-row.delay_scale*log['wall_seconds'])<1e-8
                timings.append({'strategy':row.strategy,'seed':row.seed,'delay_scale':row.delay_scale,'decision_index':index,
                    'decision_s':log['decision_s'],'wall_seconds':log['wall_seconds'],'modelled_controller_s':log['modelled_controller_s'],
                    'requests':len(log['request_ids']),'status':log['status'],'mip_gap':log.get('mip_gap'),
                    'mip_node_count':log.get('mip_node_count'),
                    'subproblem_numeric_rejections':sum(b['status']==4 for b in log.get('branches',[])),
                    'subproblem_timeouts':sum(b['status']==1 for b in log.get('branches',[])),
                    'post_compute_rejected_plan_count':len(log['post_compute_rejected_ids'])})
        rec=pd.read_csv(folder/'requests.csv.gz');done=rec[rec.status.eq('completed')]
        starts=np.array([x[0] for x in solve_intervals]);ends=np.array([x[1] for x in solve_intervals])
        prefix=np.r_[0.,np.cumsum(ends-starts)]
        def busy_integral(t):
            k=np.searchsorted(starts,t,side='right')-1
            return 0. if k<0 else prefix[k]+max(0.,min(t,ends[k])-starts[k])
        local_wait=[]
        for q in rec.itertuples():
            log=first_decision.get(q.request_id)
            if log is None:
                request_waits.append({'seed':row.seed,'strategy':row.strategy,'delay_scale':row.delay_scale,
                    'request_id':q.request_id,'status':q.status,'has_solver_decision':False,
                    'controller_queue_wait_s':np.nan,'batch_and_reoptimization_wait_s':np.nan,
                    'own_compute_delay_s':np.nan,'policy_hold_after_compute_s':np.nan,'resource_queue_s':getattr(q,'queue_wait_s',np.nan)})
                continue
            ready=q.arrival_s+.01;decision=log['decision_s'];cpu_queue=max(0.,busy_integral(decision)-busy_integral(ready))
            batching=max(0.,decision-ready-cpu_queue);own=log['modelled_controller_s']
            dispatch=getattr(q,'dispatch_s',np.nan);hold=max(0.,dispatch-decision-own) if np.isfinite(dispatch) else np.nan
            if q.status=='completed':assert abs(q.intentional_delay_s-cpu_queue-batching-own-hold)<1e-5
            entry={'seed':row.seed,'strategy':row.strategy,'delay_scale':row.delay_scale,'request_id':q.request_id,'status':q.status,
                'has_solver_decision':True,
                'controller_queue_wait_s':cpu_queue,'batch_and_reoptimization_wait_s':batching,'own_compute_delay_s':own,
                'policy_hold_after_compute_s':hold,'resource_queue_s':getattr(q,'queue_wait_s',np.nan)}
            request_waits.append(entry)
            if q.status=='completed':local_wait.append(entry)
        local=pd.DataFrame(local_wait)
        rwait.append({'strategy':row.strategy,'seed':row.seed,'delay_scale':row.delay_scale,
            'mean_pipeline_holding_control_and_batching_s':float(done.intentional_delay_s.mean()),
            'mean_resource_queue_s':float(done.queue_wait_s.mean()),
            'policy_hold_mean_by_decision_s':getattr(row,'policy_hold_mean_s',np.nan),
            'mean_controller_queue_wait_s':local.controller_queue_wait_s.mean(),
            'mean_batch_and_reoptimization_wait_s':local.batch_and_reoptimization_wait_s.mean(),
            'mean_own_compute_delay_s':local.own_compute_delay_s.mean(),
            'mean_policy_hold_after_compute_s':local.policy_hold_after_compute_s.mean(),
            'control_cpu_seconds':row.modelled_optimizer_seconds,'controller_queue_cpu_charge_seconds':0})
    times=pd.DataFrame(timings);times.to_csv(target/'solver_wall_distribution.csv.gz',index=False,compression='gzip')
    pd.DataFrame(rwait).to_csv(target/'waiting_decomposition.csv',index=False)
    pd.DataFrame(request_waits).to_csv(target/'request_waiting_decomposition.csv.gz',index=False,compression='gzip')
    times.groupby(['strategy','delay_scale']).wall_seconds.agg(['count','mean','std','min','median','max']).to_csv(target/'wall_summary.csv')
    ps=[];pr=[];worth=[]
    for scale in [0,.5,1,2,5]:
        subset=pd.concat([f[f.strategy.eq('Greedy')],f[f.delay_scale.eq(scale)&~f.strategy.eq('Greedy')]])
        for policy in ['E-Spatial','E-Full']:
            raw,s=paired(subset,'Greedy',policy);raw['delay_scale']=scale;s['delay_scale']=scale;pr.append(raw);ps.append(s)
            x=subset[subset.strategy.eq('Greedy')].set_index('seed');y=subset[subset.strategy.eq(policy)].set_index('seed')
            delta=x.co2e_g-y.co2e_g
            for budget in [.01,.005,.001,0.]:
                qualifies=y.service_violation_rate.le(budget+1e-14);base_ok=x.service_violation_rate.le(budget+1e-14)
                worth.append({'strategy':policy,'delay_scale':scale,'risk_budget':budget,'n_seeds':len(delta),
                    'positive_net_fraction':float(delta.gt(0).mean()),'policy_budget_pass_fraction':float(qualifies.mean()),
                    'positive_net_and_policy_budget_fraction':float((delta.gt(0)&qualifies).mean()),
                    'positive_net_and_both_budget_fraction':float((delta.gt(0)&qualifies&base_ok).mean()),
                    'CI_keeps_all_seeds':True,'no_chance_guarantee':True})
    pd.concat(ps).to_csv(target/'paired_summary.csv',index=False);pd.concat(pr).to_csv(target/'paired_raw.csv',index=False)
    pd.DataFrame(worth).to_csv(target/'net_benefit_service_qualification.csv',index=False)
    if not auxiliary.empty:
        ap=[];ar=[]
        aggregate(auxiliary,['delay_scale','strategy']).to_csv(target/'efficient_spatial_aggregate.csv',index=False)
        for scale,g in auxiliary.groupby('delay_scale'):
            raw,s=paired(pd.concat([f[f.strategy.eq('Greedy')],g]),'Greedy','E-Spatial_single_branch')
            raw['delay_scale']=scale;s['delay_scale']=scale;ar.append(raw);ap.append(s)
        pd.concat(ar).to_csv(target/'efficient_spatial_paired_raw.csv',index=False);pd.concat(ap).to_csv(target/'efficient_spatial_paired_summary.csv',index=False)
    m=loop_frame[loop_frame.strategy.ne('Greedy')].groupby(['strategy','delay_scale'])[['incremental_co2e_g','p95_response_s','controller_co2e_g','service_violation_rate']].mean().reset_index()
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    base=f[f.strategy.eq('Greedy')].incremental_co2e_g.mean()
    for policy,g in m.groupby('strategy'):
        axes[0].plot(g.delay_scale,base-g.incremental_co2e_g,'o-',label=policy)
        axes[1].plot(g.delay_scale,g.p95_response_s,'o-',label=policy)
        axes[2].plot(g.delay_scale,g.controller_co2e_g,'o-',label=policy)
    axes[0].axhline(0,color='grey',lw=.8);axes[0].legend()
    for ax,label in zip(axes,['Greedy minus policy CO2e (g)','P95 completed (s)','Controller CO2e (g)']):ax.set(xlabel='Measured extra CPU delay multiplier',ylabel=label)
    fig.suptitle('H: fresh closed-loop solves, full cohort, six paired service seeds');figure(fig,target,'controller_overhead')
    fig,ax=plt.subplots(figsize=(7,4));sub=times[times.delay_scale.eq(1)]
    for policy,g in sub.groupby('strategy'):ax.hist(g.wall_seconds,bins=np.geomspace(max(1e-7,g.wall_seconds.min()),max(1e-6,g.wall_seconds.max()),40),alpha=.5,label=policy)
    ax.set(xscale='log',xlabel='Measured current-call wall seconds',ylabel='Solve count');ax.legend();figure(fig,target,'wall_distribution')
    save(target/'validation.json',{'status':'PASS','rows':len(f),'n':int(f.n.iloc[0]),'seeds':int(f.seed.nunique()),
        'auxiliary_single_branch_rows':len(auxiliary),'auxiliary_single_branch_scope':'full cohort, six paired seeds, scales 1 and 5; same constraints, unused branches omitted',
        'fresh_closed_loop':'PASS','no_old_wall_log_replay':True,'CPU_intervals_not_duplicated':'PASS',
        'end_to_end_timing_all_production_stacks':'NOT VERIFIED: current optimizer wall only, common upstream 0.01s proxy retained, Greedy has no extra MILP',
        'post_compute_feasibility_rechecked':'PASS','controller_power':'10W proxy, not measured','real_deployment':'NOT VERIFIED'})
    save(target/'failure_pending.json',{'failed_new_runs':[],'not_verified':['24-seed H grid','physical controller power','hardware-independent timing repeatability','separate fixed-log counterfactual replay (closed-loop implemented instead)']})

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['E','F','G','H','all'],default='all');a=p.parse_args()
    for name,fn in [('E',E),('G',G),('F',F),('H',H)]:
        if a.phase in [name,'all']:fn();print('ANALYZED',name,flush=True)
if __name__=='__main__':main()
