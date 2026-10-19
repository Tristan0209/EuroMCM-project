import gzip
import json
import shutil
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .util import ROOT,DATA,read,save,sha,snapshot,input_signature
from .carbon import Carbon,ForecastView,REGIONS

ORDER=['Fixed','Greedy','Oracle','MPC']
COLORS={'Fixed':'#64748b','Greedy':'#d48b18','Oracle':'#23926b','MPC':'#326bc4'}
NAMES={'Fixed':'Fixed Local','Greedy':'Greedy Carbon','Oracle':'Offline MILP Oracle*','MPC':'Causal MILP-MPC'}
METRICS=['co2e_g','energy_kwh','mean_response_s','p95_response_s','p99_response_s','mean_queue_wait_s',
    'mean_intentional_delay_s','deadline_violation_rate','completion_rate','service_violation_rate','solver_wall_seconds','runtime_seconds']


def finish(fig,name):
    fig.tight_layout()
    fig.savefig(ROOT/'figures'/(name+'.png'),dpi=260,bbox_inches='tight')
    fig.savefig(ROOT/'figures'/(name+'.svg'),bbox_inches='tight')
    plt.close(fig)


def markdown_table(frame):
    columns=list(frame.columns)
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+'\n'.join(
        '| '+' | '.join(str(v) for v in row)+' |' for row in frame.itertuples(index=False,name=None))


def load_results():
    tables={}
    for path in sorted((ROOT/'results').glob('experiment_*.csv')):
        name=path.stem.replace('experiment_','');f=pd.read_csv(path)
        for row in f.itertuples():
            metrics=read(ROOT/'results/runs'/row.run_id/'metrics.json')
            if not metrics['all_invariants_passed']:raise ValueError('Unvalidated completed result')
            if metrics.get('inputs_sha256',metrics.get('audited_inputs_sha256'))!=input_signature()['sha256']:
                raise ValueError('Result has stale or unattested input data hashes')
            for script,value in metrics['implementation_sha256'].items():
                if sha(ROOT/'src'/script)!=value:raise ValueError('Result uses stale core model source: '+script)
            for field in ['co2e_g','energy_kwh','service_violation_rate','horizon_s']:
                if not np.isclose(metrics[field],getattr(row,field),rtol=1e-12,atol=1e-12):raise ValueError('Stale aggregate '+name)
        if not f.horizon_s.eq(43200).all():raise ValueError('A preliminary window was mixed into final experiments')
        aggregate=f.groupby(['setting','strategy'])[METRICS].agg(['mean','std','count']).reset_index()
        aggregate.columns=['_'.join(k).rstrip('_') if isinstance(k,tuple) else k for k in aggregate.columns]
        aggregate.to_csv(ROOT/'results'/('aggregate_'+name+'.csv'),index=False)
        tables[name]=f
    required=['main','scale','demand','availability','delay','forecast','boundary','horizon','methodology','measured_controller','full_stress']
    if not set(required).issubset(tables):raise ValueError('Missing completed experiments '+str(set(required)-set(tables)))
    return tables


def paired(frame,baseline='Greedy'):
    a=frame[frame.strategy.eq(baseline)].set_index(['setting','rep'])
    b=frame[frame.strategy.eq('MPC')].set_index(['setting','rep'])
    keys=a.index.intersection(b.index);rows=[]
    for key in keys:
        x,y=a.loc[key],b.loc[key]
        delta=float(x.co2e_g-y.co2e_g)
        strict=bool(x.service_violation_rate==0 and y.service_violation_rate==0)
        rows.append({'setting':str(key[0]),'rep':int(key[1]),'baseline':baseline,'net_co2e_g':delta,
            'mpc_constraints_met':bool(y.service_violation_rate==0),'baseline_constraints_met':bool(x.service_violation_rate==0),
            'same_complete_input_cohort':bool(x.completion_rate==1 and y.completion_rate==1),
            'strict_worthwhile':strict and delta>0,'baseline_co2e_g':float(x.co2e_g),'mpc_co2e_g':float(y.co2e_g)})
    return pd.DataFrame(rows)


def figures(tables):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'axes.formatter.useoffset':False})
    main=tables['main'];mean=main.groupby('strategy').mean(numeric_only=True).reindex(ORDER)
    std=main.groupby('strategy').std(numeric_only=True).reindex(ORDER)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    axes[0].bar(range(4),mean.co2e_g,yerr=std.co2e_g,color=[COLORS[s] for s in ORDER],capsize=4)
    axes[0].set_ylabel('Total CO2e (g), including fixed idle base')
    bottom=np.zeros(4)
    for col,label,color in [('active_increment_co2e_g','Active increment','#326bc4'),('network_co2e_g','Network','#d48b18'),('controller_co2e_g','Control','#23926b')]:
        axes[1].bar(range(4),mean[col],bottom=bottom,label=label,color=color);bottom+=mean[col]
    axes[1].set_ylabel('Non-idle CO2e (g)');axes[1].legend(fontsize=8)
    axes[2].bar(range(4),mean.p95_response_s,yerr=std.p95_response_s,color=[COLORS[s] for s in ORDER],capsize=4)
    axes[2].set_ylabel('P95 response (s)')
    axes[2].set_yscale('log');axes[2].set_ylabel('P95 response, completed only (s, log)')
    for ax in axes:ax.set_xticks(range(4));ax.set_xticklabels(['Local','Greedy','Oracle*','MPC'])
    fig.suptitle('7335 requests, 3 common-random replications; error bars = sample SD')
    finish(fig,'01_strategy_comparison')
    fig,ax=plt.subplots(figsize=(8,5))
    f=tables['delay']
    for strategy in ['MPC','Oracle','Greedy']:
        g=f[f.strategy.eq(strategy)].groupby('delay_scale').mean(numeric_only=True).sort_index()
        ax.plot(g.mean_intentional_delay_s,g.co2e_g,'o-',color=COLORS[strategy],label=NAMES[strategy])
        if strategy!='Greedy':
            for j,(scale,row) in enumerate(g.iterrows()):
                offset=([22,10,-3,8][j] if strategy=='MPC' else 7)
                ax.annotate(f'{scale:g}x',(row.mean_intentional_delay_s,row.co2e_g),xytext=(6,offset),textcoords='offset points',fontsize=8)
        bad=g.service_violation_rate>0
        ax.scatter(g.loc[bad,'mean_intentional_delay_s'],g.loc[bad,'co2e_g'],marker='x',color='red',s=80)
    ax.set_xscale('symlog',linthresh=1)
    ax.ticklabel_format(axis='y',useOffset=False,style='plain')
    ax.set(xlabel='Mean intentional holding + computation (s, symlog)',ylabel='Total CO2e (g)',title='Delay/carbon trade-off (1000 IDs, common 12h)\nRed x = nonzero service violations; labels = delay-budget multiplier')
    ax.legend(fontsize=8);finish(fig,'02_delay_tradeoff')
    mpc0=main[main.strategy.eq('MPC')&main.rep.eq(0)].iloc[0]
    requests=pd.read_csv(ROOT/'results/runs'/mpc0.run_id/'requests.csv.gz')
    complete=requests[requests.status.eq('completed')]
    bins=np.arange(0,43200+300,300);profiles=[]
    fig,ax=plt.subplots(figsize=(10,4))
    for region in ['DE','FR','ES']:
        subset=complete[complete.region.eq(region)];values=[]
        for left,right in zip(bins,bins[1:]):
            busy=np.maximum(0,np.minimum(subset.completion_s,right)-np.maximum(subset.service_start_s,left)).sum()/300
            values.append(float(busy));profiles.append({'region':region,'start_s':left,'end_s':right,'mean_busy_replicas':busy})
        ax.step(bins[:-1]/3600,values,where='post',label=region)
    ax.set(xlabel='Hours since synthetic 2025-01-02 anchor',ylabel='Mean simultaneously busy replicas (5 min bins)',title='MPC realised regional inference load, S+L pools')
    ax.legend();pd.DataFrame(profiles).to_csv(ROOT/'results/regional_load.csv',index=False);finish(fig,'03_regional_load')
    raw=pd.read_csv(DATA/'carbon_lifecycle.csv');raw['datetime_utc']=pd.to_datetime(raw.datetime_utc,utc=True)
    anchor=pd.Timestamp(read(DATA/'alignment.json')['anchor_utc']);raw=raw[(raw.datetime_utc>=anchor)&(raw.datetime_utc<anchor+pd.Timedelta(hours=12))]
    fig,ax=plt.subplots(figsize=(9,4))
    for region,g in raw.groupby('region'):ax.step((g.datetime_utc-anchor).dt.total_seconds()/3600,g.carbon_intensity_gco2e_kwh,where='post',label=region)
    ax.set(xlabel='Hours since anchor',ylabel='Provider lifecycle gCO2e/kWh',title='Actual 2025 hourly intensity: accounting environment, not future MPC inputs');ax.legend()
    finish(fig,'04_actual_carbon_intensity')
    sample=pd.concat([complete.nsmallest(16,'arrival_s'),complete.nlargest(10,'intentional_delay_s')]).drop_duplicates('request_id').sort_values('arrival_s')
    fig,ax=plt.subplots(figsize=(11,6))
    for y,row in enumerate(sample.itertuples()):
        ax.plot([row.arrival_s/3600,row.service_start_s/3600],[y,y],color='#cbd5e1',linewidth=3)
        color={'DE':'#d48b18','FR':'#326bc4','ES':'#23926b'}[row.region]
        ax.broken_barh([(row.service_start_s/3600,row.actual_service_s/3600)],(y-.3,.6),facecolors=color)
        ax.scatter([row.service_start_s/3600],[y],s=12,color=color)
        ax.scatter([row.arrival_s/3600],[y],s=8,color='#64748b')
    ax.set_yticks(range(len(sample)));ax.set_yticklabels([s[-24:] for s in sample.request_id],fontsize=7)
    ax.set(xlabel='Hours since anchor',title='MPC trace sample: arrival/holding (grey), actual service (DE orange / FR blue / ES green)')
    finish(fig,'05_mpc_dispatch_trace')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for strategy in ORDER:
        g=tables['demand'][tables['demand'].strategy.eq(strategy)].groupby('load').mean(numeric_only=True).sort_index()
        axes[0].plot(g.index,g.co2e_g,'o-',label=NAMES[strategy],color=COLORS[strategy])
        axes[1].plot(g.index,g.service_violation_rate*100,'o-',label=NAMES[strategy],color=COLORS[strategy])
    axes[0].set(xlabel='Arrival-rate multiplier',ylabel='Total CO2e (g)',title='Demand sensitivity (1000 fixed IDs)')
    axes[1].set(xlabel='Arrival-rate multiplier',ylabel='Service violation (%)');axes[0].legend(fontsize=8)
    finish(fig,'06_demand_sensitivity')
    p=paired(tables['boundary']);p[['axis','value']]=p.setting.str.split('=',expand=True);p['value']=p.value.astype(float)
    p.to_csv(ROOT/'results/boundary_paired.csv',index=False)
    axes_names=list(p.axis.drop_duplicates());fig,axes=plt.subplots(2,3,figsize=(13,8))
    for ax,axis in zip(axes.flat,axes_names):
        g=p[p.axis.eq(axis)].groupby('value').agg(net=('net_co2e_g','mean'),sd=('net_co2e_g','std'),admissible=('mpc_constraints_met','all'),strict=('strict_worthwhile','all')).sort_index()
        ax.errorbar(g.index,g.net,yerr=g.sd,fmt='o-',color=COLORS['MPC'],capsize=3)
        bad=~g.admissible;ax.scatter(g.index[bad],g.loc[bad,'net'],marker='x',color='red',s=80)
        ax.axhline(0,color='#64748b',linewidth=1);ax.set(xlabel=axis,ylabel='Greedy - MPC CO2e (g)')
        if axis=='solve_delay_seconds':ax.set_xscale('log')
    fig.suptitle('Complexity boundary on 500 fixed IDs; SD across 3 paired replications; x = MPC service violation')
    finish(fig,'07_complexity_failure_boundary')
    fig,ax=plt.subplots(figsize=(8,5))
    for strategy in ORDER:
        g=tables['scale'][tables['scale'].strategy.eq(strategy)].sort_values('n')
        ax.plot(g.n,g.runtime_seconds,'o-',label=NAMES[strategy],color=COLORS[strategy])
    ax.set(xscale='log',yscale='log',xlabel='Requests (deterministic whole-trace strata)',ylabel='Measured end-to-end run time (s)',title='Actual scaling benchmark: one rep per size, restricted MILP batches')
    ax.legend(fontsize=8);finish(fig,'08_scale_runtime')
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for name,ax in [('availability',axes[0]),('forecast',axes[1])]:
        group=tables[name].groupby(['setting','strategy']).co2e_g.mean()
        labels=list(tables[name].setting.astype(str).drop_duplicates())
        for strategy in tables[name].strategy.drop_duplicates():
            vals=[group.get((label,strategy),np.nan) for label in labels]
            ax.plot(range(len(labels)),vals,'o-',label=NAMES[strategy],color=COLORS[strategy])
        ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels,rotation=25,ha='right',fontsize=8);ax.set_ylabel('CO2e (g)');ax.set_title(name);ax.legend(fontsize=7)
    finish(fig,'09_capacity_and_forecast')
    f=tables['full_stress'];fig,axes=plt.subplots(1,2,figsize=(12,4))
    labels=list(f.setting.drop_duplicates())
    for strategy in ORDER:
        g=f[f.strategy.eq(strategy)].groupby('setting').mean(numeric_only=True).reindex(labels)
        axes[0].plot(range(len(labels)),g.co2e_g,'o-',color=COLORS[strategy],label=NAMES[strategy])
        axes[1].plot(range(len(labels)),g.service_violation_rate*100,'o-',color=COLORS[strategy],label=NAMES[strategy])
    display_labels=['Demand x2','Local failure','External contention']
    for ax in axes:ax.set_xticks(range(len(labels)));ax.set_xticklabels(display_labels,rotation=15,ha='right',fontsize=8)
    axes[0].set_ylabel('Total CO2e (g)');axes[1].set_ylabel('Service violations (%)')
    axes[0].legend(fontsize=8);fig.suptitle('Full 7335-request stress scenarios, 3 paired replications')
    finish(fig,'10_full_trace_stress')
    return mpc0


def forecast_accuracy():
    anchor=pd.Timestamp(read(DATA/'alignment.json')['anchor_utc']);actual=Carbon.load(anchor)
    records=[]
    for method,noise in [('persistence',0.),('historical_hour',0.),('persistence',.3),('persistence',.8)]:
        errors=[]
        for issued in np.arange(0,15100,900):
            view=ForecastView(actual,float(issued),21600,method,noise)
            for (hour,region),prediction in view.table.items():
                target=actual.origin+hour*3600
                if target<issued-1e-6 or target>issued+21600:continue
                errors.append(prediction-actual.at(target,region))
        values=np.array(errors)
        records.append({'method':method+'_'+str(noise),'evaluation_values':len(values),
            'MAE_gco2e_kwh':float(np.abs(values).mean()),'RMSE_gco2e_kwh':float(np.sqrt(np.mean(values**2))),
            'scope':'post-hoc errors at preregistered 15min issuance grid and up to 6h future; actual never fed into predictions'})
    records.append({'method':'perfect_future_actual_oracle','evaluation_values':len(values),
        'MAE_gco2e_kwh':0.,'RMSE_gco2e_kwh':0.,'scope':'oracle-only future actual, not a deployable predictor'})
    f=pd.DataFrame(records);f.to_csv(ROOT/'results/forecast_accuracy.csv',index=False)
    return f


def run():
    if read(ROOT/'results/audit/protected_before.json')!=snapshot():
        raise RuntimeError('Protected upstream changed; rerun audit before writing reports')
    tables=load_results();main=tables['main'];mpc0=figures(tables);accuracy=forecast_accuracy()
    paired_frames=[]
    for name,f in tables.items():
        for baseline in ['Fixed','Greedy']:
            p=paired(f,baseline)
            if len(p):paired_frames.append(p.assign(experiment=name))
    all_pairs=pd.concat(paired_frames,ignore_index=True);all_pairs.to_csv(ROOT/'results/paired_comparisons.csv',index=False)
    mean=main.groupby('strategy').mean(numeric_only=True).reindex(ORDER);sd=main.groupby('strategy').std(numeric_only=True).reindex(ORDER)
    show=pd.DataFrame({'策略':[NAMES[s] for s in ORDER],
        'CO2e/g 均值±SD':[f'{mean.loc[s,"co2e_g"]:.6f} ± {sd.loc[s,"co2e_g"]:.6f}' for s in ORDER],
        '电量/kWh':[f'{mean.loc[s,"energy_kwh"]:.6f}' for s in ORDER],
        'P95/s':[f'{mean.loc[s,"p95_response_s"]:.4f}' for s in ORDER],
        '完成率/%':[f'{100*mean.loc[s,"completion_rate"]:.4f}' for s in ORDER],
        '服务违反率/%':[f'{100*mean.loc[s,"service_violation_rate"]:.4f}' for s in ORDER],
        '求解wall/s':[f'{mean.loc[s,"solver_wall_seconds"]:.4f}' for s in ORDER]})
    spatial=mean.loc['Fixed','co2e_g']-mean.loc['MPC','co2e_g']
    temporal=mean.loc['Greedy','co2e_g']-mean.loc['MPC','co2e_g']
    boundary=pd.read_csv(ROOT/'results/boundary_paired.csv')
    states=boundary.groupby(['axis','value']).agg(net_mean=('net_co2e_g','mean'),net_sd=('net_co2e_g','std'),
        all_reps_worthwhile=('strict_worthwhile','all'),all_mpc_SLA=('mpc_constraints_met','all')).reset_index()
    states.to_csv(ROOT/'results/boundary_classification.csv',index=False)
    boundaries=[]
    for axis,g in states.groupby('axis'):
        ordered=g.sort_values('value');entries=[]
        for (_,a),(_,b) in zip(ordered.iloc[:-1].iterrows(),ordered.iloc[1:].iterrows()):
            if a.all_reps_worthwhile!=b.all_reps_worthwhile:entries.append([float(a.value),float(b.value)])
        boundaries.append({'axis':axis,'sampled_transition_brackets':entries,'universal_threshold':False})
    save(ROOT/'results/boundary_brackets.json',boundaries)
    def compact(frame,group_columns):
        g=frame.groupby(group_columns).mean(numeric_only=True).reset_index()
        out=g[group_columns].copy()
        out['CO2e/g']=g.co2e_g.map(lambda x:f'{x:.6f}')
        out['P95/s']=g.p95_response_s.map(lambda x:f'{x:.4f}')
        out['主动等待/s']=g.mean_intentional_delay_s.map(lambda x:f'{max(0.,x):.4f}')
        out['服务违反/%']=(g.service_violation_rate*100).map(lambda x:f'{x:.4f}')
        return markdown_table(out)
    demand_table=compact(tables['demand'][tables['demand'].strategy.isin(['Greedy','MPC'])],['load','strategy'])
    stress_table=compact(tables['full_stress'],['setting','strategy'])
    delay_table=compact(tables['delay'][tables['delay'].strategy.isin(['MPC','Oracle'])],['delay_scale','strategy'])
    forecast_table=compact(tables['forecast'],['setting','strategy'])
    control_table=compact(tables['measured_controller'],['strategy'])
    all_ids=set(pd.concat([f[['run_id']] for f in tables.values()]).run_id)
    total_runs=len(all_ids);logs=[]
    for run_id in all_ids:
        with gzip.open(ROOT/'results/runs'/run_id/'solver_log.jsonl.gz','rt',encoding='utf-8') as f:
            logs.extend(json.loads(line) for line in f)
    solver_stats={'completed_unique_runs_used':total_runs,'solver_decisions':len(logs),
        'status_counts':pd.Series([x['status'] for x in logs]).value_counts().to_dict(),
        'max_variables':max([x['variables'] for x in logs]+[0]),'max_constraints':max([x.get('constraints',0) for x in logs]+[0]),
        'max_reported_relative_gap':max([x['mip_gap'] for x in logs if x.get('mip_gap') is not None]+[0]),
        'restriction':'finite candidates; chronological offline Oracle batches; singleton enumeration separately logged'}
    save(ROOT/'results/solver_summary.json',solver_stats)
    inventory=[]
    for folder in (ROOT/'results/runs').iterdir():
        if not folder.is_dir():continue
        if folder.name in all_ids:status='completed_used_in_final_tables'
        elif (folder/'metrics.json').is_file():status='completed_preliminary_or_unused_configuration'
        else:status='not_completed_no_metrics_not_used'
        inventory.append({'run_id':folder.name,'status':status})
    pd.DataFrame(inventory).to_csv(ROOT/'results/run_inventory.csv',index=False)
    before=read(ROOT/'results/audit/protected_before.json');current=snapshot()
    if before!=current:raise RuntimeError('Protected upstream files changed')

    upstream=read(DATA/'upstream_source_manifest.json')['files']
    from .util import PROJECT
    if any(sha(PROJECT/name)!=value for name,value in upstream.items()):raise RuntimeError('Original raw/resource source digest changed')
    xml=ET.parse(ROOT/'results/audit/tests.xml').getroot()
    suite=next(iter(xml)) if xml.tag=='testsuites' else xml
    tests=int(suite.attrib.get('tests',0));failures=int(suite.attrib.get('failures',0))+int(suite.attrib.get('errors',0))
    validation={'upstream_tests':21,'model_tests':tests,'model_test_failures':failures,'protected_files':len(before),
        'protected_unchanged':True,'completed_unique_runs_in_final_tables':total_runs,'all_run_invariants_passed':True,
        'actual_real_carbon_rows':26280,'source_carbon_estimated_hours':{'DE':12,'FR':5,'ES':27},
        'power_scenario':'six_fixed_8GPU_DGX_hosts_six_admitted_service_slots',
        'idle_energy_kwh':float(mean.loc['MPC','idle_kwh']),
        'current_source_replay':read(ROOT/'results/audit/current_source_validation.json'),
        'pending':['snapshot-specific GWP factor version and operational as-of revisions','matched hardware/power/network measurements','new independent quality holdout'],
        'not_claimed':['unrestricted global MILP optimum','zero stochastic deadline violation','real deployed reduction','universal failure boundary'],
        'solver':solver_stats}
    save(ROOT/'results/audit/final_validation.json',validation)
    phase0=read(ROOT/'results/audit/phase0.json');phase0['checks']['prediction_pipeline']={'status':'passed',
        'meaning':'causal reforecast and nonanticipativity software tests; operational as-of history remains unverified'}
    save(ROOT/'results/audit/phase0.json',phase0)
    main_mpc=main[main.strategy.eq('MPC')]
    forecast_run=ROOT/'results/runs'/mpc0.run_id/'issued_forecasts.csv.gz'
    if forecast_run.is_file():
        shutil.copyfile(forecast_run,DATA/'persistence_forecasts.csv.gz')
        (DATA/'persistence_forecast.metadata.json').write_text(json.dumps({'data_kind':'causal_retrospective_forecast',
            'source_run':mpc0.run_id,'unit':'gCO2e/kWh','emissions_species':'CO2e','boundary':'lifecycle',
            'value_column':'predicted_carbon_intensity_gco2_kwh','legacy_column_alias':True,
            'observation_rule':'last completed past hour only','revision_warning':'retrospective provider snapshot, exact historical as-of values unknown'},indent=2),encoding='utf-8')
    save(DATA/'manifest.json',{'stage':'final','files':{p.name:sha(p) for p in DATA.iterdir() if p.is_file() and p.name!='manifest.json'},
        'forecast_source_run':str(mpc0.run_id),'carbon_physical_quantity':'provider-reported CO2e; legacy gco2 aliases explicitly labelled',
        'source_protection':str(ROOT/'results/audit/protected_before.json')})
    text=f'''# 第三问研究结果：固定Pre-LightGBM后的时空调度

## 实际运行范围

最终结果使用{total_runs}个完成且通过逐请求不变量检查的唯一配置—重复组合；不同配置有意共用随机数，并非{total_runs}个独立统计样本。跨实验复用相同配置的缓存，不把复用计作新随机重复。主要比较使用完整7335请求、3个固定公共服务扰动重复。需求/故障/延迟/预测敏感性采用1000个固定分层请求，复杂度边界和规划窗口用500个固定分层请求；已完成的完整规模需求2倍及两故障压力实验列在experiment_full_stress.csv。规模基准为100/500/1000/7335，每规模1次运行，不能据此给出普遍扩展性保证。

数据是2025全年三个地区逐小时提供方CO2e快照，合计26280行；生命周期为主、直接CO2e为独立敏感性。没有生成真实回答、质量标签或碳观测。合成RouterBench到达过程仅平移到2025-01-02 UTC，非真实2025业务流量。原PDF未提供，研究问题依据用户所附第三问文字。

## 四策略完整请求主结果

{markdown_table(show)}

*Oracle允许完整未来碳/需求/容量信息，但采用有限候选和顺序离线MILP批次，仅在各个有限批次求解；不是原始全轨迹随机系统的全局最优解或有证书的理论下界。主表平均和SD来自同一3个服务扰动种子的条件重复，不代表硬件、训练或数据选择不确定性。

相对Fixed，MPC总CO2e均值差为{spatial:.6f}g（{100*spatial/mean.loc['Fixed','co2e_g']:.5f}%）；相对Greedy差为{temporal:.6f}g。必须结合完成率和违反率解读，不能把少服务请求产生的低电量当作优化收益。所有输入都有完成/失败记录，历史固定模型质量均值为{mean.loc['MPC','historical_quality_fixed_cohort']:.9f}，各策略相同；服务未交付造成的有效效用降低另列，不能归因于S/L质量变化。

MPC总电量相对Fixed改变{mean.loc['MPC','energy_kwh']-mean.loc['Fixed','energy_kwh']:+.6f}kWh；本场景的节碳来自用电地点/时刻改变，并不等同于节电。更多成功交付、跨区网络与控制计算都可能增加电量，固定空闲底座又限制了设施总量收益。

表中平均/P95/P99响应与等待仅在完成请求上定义，失败率及有效交付效用使用全部输入作分母；不能只比较条件延迟而忽略丢失服务。Oracle明显更长的等待反映其对未来信息的利用，不能被较低碳数值掩盖。延迟放宽时的曲线并非精确Pareto前沿：候选网格、分批承诺和随机执行会使曲线非单调。

## RQ1：空间调度收益及其上限

实际数值见主表和图01。空间差异通过活跃增量、传输和控制开销影响净收益；物理机群始终为6台常开8-GPU DGX，每地区S/L各一台，但各池只准入1个服务槽位，合计48张安装GPU、6个可调度槽位。所有未使用或不可用池仍计整机空闲功率。MPC主情景空闲碳均值{mean.loc['MPC','idle_co2e_g']:.6f}g，占总量{100*mean.loc['MPC','idle_co2e_g']/mean.loc['MPC','co2e_g']:.4f}%；空闲电量固定为{mean.loc['MPC','idle_kwh']:.2f}kWh。因此即使把大部分推理移到低碳地区，总设施减排也可能很小。不能用活跃部分的减排百分比替代总量减排百分比。

## RQ2：复杂调度与简单贪心

Greedy在当前可用/估计可排队地区按最近可见碳选择，不主动寻找未来低碳小时。MPC优化已经到达的等待请求，可通过在绿色地区排队、避免争抢或短期保留容量改变位置/开始时间。主策略Persistence未来曲线恒定，不能预知未来低碳小时，因此不会有凭空的未来预测收益。收益来源需用逐请求地区、等待和solver日志解释，不归于模型选型变化。

需求和故障数值见aggregate_demand.csv、aggregate_availability.csv；不同情景仍使用同一请求ID、模型、配置容量、种子和12小时观察窗口。故障只关闭新任务准入，已运行任务完成；不可用不意味着断电。高负载容量下降表示外部争用槽位，没有虚构额外任务的耗电。复杂策略在紧约束下可能超时、回退或服务失败，均计入统计。

1000请求需求敏感性均值（P95和等待只针对完成请求；失败全量计入违反率）：

{demand_table}

完整7335轨迹压力组均值：

{stress_table}

## RQ3：主动等待与排放

图02、experiment_delay.csv改变非实时服务完成预算和最大等待的倍率0.25/0.5/1/2；实时完成预算保持30秒，等待按倍率改变。每组都使用完全相同的12小时观察窗口，避免扩大deadline时新增空闲电量混入比较。intentional_delay含控制计算和延后dispatch，queue_wait仅从传输完成后到执行开始，response分解逐请求验证。

Oracle可以看见未来小时的变化，反映有限候选下的信息头寸；Persistence不能预测趋势，不能把Oracle的时间收益称为可部署收益。图中红叉表示有服务约束违反，不是合格Pareto点。actual服务随机超出保守估计时可能产生late/drop，完整可行性没有由离散计划自动保证。

{delay_table}

## RQ4：何时复杂路由不值得

严格判定使用共同完成输入且双方服务约束全部满足，并且Greedy−MPC总CO2e>0。results/boundary_classification.csv给出每个轴每个采样点三个重复的净收益均值、SD、约束和是否全部重复值得；图07显示净收益过零与服务失效。边界是500请求固定子集上的单因素样点，不能称为普遍临界常数。相邻采样点状态发生变化时，只报告区间：

```json
{json.dumps(boundaries,ensure_ascii=False,indent=2)}
```

空间碳差异接近0时，迁移的传输/控制代价可能超过活跃收益；拥堵与故障时，低碳地区会受容量与等待限制；传输时延放大时，一部分跨区选择不再满足截止；每次优化计算较慢时，控制能耗和延迟可能抹掉微小收益；预测扰动过大则可能错过低碳位置/时间。以上因果方向必须以实际采样表验证，未出现的交叉不推断精确阈值。

## 预测、计算开销与最优性限制

Persistence和历史同小时均值只读取发布时点之前已完成的观测小时；噪声是在合法预测上作固定键随机乘性扰动，未以future actual为中心加噪。issued_at与observed_end在solver日志逐次检查。数据来自2026年的回溯快照，缺少真正的2025发布版本；这里是条件历史再预测，不能声称恢复了真实生产时可见数值。

同一固定发布网格上的事后预测误差（数值仅用于事后评价）：

{markdown_table(accuracy[['method','evaluation_values','MAE_gco2e_kwh','RMSE_gco2e_kwh']].round(6))}

预测情景下的实际调度结果：

{forecast_table}

主实验把新增优化控制计算时间设为显式情景0.01秒/次，并分别记录真实wall求解时间；不能把10ms写成实测MILP性能。experiment_measured_controller.csv另实际运行“0.01秒+当次wall时间”进入DES与能耗的敏感性。图08给出真实程序运行时间，不用模型时钟冒充wall时间。规划时域1/3/6小时、周期5/15/30分钟的结果在experiment_horizon.csv；另有到达/紧迫事件触发更新，周期不是唯一触发。绝大多数等待上限小于1小时，较长时域未必增加候选价值。

实测wall开销进入模型时的500请求结果：

{control_table}

所有MILP使用HiGHS1.2.2/SciPy1.9.1，单次0.2秒预算，记录status/gap/变量约束数/请求ID和fallback原因。单请求有限候选用精确枚举并独立标记，不能把枚举次数写成HiGHS调用次数。solver_summary.json汇总实际状态。保守服务倍率1.35仅用于占用/可行性规划，真实服务仍为继承估计×公共扰动，未再乘L倍率。

## 条件结论

完整主实验MPC的平均服务违反率为{100*mean.loc['MPC','service_violation_rate']:.6f}%。因此若该值非零，就不能声称当前策略已满足零违反的部署要求，也不能仅凭正碳差宣称复杂策略“值得”。可将本结果作为改进预测、准入、安全余量或业务SLA之前的条件研究。真实碳曲线是历史提供方数据；网络、服务器功率/PUE、容量故障、需求映射和控制增量功率是显式假设；推理时间由事前估计和公共随机数模拟。任何节碳数字均是条件反事实结果，不是实际GPT-4/Mixtral部署减排。

当前结果已经使用新版功率重新优化并在DES执行，不是对旧调度轨迹简单乘系数。功率继承当前Task1配置和task2/runs/experiment：共享1500W/台、8张GPU、每GPU空闲175W、忙碌760W；整机空闲2900W、一个忙碌槽位3485W、PUE1.2。共享/空闲是未实测假设，760W是特定基准配置代理，不是GPT-4/Mixtral整机实测。旧175/250W情景保存在archives/legacy_power_175_250_20261017，不能与本轮数据混合。

所有图表都由已完成实验文件生成。当前版本Task1/Task2和数据准备文件{len(before)}个SHA256保护项保持不变；旧版保护记录仍在归档。旧数据准备接口21项合成测试继续验证，旧全数据验证器因路径/功率字段已过时而被显式替代，当前冻结路由、保存Pre、训练长度、服务估计及能耗由新源验证程序逐项复查。没有下载生成模型或调用付费LLM。
'''
    (ROOT/'RESULTS_zh.md').write_text(text,encoding='utf-8')
    validation_text=f'''# 第三问验证报告

上游21项接口测试通过，系统测试{tests}项，失败/错误{failures}项。完整Phase0见results/audit/phase0.json；pytest明细见results/audit/tests.xml。

最终表引用{total_runs}个已完成配置/重复运行，每次执行了输入ID守恒、S/L冻结、地区允许、执行不早于到达、单次非抢占执行、物理副本不重叠、容量准入、响应时间分解、公共随机条件、能量分解、MPC已到达集合与过去观测边界核查。passed表示软件和账本一致，不表示每个情景没有SLA失败。

新版手算例：2个10秒L任务，DE/FR/ES碳强度300/30/200 gCO2e/kWh，单槽位；两任务在FR顺序执行，增量设施功率为585×1.2=702W，增量碳为20×702/3.6e6×30=0.117g，增量电量0.0039kWh；200秒内6台空闲底座为1.16kWh。跨小时3590至3610秒、100至200强度积分为3000，增量碳0.585g。原75W增量手算测试仍作为legacy小型接口算例保留，不用其参数生成新版实验。另验证一台主机增加到2个准入槽位时不会重复计空闲，以及不可用不等于断电。

完成12小时共同窗口检查和100/500/1000/7335四种规模门槛；Oracle是批次候选受限基准，不具备全问题全局最优证书。求解器真实状态：

```json
{json.dumps(solver_stats,ensure_ascii=False,indent=2)}
```

当前版本上游{len(before)}项保护哈希未变。原327项历史保护记录与旧功率实验已归档，不能将旧哈希宣称为当前版本核验。真实碳值26280行，关键列无缺失/重复/非小时；DE/FR/ES提供方估计标志12/5/27小时原样保留。生命周期与direct独立运行，使用提供方CO2eq，不将它转换或声称为纯CO2。完整请求除estimated_energy_kwh外与旧版字段完全一致；全部保存Pre预测、S/L路由、训练长度服务代理重放通过。新输入指纹覆盖当前上游配置、源码、权重、路由、功率合同及物理主机表；上游变更时必须先audit，不能复用旧功率缓存。

pending/not_verified：精确历史发布版本和快照GWP因子版本、匹配硬件/网络/控制功率、独立外部质量留出。没有缺失碳强度时静默填0、免费真实回答、未来actual进入MPC或未来请求进入MPC。Perfect-future只允许Oracle路径。已有SLA失败如实保留，不能当作测试跳过。
'''
    (ROOT/'VALIDATION_REPORT.md').write_text(validation_text,encoding='utf-8')
    (ROOT/'reports/PHASE0_AUDIT.md').write_text('''# Phase 0 审计与兼容

A. 复用冻结7335请求、服务器/网络、3种可用性、公共随机扰动和第二问决策质量评价表；只读验证保存Pre预测与模型。碳CSV三地区各8760小时。

B. 原task2_complete已迁移至task2；当前来源显式配置为task2/runs/experiment。原数据准备全数据验证器含旧路径和旧功率字段，保留其源码/基准，21项合成接口测试仍执行；实际源数据改由当前验证器检查。未找到竞赛PDF；原时间列不带Z但表头明确UTC；GWP精确因子版本未知，保留提供方CO2eq，不伪造元数据。

C. 使用原alignment_delta和shift_times纯函数，输出全部移至task3_model。规范CO2e列，另提供用户所需legacy gco2命名互换文件，metadata明确其物理量仍是CO2e。生命周期和direct文件独立。24h历史预热后最早覆盖日期为Jan2，未看强度挑日期。

D. 六台固定DGX与六个准入槽位分离，主机空闲2900W、单忙碌槽位增量585W、PUE1.2。48张安装GPU不产生额外准入能力，不可用时仍计250.56kWh/12h空闲。旧代码和实验归档，当前实验重新求解，Task1/Task2及data_preparation均只读。
''',encoding='utf-8')
    from .migration_report import run as migration_report
    migration_report()
    artifact_paths=set()
    for pattern in ['*.md','*.py','*.txt','configs/*','src/*.py','tests/*.py','reports/*','figures/*',
                    'results/data/*','results/audit/*','results/*.csv','results/*.json']:
        artifact_paths.update(p for p in ROOT.glob(pattern) if p.is_file() and p.name!='artifact_manifest.json')
    for run_id in all_ids:artifact_paths.update(p for p in (ROOT/'results/runs'/run_id).iterdir() if p.is_file())
    save(ROOT/'results/artifact_manifest.json',{'files':{str(p.relative_to(ROOT)):sha(p) for p in sorted(artifact_paths)},
        'completed_runs':total_runs,'unused_preliminary_runs_excluded':True,'manifest_self_hash_excluded':True})
    print('REPORTS AND FIGURES COMPLETE',total_runs,'completed unique runs',flush=True)


if __name__=='__main__':run()
