from pathlib import Path
import sys
import json
import math
import shutil
import time
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parents[1]
ROOT = HERE.parent
sys.path[:0] = [str(HERE),str(ROOT)]
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiment import SEEDS,POLICIES,scenarios
from task4_core.sources import BASE,RELEASE,read,sha
from task4_core.serialization import save
from task4_core.registry import reusable

ORDER = ['Baseline','Difficult','Prediction Error','Demand Growth']
TASKS = ['General','Math','Code']
COLORS = {'Greedy':'#5277a3','Spatial_MILP':'#d77936'}


def md_table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                    ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def link(path,label):
    return '['+label+'](<'+str(path.resolve()).replace('\\','/')+'>)'


def measures(frame):
    qm = np.where(frame.selected_model.eq('L'),frame.q_l,frame.q_s)
    done = frame.status.eq('completed')
    valid = done & ~frame.service_constraint_violation
    return {'requests':len(frame),'unique_base_ids':frame.base_request_id.nunique(),
            'unique_prompt_groups':frame['group'].nunique(),'mean_historical_quality':float(qm.mean()),
            'answer_pass_rate':float((qm>=.5).mean()),'effective_quality_per_input':float((valid*qm).mean()),
            'valid_answer_pass_rate':float((valid & (qm>=.5)).mean()),
            'S_count':int(frame.selected_model.eq('S').sum()),'L_count':int(frame.selected_model.eq('L').sum()),
            'L_share':float(frame.selected_model.eq('L').mean()),
            'p95_response_s':float(frame.loc[done,'response_s'].quantile(.95)) if done.any() else None,
            'service_violation_rate':float(frame.service_constraint_violation.mean()),
            'completion_rate':float(done.mean()),'valid_completion_rate':float(valid.mean()),
            'completed':int(done.sum()),'failed':int(frame.status.eq('failed').sum()),
            'censored':int(frame.status.eq('censored_at_H').sum()),
            'nominal_service_sum_s':float(frame.estimated_service_s.sum()),
            'model_selection_harm':float(frame.model_selection_harm.mean()),
            'delivery_incremental_harm':float(frame.delivery_incremental_harm.mean()),
            'end_to_end_harm':float(frame.end_to_end_harm.mean()),
            'mean_q_l_reference':float(frame.q_l.mean())}


def collect():
    completion = read(HERE/'runs/completion.json')
    assert completion['status']=='PASS' and len(completion['rows'])==24
    summary,groups,loads,routes,benchmarks = [],[],[],{},[]
    validation_inputs = []
    for row in completion['rows']:
        record = read(row['record'])
        assert reusable(record,record['config'])
        folder = Path(record['folder'])
        f = pd.read_csv(folder/'quality_harms.csv.gz')
        assert len(f)==(11002 if row['scenario']=='Demand Growth' else 7335)
        raw = read(folder/'metrics.json')
        route = pd.read_csv(folder/'routing.csv.gz',keep_default_na=False)
        routes[(row['scenario'],row['seed'],row['policy'])] = route
        ledger = pd.read_csv(folder/'energy_ledger.csv.gz')
        busy_s = float((ledger.loc[ledger.component.eq('active_increment'),'end_s']-
                        ledger.loc[ledger.component.eq('active_increment'),'start_s']).sum())
        width = float((f.arrival_s.max()-f.arrival_s.min()))
        item = {k:row[k] for k in ['scenario','seed','policy','world_id']}
        item.update(measures(f))
        for key in ['energy_kwh','co2e_g','idle_kwh','idle_co2e_g','active_increment_kwh','active_increment_co2e_g',
                    'network_kwh','network_co2e_g','controller_kwh','controller_co2e_g','unfinished_at_H',
                    'remaining_work_nominal_s','remaining_running_actual_s_evaluation_only','fallback_count','solver_calls',
                    'solver_timeout_count','highs_calls']:
            item[key] = raw[key]
        item['incremental_co2e_g'] = raw['co2e_g']-raw['idle_co2e_g']
        item['busy_slot_utilization_12h'] = busy_s/(6*43200)
        item['offered_nominal_slot_load_during_arrivals'] = f.estimated_service_s.sum()/(6*width)
        item['not_completed_count'] = raw['not_completed_count']
        item['source_record'] = row['record']
        summary.append(item)
        weights = f.service_class.value_counts(normalize=True).to_dict()

        for task,part in f.groupby('task_type'):
            g = {k:row[k] for k in ['scenario','seed','policy','world_id']}
            g.update(task_type=task,**measures(part))
            rates = part.groupby('service_class').service_constraint_violation.mean()
            g['class_standardized_violation_rate'] = float(sum(weights[c]*rates[c] for c in weights)) if set(weights)<=set(rates.index) else None
            groups.append(g)
        for (region,model),part in f[f.service_start_s.notna()].groupby(['region','selected_model']):
            duration = np.minimum(part.actual_service_s,43200-part.service_start_s)
            loads.append({**{k:row[k] for k in ['scenario','seed','policy','world_id']},'region':region,'model':model,
                          'started_requests':len(part),'window_busy_s':float(duration.sum()),
                          'slot_utilization_12h':float(duration.sum()/43200)})
        benchmark = read(folder/'benchmark.json')
        benchmarks.append({**{k:row[k] for k in ['scenario','seed','policy']},**benchmark,
                           'actual_execution_bytes':sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())})
        validation_inputs.append({'record':row['record'],'signature':record['signature'],'world_id':record['world_id']})
    results = pd.DataFrame(summary)
    fair = pd.DataFrame(groups)
    baseline_routes = {seed:routes[('Baseline',seed,'Greedy')].set_index('request_id') for seed in SEEDS}
    for idx,row in results.iterrows():
        route = routes[(row.scenario,row.seed,row.policy)].set_index('request_id')
        base = baseline_routes[row.seed]
        common = base.index.intersection(route.index)
        changes = int((route.loc[common].selected_model!=base.loc[common].selected_model).sum())
        results.loc[idx,'shared_original_route_changes'] = changes
        results.loc[idx,'changed_original_denominator'] = len(common)
        results.loc[idx,'route_change_comparison_applicable'] = row.scenario!='Difficult'
    for scenario in ORDER:
        for seed in SEEDS:
            a,b = [routes[(scenario,seed,p)] for p in POLICIES]
            pd.testing.assert_frame_equal(a,b,check_exact=True)
    results.to_csv(HERE/'Task4_Simple_Results.csv',index=False,encoding='utf-8-sig')
    fair.to_csv(HERE/'Task4_Simple_Fairness.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(loads).to_csv(HERE/'Task4_Simple_Server_Load.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(benchmarks).to_csv(HERE/'audit/runtime_measurements.csv',index=False)
    save(HERE/'audit/result_inputs.json',validation_inputs)
    return results,fair,pd.DataFrame(benchmarks)


def paired(results):
    rows = []
    for (scenario,seed),f in results.groupby(['scenario','seed']):
        a,b = [f[f.policy.eq(p)].iloc[0] for p in POLICIES]
        row = {'scenario':scenario,'seed':int(seed),'world_id':a.world_id,
               'scheduler_carbon_saving_g':a.co2e_g-b.co2e_g,
               'scheduler_facility_saving_pct':100*(a.co2e_g-b.co2e_g)/a.co2e_g,
               'scheduler_incremental_saving_pct':100*(a.incremental_co2e_g-b.incremental_co2e_g)/a.incremental_co2e_g,
               'P1_minus_P0_service_pp':100*(b.service_violation_rate-a.service_violation_rate),
               'P1_minus_P0_completion_pp':100*(b.completion_rate-a.completion_rate),
               'P1_minus_P0_p95_s':b.p95_response_s-a.p95_response_s,
               'P1_minus_P0_energy_kwh':b.energy_kwh-a.energy_kwh}
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(HERE/'Task4_Simple_Paired_Differences.csv',index=False,encoding='utf-8-sig')
    return frame


def rebound(results):
    rows = []
    for seed in SEEDS:
        s = results[results.seed.eq(seed)]
        a = s[s.scenario.eq('Baseline')&s.policy.eq('Greedy')].iloc[0]
        b = s[s.scenario.eq('Baseline')&s.policy.eq('Spatial_MILP')].iloc[0]
        g = s[s.scenario.eq('Demand Growth')&s.policy.eq('Spatial_MILP')].iloc[0]
        assert abs(a.idle_co2e_g-b.idle_co2e_g)<1e-8 and abs(a.idle_co2e_g-g.idle_co2e_g)<1e-8
        inc = b.co2e_g-b.idle_co2e_g
        critical = (a.co2e_g-b.co2e_g)/inc if inc>0 else None
        prediction = b.idle_co2e_g+(11002/7335)*inc
        rows.append({'seed':seed,'C_idle_g':b.idle_co2e_g,'N_baseline':7335,'N_growth':11002,
                     'growth_fraction_actual':11002/7335-1,'C_baseline_Greedy_g':a.co2e_g,'C_baseline_Spatial_g':b.co2e_g,
                     'C_growth_Spatial_g':g.co2e_g,'C_incremental_per_baseline_request_Spatial_g':inc/7335,
                     'theoretical_critical_growth_rate':critical,'growth_linear_prediction_g':prediction,
                     'growth_actual_minus_linear_g':g.co2e_g-prediction,
                     'growth_Spatial_minus_baseline_Greedy_g':g.co2e_g-a.co2e_g,
                     'growth_Spatial_failed':g.failed,'growth_Spatial_censored':g.censored,
                     'conclusion_scope':'linear conditional diagnostic, not observed behavioral rebound'})
    f = pd.DataFrame(rows)
    f.to_csv(HERE/'Task4_Simple_Rebound.csv',index=False,encoding='utf-8-sig')
    return f


def fair_pairs(fair):
    keys = ['scenario','seed','task_type']
    a,b = [fair[fair.policy.eq(p)] for p in POLICIES]
    joined = a.merge(b,on=keys,suffixes=('_Greedy','_Spatial'),validate='one_to_one')
    for field in ['service_violation_rate','completion_rate','valid_completion_rate','class_standardized_violation_rate']:
        joined[field+'_delta_pp'] = 100*(joined[field+'_Spatial']-joined[field+'_Greedy'])
    joined['p95_response_delta_s'] = joined.p95_response_s_Spatial-joined.p95_response_s_Greedy
    joined.to_csv(HERE/'Task4_Simple_Fairness_Paired.csv',index=False,encoding='utf-8-sig')
    return joined


def draw(results,fair_pair):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                        'axes.titleweight':'semibold','axes.grid':True,'grid.alpha':.18,'savefig.dpi':180})
    figdir = HERE/'figures'
    figdir.mkdir(exist_ok=True)
    ticks = ['Baseline','Difficulty\n40% hard','Prediction\nnoise 0.10','Demand\n+50%']
    def paired_plot(ax,field,scale=1):
        for pi,p in enumerate(POLICIES):
            for si,scenario in enumerate(ORDER):
                ys = results[results.policy.eq(p)&results.scenario.eq(scenario)][field].to_numpy()*scale
                x = si+(pi-.5)*.18
                ax.plot([x,x],[min(ys),max(ys)],color=COLORS[p],lw=1.2)
                ax.scatter([x]*len(ys),ys,c=COLORS[p],s=15,alpha=.65)
                ax.scatter([x],[ys.mean()],c=COLORS[p],s=65,marker='_',lw=2.4,label=p if si==0 else None)
        ax.set_xticks(range(4),ticks)
    fig,axes = plt.subplots(2,2,figsize=(11,7.4),constrained_layout=True)
    for ax,field,scale,title in zip(axes.flat,
            ['mean_historical_quality','valid_answer_pass_rate','p95_response_s','service_violation_rate'],
            [1,100,1,100],['Selected-answer historical quality','Valid delivered answers (% of all inputs)',
                         'P95 response of completed requests (s)','Service violation (% of all inputs)']):
        paired_plot(ax,field,scale)
        ax.set_title(title,fontsize=11)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Scenario robustness | 3 paired seeds; points = seeds, bars = observed range',fontsize=13)
    fig.savefig(figdir/'Figure1_Robustness.png')
    plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(11,4.6),constrained_layout=True)
    chosen = [('Baseline','Greedy'),('Baseline','Spatial_MILP'),('Demand Growth','Greedy'),('Demand Growth','Spatial_MILP')]
    xlabels = ['Base\nGreedy','Base\nSpatial','+50%\nGreedy','+50%\nSpatial']
    for i,(scenario,p) in enumerate(chosen):
        f = results[results.scenario.eq(scenario)&results.policy.eq(p)]
        idle = f.idle_co2e_g.mean()/1000
        inc = f.incremental_co2e_g.mean()/1000
        axes[0].bar(i,idle,color='#c9ced4',label='Fixed idle' if i==0 else None)
        axes[0].bar(i,inc,bottom=idle,color=COLORS[p],label='Operational increment' if i==0 else None)
        axes[0].text(i,idle+inc+.10,f'{idle+inc:.3f}',ha='center',fontsize=9)
        vals = f.incremental_co2e_g.to_numpy()
        axes[1].bar(i,vals.mean(),color=COLORS[p],width=.65)
        axes[1].scatter([i]*3,vals,c='#303030',s=13,zorder=3)
    axes[0].set_ylabel('Total facility CO2e (kg)')
    axes[0].set_ylim(bottom=0)
    axes[0].legend(loc='lower right',fontsize=8)
    axes[0].set_title('Full 12-hour facility boundary')
    axes[1].set_ylabel('CO2e above fixed idle (g)')
    axes[1].set_title('Operational increment shown separately')
    for ax in axes:ax.set_xticks(range(4),xlabels)
    fig.suptitle('Demand growth with the same six powered hosts',fontsize=13)
    fig.savefig(figdir/'Figure2_Carbon_Growth.png')
    plt.close(fig)
    fig,axes = plt.subplots(2,2,figsize=(10,7),constrained_layout=True)
    for ax,scenario in zip(axes.flat,ORDER):
        for j,task in enumerate(TASKS):
            part = fair_pair[fair_pair.scenario.eq(scenario)&fair_pair.task_type.eq(task)]
            y = part.service_violation_rate_delta_pp.to_numpy()
            ax.plot([j,j],[y.min(),y.max()],c='#5277a3')
            ax.scatter([j]*3,y,c='#5277a3',s=25)
            ax.scatter([j],[y.mean()],c='#d77936',s=90,marker='_',lw=2.5)
            ax.text(j,1.02,f'n={int(part.requests_Greedy.min())}–{int(part.requests_Greedy.max())}',
                    transform=ax.get_xaxis_transform(),ha='center',fontsize=8)
        ax.axhline(0,color='black',lw=.8)
        ax.set_xticks(range(3),TASKS)
        ax.set_title(scenario,pad=23)
        ax.set_ylabel('Spatial − Greedy violation (pp)')
    fig.suptitle('Within-task service comparison | lower favors Spatial; 3 paired seeds',fontsize=12)
    fig.savefig(figdir/'Figure3_Fairness.png')
    plt.close(fig)


EVIDENCE = [
    ['GPU实际功率','采用busy760W、idle175W；busy增量585W。来源合同注明ML.ENERGY特定Qwen/B200配置转用，不是Mixtral/GPT-4实测。',
     '需要目标模型、精度、batch、并发及利用率下的NVML与整机墙插同步测量。',
     '[ML.ENERGY研究](https://arxiv.org/abs/2505.06371)；冻结resource_accounting_contract.json'],
    ['物理host与idle','6台8GPU主机、每host1准入slot；共享1500W，全部安装GPU计idle；12h idle250.56kWh。',
     '需要整机idle、共享部件功率、可关机/休眠能力和容量策略日志；不能假定未用GPU不耗电。',
     '冻结physical_host_config.csv；[NVIDIA B200系统规格](https://docs.nvidia.com/dgx/dgxb200-user-guide/introduction-to-dgxb200.html)'],
    ['PUE与冷却','PUE=1.2为统一假设，IT只乘一次；network依既定独立边界不重复PUE。',
     '需要逐站点、逐时段设施电表与IT电表，覆盖冷却/UPS损耗；厂商上限不能替代PUE。',
     '[The Green Grid PUE定义](https://www.thegreengrid.org/node/372)'],
    ['DE/FR/ES逐小时碳','项目保留2025年每地8760小时消费侧生命周期CO2e，2026-02快照；本实验只用Jan2固定窗。提供者归属按格式推断。',
     '需要来源/许可链、因子版本、修订记录及当时可用的预测vintage；当前回溯actual不能当2025已发布预测。',
     'carbon_lifecycle.metadata.json；[Electricity Maps方法](https://www.electricitymaps.com/data/methodology)'],
    ['平均与边际排放','当前平均强度支持归属核算；路由改变负荷造成的真实系统减排是另一估计对象。',
     '需要适用时间尺度的边际排放、跨区电力流与可识别反事实；不能将平均核算差直接称电网因果减排。',
     '[NREL碳因子选择报告](https://www.nrel.gov/docs/fy24osti/86682.pdf)'],
    ['RouterBench与公共部门','历史benchmark可复现选模；任务混合/复制没有新答案，也没有真实公众群体属性。',
     '需要代表性公共服务请求、语言/可及性分组、双模型独立评分和隐私合规数据。',
     '[RouterBench原论文](https://arxiv.org/abs/2403.12031)'],
    ['DES与真实服务器','服务是train-only长度统计和固定参数代理，乘均值1 lognormal；到达/class/origin为构造环境。',
     '需要真实arrival/queue/start/finish、TTFT/TPOT、并发和超时日志，独立留出校准；n=3仅条件描述。',
     '冻结requests.csv、resource_accounting_contract.json和Task3论文'],
    ['系统边界与部署证据','本轮是条件仿真；controller10W、network电量为代理，未计训练/制造/硬件生命周期。',
     '需要完整边界的设施遥测和公平服务证据；新增生命周期主张须另有制造/寿命数据，本轮不补造。',
     '冻结能源合同、逐请求ledger和本轮验证报告']]


def report(results,fair,pairs,rebounds,fp,benchmarks):
    avg = results.groupby(['scenario','policy']).mean(numeric_only=True)
    def a(s,p,k):return float(avg.loc[(s,p),k])
    table1 = md_table(['情景','策略','N','L份额%','历史质量','答案达标%','有效交付达标%','P95秒','违反%','完成%','总CO2e g'],[
        [s,p,int(a(s,p,'requests')),f"{a(s,p,'L_share')*100:.2f}",f"{a(s,p,'mean_historical_quality'):.5f}",
         f"{a(s,p,'answer_pass_rate')*100:.3f}",f"{a(s,p,'valid_answer_pass_rate')*100:.3f}",
         f"{a(s,p,'p95_response_s'):.3f}",f"{a(s,p,'service_violation_rate')*100:.4f}",
         f"{a(s,p,'completion_rate')*100:.4f}",f"{a(s,p,'co2e_g'):.3f}"] for s in ORDER for p in POLICIES])
    diff = results[results.scenario.eq('Difficult')&results.policy.eq('Spatial_MILP')]
    err = results[results.scenario.eq('Prediction Error')&results.policy.eq('Spatial_MILP')]
    growth = results[results.scenario.eq('Demand Growth')&results.policy.eq('Spatial_MILP')]
    baseline = results[results.scenario.eq('Baseline')&results.policy.eq('Spatial_MILP')]
    meanpairs = pairs.groupby('scenario').mean(numeric_only=True)
    route_changes = err.shared_original_route_changes.astype(int).tolist()
    carbon_delta = rebounds.growth_Spatial_minus_baseline_Greedy_g
    fairrows = []
    for s in ORDER:
        for task in TASKS:
            f = fair[fair.scenario.eq(s)&fair.task_type.eq(task)]
            p = fp[fp.scenario.eq(s)&fp.task_type.eq(task)]
            g = f[f.policy.eq('Spatial_MILP')]
            fairrows.append([s,task,f"{int(g.requests.min())}–{int(g.requests.max())}",
                f"{int(g.unique_prompt_groups.min())}–{int(g.unique_prompt_groups.max())}",
                f'{g.mean_historical_quality.mean():.4f}',f'{g.answer_pass_rate.mean()*100:.2f}',
                f'{g.p95_response_s.mean():.3f}',f'{g.service_violation_rate.mean()*100:.3f}',
                f'{g.completion_rate.mean()*100:.3f}',f'{p.service_violation_rate_delta_pp.mean():+.4f}',
                f'[{p.service_violation_rate_delta_pp.min():+.4f}, {p.service_violation_rate_delta_pp.max():+.4f}]'])
    fairtable = md_table(['情景','类别','实例n/seed范围','独立prompt组范围','P1历史质量','P1达标%','P1 P95秒',
        'P1违反%','P1完成%','P1−P0违反pp均值','3seed观察范围pp'],fairrows)
    evidence_table = md_table(['证据环节','现有支持与假设','尚缺证据','来源'],EVIDENCE)
    failed = int(results.failed.sum())
    censored = int(results.censored.sum())
    total_idle = a('Baseline','Spatial_MILP','idle_co2e_g')
    save(HERE/'audit/analysis_summary.json',{'status':'ACTUAL_SIMPLE_SIMULATION_RESULTS','seeds':SEEDS,
        'new_strategy_executions':24,'all_C1_C8_executions':0,'prediction_changed_ids':route_changes,
        'baseline_P1_service_violation_mean':a('Baseline','Spatial_MILP','service_violation_rate'),
        'growth_P1_service_violation_mean':a('Demand Growth','Spatial_MILP','service_violation_rate'),
        'growth_minus_original_baseline_carbon_g':carbon_delta.tolist(),
        'theoretical_critical_growth_rate':rebounds.theoretical_critical_growth_rate.tolist(),
        'failed_request_instances_across_policy_executions':failed,'censored_across_policy_executions':censored,
        'not_real_deployment_measurements':True,'no_new_optimizer':True})
    theoretical = rebounds.theoretical_critical_growth_rate
    basepair = meanpairs.loc['Baseline']
    demandpair = meanpairs.loc['Demand Growth']
    increase_count = int((carbon_delta>0).sum())
    actual_growth_statement = (f'本次3个seed中有{increase_count}个增长后Spatial总碳高于原需求Greedy；均值多'
        f'{carbon_delta.mean():.3f}g（原Greedy总量的{carbon_delta.mean()/a("Baseline","Greedy","co2e_g")*100:.4f}%）。'
        '这是模型内很小的总设施增幅，不能当作超越功率/碳因子不确定性的现实显著增排。')
    intensity_base = a('Baseline','Spatial_MILP','active_increment_co2e_g')/a('Baseline','Spatial_MILP','active_increment_kwh')
    intensity_growth = a('Demand Growth','Spatial_MILP','active_increment_co2e_g')/a('Demand Growth','Spatial_MILP','active_increment_kwh')

    harms = fp.groupby(['scenario','task_type']).service_violation_rate_delta_pp.mean()
    adverse = [(s,t,float(v)) for (s,t),v in harms.items() if v>1e-10]
    fairness_statement = ('在这12个task×scenario平均配对差中，Spatial没有增加任何类别的平均服务违反率。'
        if not adverse else '存在Spatial服务违反率高于同类别Greedy的情景，必须保留而不能写全场景优越：'+str(adverse)+'。')
    task_growth = []
    for task in TASKS:
        bf = fair[fair.scenario.eq('Baseline')&fair.policy.eq('Spatial_MILP')&fair.task_type.eq(task)]
        gf = fair[fair.scenario.eq('Demand Growth')&fair.policy.eq('Spatial_MILP')&fair.task_type.eq(task)]
        task_growth.append((task,float((gf.service_violation_rate.mean()-bf.service_violation_rate.mean())*100)))
    worst_growth = max(task_growth,key=lambda x:x[1])
    positive_seed_differences = int((fp.service_violation_rate_delta_pp>1e-10).sum())
    positive_average_cases = [(s,t,float(v)) for (s,t),v in harms.items() if v>1e-10]
    text = f'''# 第四问：基于情景的鲁棒性与可持续性评估

日期：2026-10-18。**本轮四情景×3seed×2策略共24次实际执行已完成；原C1–C8全量、CVaR和LHS未执行。**全部新增文件位于simple_assessment，原代码/论文/权重/结果和42项合同、Pilot、失败日志保留。结果是固定数据与假设下的条件仿真，不是实际数据中心部署测量。

## 1. 第四问问题分析

第四问需要检验冻结Router在任务结构、预测和数量偏移下的可靠性，区分同需求调度减碳与需求增长后的总排放，并审查任务类别服务差异和真实环保所需证据。使用Scenario-Based Robustness and Sustainability Assessment，复用冻结Pre-LightGBM + finite-sample risk guard路由规则，以及第三问最终Spatial MILP + DES；Greedy是同一路由下的调度对照。没有新增优化模型、质量标签或硬件测量。

## 2. 简化情景分析方法

四情景：Baseline维持7335原请求和任务比例；Difficult固定7335、Code+Math按最大余数法取40%（Code:Math=86:1710）；Prediction Error只对S/L预测各加独立N(0,.10²)并clip，重复prompt group共用对应模型误差；Demand Growth保留原7335并追加floor(.5×7335)=3667个唯一实例，总N=11002。增长仍为6host、48GPU、6准入slots，原经验arrival支持和固定H=43200秒。实例映射和同组复制完整记录，重复历史题不增加独立质量证据。

固定种子{SEEDS}，每情景3个独立服务世界、同seed跨情景公共随机数；同world两策略共享路由、服务潜变量、资源/碳/网络。预测误差情景的服务基础时长按新动作切换，Code/Math/unknown保留L回退。实际q只在DES结束后连接两动作答案；不能按test q选重采样请求。既有Baseline检查没有找到同N、同seed、同world与版本的完整可复用记录，故本轮新执行；历史第三问结果仅作为旧证据引用，不算新样本。

所有表为每seed计算后取3seed均值，P95为seed内已完成请求分位数再平均。没有把三个seed所有请求当独立大样本，也没有声称统计显著性或95%概率保证。表中历史质量/答案达标率在全输入的**已选择历史答案**上计算，阈值q>=.5，不代表任务实际完成；有效交付达标率要求完成且满足deadline/wait且答案达标。服务违反率以所有输入为分母，失败/截尾也违反；完成但超时仍可能违反。完整区分这三个口径防止幸存者偏差。

{table1}

公式：I_i=1{{completed且完成/等待约束满足}}，v=Σ(1−I_i)/N，Q_hist=Σq_{{i,m_i}}/N，pass_hist=Σ1{{q_{{i,m_i}}≥.5}}/N；有效质量=ΣI_i q_{{i,m_i}}/N。完整设施总账C=C_idle+C_busy_increment+C_network+C_controller，每项分时乘实际CO2e强度，PUE一次；所有已发生的部分执行/网络/控制费用保留。H时非终结者censored并保留剩余名义工作。

![鲁棒性对比]({str((HERE/'figures/Figure1_Robustness.png').resolve()).replace(chr(92),'/')})

图1点为3个seed，横杠为均值、竖线为观察范围；不是置信区间。P95仅完成请求，所以同时展示全输入有效交付和服务违反。

## 3. 三个鲁棒性实验

### 3.1 请求难度变化

Baseline L份额{a('Baseline','Spatial_MILP','L_share')*100:.2f}%，Difficult为{a('Difficult','Spatial_MILP','L_share')*100:.2f}%；P1名义服务总量由{a('Baseline','Spatial_MILP','nominal_service_sum_s'):.2f}s变为{a('Difficult','Spatial_MILP','nominal_service_sum_s'):.2f}s。历史质量从{a('Baseline','Spatial_MILP','mean_historical_quality'):.5f}变为{a('Difficult','Spatial_MILP','mean_historical_quality'):.5f}，答案达标率由{a('Baseline','Spatial_MILP','answer_pass_rate')*100:.3f}%变为{a('Difficult','Spatial_MILP','answer_pass_rate')*100:.3f}%。

P1 P95由{a('Baseline','Spatial_MILP','p95_response_s'):.3f}s变为{a('Difficult','Spatial_MILP','p95_response_s'):.3f}s，服务违反由{a('Baseline','Spatial_MILP','service_violation_rate')*100:.4f}%变为{a('Difficult','Spatial_MILP','service_violation_rate')*100:.4f}%；P0对应违反率见总表。任务混合同时改变基础质量、class与service组成，不能把总体质量变化全归咎于Router，也不能先验认定L份额一定上升。Code/Math按guard机制固定L，其原本质量分布和负荷需分别看。

### 3.2 质量预测误差

三个seed分别有{route_changes}条原请求改变选模（分母7335）；两调度策略同world动作完全相同。P1 L份额变为{a('Prediction Error','Spatial_MILP','L_share')*100:.2f}%，历史质量{a('Prediction Error','Spatial_MILP','mean_historical_quality'):.5f}、历史达标{a('Prediction Error','Spatial_MILP','answer_pass_rate')*100:.3f}%；相较Baseline历史质量变化{a('Prediction Error','Spatial_MILP','mean_historical_quality')-a('Baseline','Spatial_MILP','mean_historical_quality'):+.5f}。P1服务违反{a('Prediction Error','Spatial_MILP','service_violation_rate')*100:.4f}%、P95 {a('Prediction Error','Spatial_MILP','p95_response_s'):.3f}s。

冻结Risk Guard的Code/Math/unknown强模型回退在噪声下仍生效，机械上隔离了这些类别的选模翻转；General仍可能变化。本实验没有取消guard的对照，也没有confirm-only重校准，因此不能量化“guard减少了多少损害”或宣称原有限样本保证自动覆盖扰动。支持的结论是既定回退机制保持、可观察的动作/质量/服务敏感性如表；不靠增加噪声后改真值制造损害。

### 3.3 使用量增长

原host数量不变，总N从7335增至11002（实得{(11002/7335-1)*100:.4f}%）。P1服务违反由{a('Baseline','Spatial_MILP','service_violation_rate')*100:.4f}%变为{a('Demand Growth','Spatial_MILP','service_violation_rate')*100:.4f}%，P95由{a('Baseline','Spatial_MILP','p95_response_s'):.3f}变为{a('Demand Growth','Spatial_MILP','p95_response_s'):.3f}s；完成率变为{a('Demand Growth','Spatial_MILP','completion_rate')*100:.4f}%。6slot全12h忙碌占比由{a('Baseline','Spatial_MILP','busy_slot_utilization_12h')*100:.3f}%变为{a('Demand Growth','Spatial_MILP','busy_slot_utilization_12h')*100:.3f}%；原arrival时段名义供给负荷比由{a('Baseline','Spatial_MILP','offered_nominal_slot_load_during_arrivals')*100:.3f}%变为{a('Demand Growth','Spatial_MILP','offered_nominal_slot_load_during_arrivals')*100:.3f}%。后者不是精确利用率，但可说明负荷增加；低12h平均值也可能掩盖池/时段拥塞。

相同增长需求下Spatial相对Greedy平均调度减碳{demandpair.scheduler_carbon_saving_g:.3f}g，服务违反配对差P1−P0={demandpair.P1_minus_P0_service_pp:+.4f}pp，P95配对差{demandpair.P1_minus_P0_p95_s:+.3f}s。优势是逐指标、逐配置的条件结果，不能自动推出整体服务/质量都优于Greedy。n=3只作效应量和种子范围描述，不将“恶化”直接写成统计显著。

Spatial较高P95可能同时涉及排队和所完成请求集合的差异；Greedy失败较多时，其完成者分位数也有选择效应。本轮没有将P95差单独归因于某一机制，不能凭P95更短就宣布Greedy端到端服务更好。

全24策略执行合计服务失败{failed}个实例、截尾{censored}个；这含同world两个策略的重复输入，不能叫独立请求总数。逐seed完成/失败/截尾和backlog均在Results中。失败的低碳不能当交付效率改善；有效交付质量必须并列。

## 4. 碳排放反弹效应

正常需求平均Greedy总碳{a('Baseline','Greedy','co2e_g'):.3f}g、Spatial总碳{a('Baseline','Spatial_MILP','co2e_g'):.3f}g；同需求调度节省{basepair.scheduler_carbon_saving_g:.3f}g，即总设施{basepair.scheduler_facility_saving_pct:.4f}%、扣idle增量{basepair.scheduler_incremental_saving_pct:.2f}%。增长后Spatial为{a('Demand Growth','Spatial_MILP','co2e_g'):.3f}g，与原需求Greedy比较的三个配对差为{[round(x,3) for x in carbon_delta]}g（正为增排）。这与同增长世界P0−P1是不同估计对象。

![增长与总碳]({str((HERE/'figures/Figure2_Carbon_Growth.png').resolve()).replace(chr(92),'/')})

在固定idle和近似线性增量条件下，C_P1(N)=C_idle+N c_inc，c_inc=[C_P1(N0)−C_idle]/N0。因此

\\[
r^* = \\frac{{C_{{P0}}(N_0)-C_{{P1}}(N_0)}}{{C_{{P1}}(N_0)-C_{{idle}}}}.
\\]

该理论临界增长率三个seed为{[round(x*100,2) for x in theoretical]}%，均值{theoretical.mean()*100:.2f}%；仅是当前基线成本斜率下的条件诊断。按N_growth/N0乘增量的线性预测与真实DES的误差分别为{[round(x,3) for x in rebounds.growth_actual_minus_linear_g]}g。不能把设施全部碳乘1.5，不能只用一个增长点拟合“真实阈值”，也不保证增量在其他r单调/线性。

{actual_growth_statement}约50%增长已经在这三个实际DES世界中跨越原Greedy基线，而线性诊断预测约99%才跨越，说明固定边际系数近似在增长点失准。由实际账本得到active增量的电量加权碳强度从{intensity_base:.3f}增至{intensity_growth:.3f}gCO2e/kWh，反映执行时空分配改变；因此不能保持基线每请求碳系数不变。本轮只有r=0与约.5两个点，没有定位实际非线性临界率或证明中间单调性。

这些结果只证明给定外生增长情景下的总账变化；没有需求弹性/真实行为数据，不证明效率改善引发了用户增长。未交付和截尾都计入风险，禁止凭服务失败宣称成功节能。

## 5. 分组公平性

{fairtable}

表以同task、同world的P1−P0为主要审计，避免把Math本就慢当不公平。{fairness_statement}36个task×scenario×seed配对中，服务违反差为正的有{positive_seed_differences}个，均保留在逐seed表。P1在增长相对自身Baseline的服务违反增量最大类别为{worst_growth[0]}（{worst_growth[1]:+.4f}pp）；这是受增长影响程度，不能直接当调度歧视。

逐seed差与共同class标准化差另存Fairness_Paired；缺class单元标NA不填0。类别原有难度和服务class组成仍可解释绝对差异，总体混合变化也不是歧视的直接证据。不同压力情景的同组差均公开，只有持续同方向、足够独立世界和代表性证据才可能支持“系统性不利”的强结论；本轮不作这种认证。

![组别服务差]({str((HERE/'figures/Figure3_Fairness.png').resolve()).replace(chr(92),'/')})

图3负值表示Spatial相对同类别Greedy违反更少，正值表示更多；全部点/观察范围保留。Code基线仅86个历史实例，重采样和增长会增加实例但不增加独立答案，因此不能用复制后的N宣称公平性证据更强。三个服务seed不足以证明不存在系统性损害；该结论仅涉及这三种任务代理，不涉及人口公平。

## 6. 环境证据和局限性

{evidence_table}

功率合同明确busy760W是特定公开benchmark转用，shared/idle/PUE与服务参数包含假设；NVIDIA整机14.3kW是规格上限，不能验证本实验的实际busy/idle值。未把ML.ENERGY其他模型测量伪称S/L目标模型测量。

本轮固定idle碳约{total_idle:.3f}g，占Baseline P1总碳{total_idle/a('Baseline','Spatial_MILP','co2e_g')*100:.3f}%。Spatial通过执行地区重新分配降低增量运行碳，已上电且不关机的全部host idle仍保留，因此扣idle减排比例明显大于设施总减排比例。第三问历史24seed正式结果为增量约49.7585%、设施总量约0.1781%；它是已有条件仿真，不替代本轮3seed结果。减碳也不等同节电，本轮P1−P0能量差保留在Paired_Differences。

公开方法只用于证据边界审查，不用当前网页反写2025碳、旧模型权重或原功率假设。所有公开来源于2026-10-18查阅；项目carbon metadata已标精确GWP版本和当时发布值未获独立确认。要证明真实环境效益，应对同等有效交付的目标工作负载测整机/设施能耗、PUE、可用预测和适当碳因子，保证服务质量及任务组不受不可接受损害，并保留增长后的总账。

## 7. 第四问整体结论

1. 困难占比40%时Spatial平均服务违反仍为{a('Difficult','Spatial_MILP','service_violation_rate')*100:.4f}%，但P95从{a('Baseline','Spatial_MILP','p95_response_s'):.3f}增至{a('Difficult','Spatial_MILP','p95_response_s'):.3f}s、历史质量从{a('Baseline','Spatial_MILP','mean_historical_quality'):.5f}降至{a('Difficult','Spatial_MILP','mean_historical_quality'):.5f}；可维持本设定下的较高完成水平，不能认定所有质量/延迟指标不变。
2. 预测噪声改变{route_changes}条选择，General仍会受影响；既定强模型回退保持，但未重新校准的Risk Guard不提供扰动下新保证。
3. 需求+50%在同一6host下增加负荷，P1服务违反变化{(a('Demand Growth','Spatial_MILP','service_violation_rate')-a('Baseline','Spatial_MILP','service_violation_rate'))*100:+.4f}pp；性能和配对优势按指标报告，不作n=3显著性或普遍鲁棒声明。
4. 增长后的Spatial与原Greedy总排放差均值{carbon_delta.mean():+.3f}g，理论反弹临界率约{theoretical.mean()*100:.2f}%只在线性假设下适用；没有识别效率诱导的真实行为增长。
5. 本轮36个task×scenario×seed配对中没有Spatial服务违反增量为正的例子；增长使Code的P1绝对违反率上升最多（{worst_growth[1]:+.4f}pp），且其样本很小。这没有显示本次Spatial相对Greedy系统性损害某类请求，但不证明不存在潜在损害或人口公平。
6. 环保结论有历史小时碳和完整账本支持，但缺目标模型实机能耗、真实PUE/服务/业务和边际排放反事实证据，仍属于条件仿真。

复现入口：simple_assessment/run_simple.py；固定protocol.json与逐次world/route/events/solver/ledger/quality工件保留。完整逐seed指标见 {link(HERE/'Task4_Simple_Results.csv','Results')}；验证报告见 {link(HERE/'Task4_Simple_Validation.md','Validation')}。旧复杂方案保留归档，本轮不再执行。
'''
    (HERE/'Task4_Simple_Report_ZH.md').write_text(text,encoding='utf-8')

    pd.DataFrame(EVIDENCE,columns=['Evidence','Current support and assumptions','Missing evidence','Sources']).to_csv(
        HERE/'Task4_Environmental_Evidence.csv',index=False,encoding='utf-8-sig')
    outline = f'''# Task 4 Paper Outline: Scenario-Based Robustness and Sustainability Assessment

Status: completed conditional simulations, four scenarios × three paired seeds × two scheduling policies (24 executions). No formal C1–C8, CVaR or LHS grid was run. Target length: 2–3 pages; this is an outline, not final prose.

## 1. Problem and scenario design (about 0.4 page)

Explain why distribution shift, prediction error, growth, unequal service and total-carbon rebound matter. Fix the Task 2 Pre-LightGBM/risk-guard policy and the final Task 3 Spatial MILP; compare against Greedy with identical routed requests and exogenous worlds. Introduce the four rows of the scenario table: original 7,335; 40% Code+Math; independent S/L Gaussian prediction noise sigma=0.10; original plus floor(0.5N0)=3,667 replay instances, N=11,002. Six hosts, H=43,200s throughout. Repeated historical tasks retain base/prompt-group identities and are not new quality samples.

Key equations: qtilde_im=clip(qhat_im+epsilon_gm,0,1), epsilon_gS independent of epsilon_gL with N(0,0.10²); rerun the frozen decision rule and switch the nominal service estimate to the selected action. Code/Math/unknown strong-model fallback remains fixed. Actual answer scores enter only after DES.

## 2. Robustness outcomes (about 0.6 page)

Place Figure 1 here. Show selected-answer quality and pass rate separately from valid delivered quality, completion and all-input service violation. Use v=N^(-1)Σ(1−I_i), with I_i requiring completion, deadline and waiting compliance. Quote seed-wise means and observed ranges; P95 is computed within completed requests for each seed. Do not pool replicated tasks or claim statistical significance with three seeds.

Facts to quote from actual results: Prediction Error changes {route_changes} original routing decisions; Spatial baseline violation={a('Baseline','Spatial_MILP','service_violation_rate')*100:.4f}%, difficulty={a('Difficult','Spatial_MILP','service_violation_rate')*100:.4f}%, prediction noise={a('Prediction Error','Spatial_MILP','service_violation_rate')*100:.4f}%, growth={a('Demand Growth','Spatial_MILP','service_violation_rate')*100:.4f}%. Interpret composition, routing harm and service-delivery harm separately. No guard ablation was run; preserved fallbacks do not establish a new finite-sample guarantee under perturbation.

## 3. Sustainability and conditional rebound (about 0.5 page)

Place Figure 2 here. C_total=C_idle+C_active_increment+C_network+C_controller; no host idle cancellation in absolute comparisons. With same hardware, C_P1(N)≈C_idle+N c_inc and r*=[C_P0(N0)−C_P1(N0)]/[C_P1(N0)−C_idle].

Quote normal scheduling saving {basepair.scheduler_carbon_saving_g:.3f}g, facility saving {basepair.scheduler_facility_saving_pct:.4f}%, incremental saving {basepair.scheduler_incremental_saving_pct:.2f}%. Growth Spatial versus original Greedy difference={carbon_delta.mean():+.3f}g (mean), theoretical critical growth≈{theoretical.mean()*100:.2f}%. Report actual DES versus linear approximation error; do not call the diagnostic an estimated behavioral elasticity or a verified nonlinear crossing. Same-demand savings and post-growth total changes answer different questions. Do not count rejection as successful energy efficiency.

## 4. Within-task service audit (about 0.4 page)

Place Figure 3 here, or use a compact table if page-limited. Compare Spatial−Greedy within General/Math/Code, same seed/world, with all-input service denominators. Include per-task historical quality/pass and completed P95. State unique prompt-group counts; Code starts with 86 historical instances. Class-standardized differences are supplementary; absent cells are NA. Slower Math alone is not evidence of unfairness. Do not claim demographic fairness or absence of systematic harm from these small conditional samples.

## 5. Evidence and limitations (about 0.4 page)

Use a compact Evidence and Limitations table. Distinguish measured public data from transferred assumptions: 760W busy GPU proxy, 175W idle and 1500W shared assumption, PUE1.2; historical DE/FR/ES hourly life-cycle CO2e snapshots; mean rather than marginal emissions; train-based service estimates rather than deployment logs; benchmark rather than representative public-sector inputs. Fixed idle explains why large incremental savings translate to small facility reductions. Real confirmation requires isolated target-model power/latency, facility/PUE telemetry, as-of forecasts, workload/quality representativeness and an appropriate carbon counterfactual.

Closing claim must be conditional: the frozen system was assessed under the three specified stresses, with observed trade-offs, not certified robust or environmentally beneficial in every deployment. Sources and numeric traces: Task4_Simple_Report_ZH.md, Task4_Simple_Results.csv and Task4_Simple_Validation.md. At most three figures were produced; no additional large experiments are proposed.
'''
    (HERE/'Task4_Paper_Outline_EN.md').write_text(outline,encoding='utf-8')


if __name__=='__main__':
    results,fair,benchmarks = collect()
    pairs = paired(results)
    rb = rebound(results)
    fp = fair_pairs(fair)
    draw(results,fp)
    report(results,fair,pairs,rb,fp,benchmarks)
    print('Analysis and three figures generated from 24 validated executions',flush=True)
