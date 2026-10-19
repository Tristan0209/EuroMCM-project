from context import *
import numpy as np
import pandas as pd

def table(f):
    f=f.copy()
    for c in f.select_dtypes('number'):f[c]=f[c].map(lambda x:'%.9g'%x)
    return '| '+' | '.join(map(str,f.columns))+' |\n| '+' | '.join(['---']*len(f.columns))+' |\n'+'\n'.join('| '+' | '.join(map(str,row))+' |' for row in f.itertuples(index=False,name=None))
def write(name,text):(ROOT/name).write_text(text,encoding='utf-8')
def safe_csv(path):
    try:return pd.read_csv(path)
    except pd.errors.EmptyDataError:return pd.DataFrame()
def main():
    gate=read(ROOT/'audit/final_gate.json');assert gate['status']=='PASS'
    f=pd.read_csv(out('core')/'raw_runs.csv');F=pd.read_csv(out('F')/'raw_runs.csv');H=pd.read_csv(out('H')/'raw_runs.csv')
    pairs=pd.read_csv(out('core')/'paired_summary.csv');hp=pd.read_csv(out('H')/'paired_summary.csv')
    risk=pd.read_csv(out('F')/'risk_qualification.csv')
    means=f.groupby('strategy').agg(total_g=('co2e_g','mean'),sd_g=('co2e_g','std'),increment_g=('incremental_co2e_g','mean'),
        energy_kwh=('energy_kwh','mean'),completion_pct=('completion_rate',lambda x:100*x.mean()),violation_pct=('service_violation_rate',lambda x:100*x.mean()),
        p50_s=('p50_response_s','mean'),p95_s=('p95_response_s','mean'),p99_s=('p99_response_s','mean'))
    def pair(a,b):return pairs[(pairs.baseline==a)&(pairs.proposed==b)&(pairs.metric=='co2e_g')].iloc[0]
    def sentence(p):return '%.9f g，95%%配对区间 [%.9f, %.9f]，正收益种子比例 %.2f%%'%(p['mean'],p.ci95_lower,p.ci95_upper,100*p.positive_fraction)
    net=pair('Greedy','Spatial_MILP');vsmpc=pair('MPC','Spatial_MILP');temporal=pair('E-Spatial','E-Full')
    before=pd.read_csv(out('core')/'before_after_metrics.csv');changes=pd.read_csv(out('core')/'before_after_request_counts.csv')
    oldstats=pd.read_csv(out('core')/'before_solver_status_counts.csv')
    numericcols=['recovery_illegal_rounded_candidate_count','recovery_attempted_count','recovery_recovered_count','recovery_unrecoverable_count','status4_count','fallback_count','native_timeout_count','recovery_lp_wall_seconds','actual_native_milp_calls']
    numeric=f.groupby('strategy')[numericcols].sum().reset_index();numeric.to_csv(ROOT/'numerical_recovery_totals.csv',index=False)
    unique_production=pd.concat([f,F,H]).drop_duplicates('run_id')
    numeric_all=unique_production.groupby('strategy')[numericcols].sum().reset_index()
    numeric_all.to_csv(ROOT/'numerical_recovery_unique_production_totals.csv',index=False)
    phase_numeric=[]
    for name,frame in [('core',f),('F',F),('H',H)]:
        g=frame.groupby('strategy')[numericcols].sum().reset_index();g.insert(0,'phase',name);phase_numeric.append(g)
    pd.concat(phase_numeric).to_csv(ROOT/'numerical_recovery_by_phase_references.csv',index=False)
    affected=safe_csv(out('core')/'numerical_affected_batches.csv');holds=safe_csv(out('core')/'nonzero_hold_decisions.csv')
    allinventory=[]
    selected=set(pd.read_csv(ROOT/'audit/artifact_invariants.csv').run_id)
    for folder in sorted((ROOT/'runs').iterdir()):
        if not folder.is_dir():continue
        row={'run_id':folder.name,'artifact_directory':str(folder),'selected_final_protocol':folder.name in selected,
            'status':'COMPLETED' if (folder/'metrics.json').exists() else 'FAILED_INCOMPLETE_RETAINED'}
        if (folder/'metrics.json').exists():row.update(read(folder/'metrics.json'))
        if (folder/'failure.json').exists():row['failure']=read(folder/'failure.json')['exception']
        allinventory.append(row)
    inventory=pd.DataFrame(allinventory);inventory.to_csv(ROOT/'run_inventory.csv',index=False)
    failed_attempts=int(inventory.status.eq('FAILED_INCOMPLETE_RETAINED').sum())
    pilot_note=('首轮pilot出现统计字段holding_s缺省的KeyError，日志及失败目录保留；汇总适配后重跑，未改模型，未纳入正式样本。' if failed_attempts else '本次独立执行没有未完成的软件运行；开发阶段pilot适配记录保留在原交付版本。')
    flags=[]
    def check(name,status,evidence):flags.append({'item':name,'status':status,'evidence':evidence})
    check('Existing 70 and new 13 tests','PASS','audit/tests.xml; 83 passed; initial failed test attempt retained')
    check('All original versions and inputs protected','PASS',str(gate['old_files_unchanged'])+' protected file hashes unchanged')
    check('A2 DES physical semantics unchanged','PASS','v2/des.py byte-identical to A2; same-time capacity tests pass')
    check('Full rounded/recovered model feasibility','PASS','original absolute threshold 1e-5; LP 1e-7; reject on any failure')
    check('Saved counterexample reproduction and safe rejection','PASS','LP infeasible at fixed integers, independently derived 0.0037348245s contradiction, rejected; not reported as successful recovery')
    check('Every raw numerical incumbent recoverable','FAIL','genuinely infeasible rounded assignments remain; counts in numerical_recovery_totals.csv; no such candidate executed')
    check('Core 7335 x 24 x 5 new executions','PASS','120 new core cases, paired IDs and common shocks')
    check('Deterministic same-environment repeats','PASS',gate['repeat_check'])
    check('Request/capacity/shock/energy/carbon/causality audits','PASS',str(gate['audited_unique_runs'])+' unique selected runs; audit/artifact_invariants.csv')
    check('Affected F/H reruns','PASS','F 72 references and H 72 references; first six seeds; shared configurations reused without increasing sample size')
    check('No failed execution hidden','PASS',f'run_inventory.csv includes all attempts, with {failed_attempts} failed/incomplete attempts preserved and excluded from independent-seed counts')
    check('Pure temporal carbon benefit independently established','NOT VERIFIED','persistence nominal future is flat; nonzero hold can rescue rejected zero branch; no claim of temporal dominance')
    check('All requests satisfy realized hard SLA','FAIL','nonzero observed violations retained; nominal feasibility is not a stochastic guarantee')
    check('Strict nominal-to-actual queue equivalence','NOT VERIFIED','nominal reservations/FCFS uncertainty and existing in-flight reservation approximation retained')
    check('Native fixed work/thread/random-seed enforcement','NOT VERIFIED','SciPy1.9.1/HiGHS1.2.2 interface limitation; ample budget and empirical repeats instead')
    check('Exact measured-H wall/trajectory reproduction','NOT VERIFIED','CPU duration depends on local platform/load; H is fresh serial closed-loop simulation')
    check('Full fresh one-command reproduction in another directory','NOT VERIFIED','individual pipeline stages actually executed; --verify checked at delivery; full reproduction entry provided but not redundantly rerun end-to-end')
    check('F/H 24 independent seeds','NOT VERIFIED','six-seed original scope explicitly retained')
    check('Physical DGX/service/network measurements','NOT VERIFIED','frozen DGX proxy; not calibrated operational hardware')
    check('Operational as-of forecast / real deployment savings','NOT VERIFIED','2025 retrospective actual snapshots and causal persistence baseline, synthetic arrivals')
    check('Unrestricted global Oracle optimum','NOT VERIFIED','not claimed; finite candidate/batch strategy retained, not rerun here')
    validation=pd.DataFrame(flags);validation.to_csv(ROOT/'validation_checklist.csv',index=False)
    write('FINAL_VALIDATION_REPORT.md','# 最终验证报告\n\n'+table(validation)+'\n\n软件准入及账本门槛 PASS。上述 FAIL 指“所有数值候选均可恢复”和“所有实际请求均满足硬 SLA”两项事实命题不成立，不能改成通过。它们不表示非法候选已执行。最终冻结仅覆盖正确执行、显式拒绝与失败统计完整的条件仿真软件，不认证零风险 SLA 或现实减排。\n')
    maxdelta=before.groupby('metric').new_minus_old.agg(lambda x:float(abs(x).max())).reset_index(name='max_absolute_new_old_difference')
    maxdelta.to_csv(out('core')/'max_before_after_difference.csv',index=False)
    write('NUMERICAL_FIX_REPORT.md',f'''# MILP 数值解恢复报告

## 路径审计

旧 queue_milp.py 和原 Spatial 均对 y/z 取整但保留 u，随后检查矩阵；原 Full 的 candidate-time MILP 全为整数变量。新 Spatial 与 E 共用 final_release/queue_milp.py，原 Full 使用独立 v2/milp.py。所有完整向量还检查变量上下界，不能仅检查选中的任务。单请求枚举没有此整数/连续取整路径，保持原实现。

## 已有反例与不能恢复的原因

原 rejected_3621.npz 的原始线性残差约 7.78e-13，整数偏差 5.3448414e-7，取整后违反 0.0037347603 秒。固定整数后 LP 返回 infeasible，而非成功恢复。独立约束证据：u10>=6987.347812924047，u1-u10>=0.370336900494294，故 u1>=6987.718149824541；但 u1<=6987.714415。差 0.003734824541 秒。只把未选 u 置零无法消除该矛盾。更改二元选择或放宽容差会改变问题，本轮不采用。

## 恢复协议

先检查整数偏差与完整向量。仅在必要时固定所有整数变量，用剩余原 wall budget 重求连续 LP；不改变任何分配/拒绝。连续目标只作正比例缩放，LP 原始/对偶容差1e-7；原完整模型复检容差保持1e-5。完整恢复后追加名义容量、网络允许、时间和队列非重叠检查。不可恢复返回 status4，进入原显式 fallback。保留失败、超时、未选 E 分支和证据矩阵。纯整数 Full 无连续自由度，不伪造修复。

## 完整核心重跑统计（每策略24种子；次数含未选分支）

{table(numeric)}

核心+F/H去重后的 {len(unique_production)} 次正式执行统计如下；重复使用的零开销/默认gamma配置只计一次。按阶段引用的计数另列 numerical_recovery_by_phase_references.csv，不可把其总和当唯一调用数。

{table(numeric_all)}

修复前日志的真实统计：

{table(oldstats.groupby('strategy')[['status4','timeout','numeric_round_rejections']].sum().reset_index())}

成功恢复数若为零，表示该实际样本中遇到的固定整数均不可行，不能写成恢复成功。新增可恢复合成例、LP超时、LP假成功、无剩余预算、纯整数无自由度、不可恢复原反例等测试全部通过。这是软件恢复能力的证据，不是实际数据已发生可恢复事件的证据。

受影响批次见 results/core/numerical_affected_batches.csv；完整原矩阵、边界、原解与恢复向量见 audit/recovery_candidates。逐请求改动见 before_after_changed_requests.csv.gz，逐事件类别对照见 before_after_event_counts.csv，按事件/请求ID/出现次序对齐的完整事件变化见 before_after_changed_events.csv.gz。不同调用可复用相同证据哈希，但调用次数分别统计；不能把调用或请求数当独立统计样本。actual_native_milp_calls按实际子问题计数并排除singleton枚举；旧账本highs_calls按顶层引擎标签计数，二者不可混用。

各阶段 numerical_affected_request_outcomes.csv.gz 逐个列出非法候选所在批次请求的新旧地区、派发、开始、完成、fallback及失败原因，并区分该分支是否实际被选择。未选分支的数值失败也保留，不能把其全部请求称为实际被迫回退。

## 时间与结果变化

完整子问题 wall 包括恢复与验证，LP wall/iterations 单列，原预算不增加。理想控制核心部分采用独立 worker 并行，因此其测得 wall 只作描述，不用于声称前后速度因果变化；H 独占串行重算其实际控制耗时。新旧计时不同是预期现象。下表是核心各指标最大绝对差：

{table(maxdelta)}

核心新增/更改终态请求总计 {int(changes.changed_requests.sum())} 个策略-种子-请求记录（非独立样本）。即使部分数值相同，也属于新恢复协议的实际重跑，不称作旧版本的精确复现。F/H另有各自完整前后差异CSV。{pilot_note}
''')
    meanS=means.loc['Spatial_MILP'];meanG=means.loc['Greedy'];pct=100*net['mean']/meanG.total_g
    incrementpct=100*net['mean']/meanG.increment_g
    compute=f.groupby('strategy').agg(mean_total_solver_wall_s=('solver_wall_seconds','mean'),
        mean_solver_call_wall_s=('mean_solver_seconds','mean'),mean_modelled_controller_compute_s=('controller_computing_seconds','mean'),
        mean_fallback_count=('fallback_count','mean'),mean_status4_subproblem_count=('status4_count','mean'))
    regional=pd.read_csv(out('core')/'regional_execution_mean.csv')
    regional=regional.pivot(index='strategy',columns='region',values='completed_requests').reset_index()
    hcarbon=hp[hp.metric.eq('co2e_g')]
    hview=hcarbon[['proposed','delay_scale','mean','ci95_lower','ci95_upper','positive_fraction']]
    hqualification=[]
    for (s,k),g in H.groupby(['strategy','delay_scale']):
        p=hcarbon[(hcarbon.proposed==s)&np.isclose(hcarbon.delay_scale,k)].iloc[0]
        hqualification.append({'strategy':s,'delay_scale':k,'mean_violation_pct':100*g.service_violation_rate.mean(),
            'max_seed_violation_pct':100*g.service_violation_rate.max(),'all_six_within_assumed_0.1pct':bool((g.service_violation_rate<=.001).all()),
            'all_six_zero_violations':bool((g.service_violation_rate==0).all()),'paired_net_ci_lower_g':p.ci95_lower})
    hqualification=pd.DataFrame(hqualification);hqualification.to_csv(out('H')/'benefit_and_service_qualification.csv',index=False)
    h_boundary=('所有已测试控制倍数的净碳收益95%配对区间下界仍为正；本轮没有找到碳收益归零的临界点，不能向5倍以外外推。' if bool((hcarbon.ci95_lower>0).all()) else '部分已测试控制倍数的净碳收益区间已触及或低于0；应按H完整表识别这些条件，不能继续引用旧版正收益结论。')
    selectedrisk=risk[np.isclose(risk.risk_budget,.001)]
    equaldata=pd.read_csv(out('core')/'equal_service_by_seed.csv')
    write('FINAL_RESULTS_zh.md',f'''# 最终重跑结果与分析

本报告全部正式数值来自本目录实际运行。核心完整7335请求、24共同服务种子、五策略共120次；F、H各保留原六种子规模并重新执行受影响配置，重复引用同一新配置不增加统计样本。原正式结果与 E/F/G/H 全部保留。固定时空输入、冻结S/L与历史质量、真实2025碳快照、DGX功率、A2事件语义和原服务扰动均未改。

## 五策略核心表现（24种子均值；SD仅对总碳另列）

{table(means.reset_index())}

完整种子级数据在 FINAL_COMPARISON.csv，均值/SD/95%区间在 results/core/aggregate.csv。完成请求响应分位数不覆盖失败等待；失败完整计入完成率、服务违反率和能耗账本，不能只看P95。固定队列历史质量不等于已交付质量效用。

控制开销与回退（理想控制核心中的wall仅为描述，不能当作旧新隔离测速）：

{table(compute.reset_index())}

每阶段 before_after_controller_timing.csv 完整对照实际求解wall、模型内优化时间与含上游router的全部计费计算时间；排队及主动持有不重复计为CPU运行。

收益机制需结合真实执行地区：

{table(regional)}

表中是每种子平均完成请求数。Greedy优先在当前能立即执行的候选中择低碳地区；Spatial同时协调到达批次的空间分配和名义容量，可接受短时资源排队。观察到的低碳地区集中及队列等待，属于空间容量协调的实际行为，不等于主动等待未来低碳小时。逐地区忙碌秒数与排队总量见regional_execution_by_seed.csv；原Full还受有限候选开始时刻及提交规则影响，不能仅凭策略名称假定它的实现必须更优。

Spatial总电量略高于Greedy，同时完成更多请求；总碳下降主要反映执行地区碳强度分配的改变。不能将这项结果写成总用电量下降，也不能为了统一完成率而删除额外交付请求或失败记录。

## 模型选择的五个回答

1. Spatial 相对 Greedy：基准减去Spatial为 {sentence(net)}。均值设施总碳减少 {pct:.6f}%，相对Greedy全部增量碳减少 {incrementpct:.6f}%。完成率、违反率和延迟必须结合上表，不以碳单指标决定现实可用性。
2. Spatial 相对原 candidate-time Full MPC：原Full减去Spatial为 {sentence(vsmpc)}。这比较的是两个保留算法在相同条件下的实现；不构成对任意时间优化方法的普遍优越证明。
3. E-Full 的独立主动等待收益：E-Spatial减去E-Full为 {sentence(temporal)}。主Persistence预测平坦，非零hold名义可行集为零hold子集；遇到的非零hold决策共 {len(holds)} 次。存在零分支拒绝时的分支救援，不能据此认定发现独立的低碳时间等待收益。纯时间收益仍未建立。
4. SLA：默认gamma不变。Spatial核心24种子平均违反率 {100*f.loc[f.strategy.eq('Spatial_MILP'),'service_violation_rate'].mean():.9f}%，最大单种子违反率 {100*f.loc[f.strategy.eq('Spatial_MILP'),'service_violation_rate'].max():.9f}%；其中 {(f.loc[f.strategy.eq('Spatial_MILP'),'service_violation_rate']<=.001).sum()}/24 个种子在假设0.1%预算内。所有实际失败/超期都保留。F在假设0.1%业务预算下的经验判定如下；0.1%不是题目官方硬标准，六种子经验合格不等于概率保证。严格“零服务违反”不能据此通过。

{table(selectedrisk)}

安全系数的影响不必单调：增大gamma同时提高服务留量并收紧最迟准入时间，可能减少运行超期，也可能增加提前拒绝。失败同样进入违反率。不能根据本次评估种子事后改选一个最有利gamma，并宣称获得泛化的最优服务标准。

5. 求解控制时间：H是 fresh closed-loop 串行测量，在每次求解结束后平移计划并重新检验约束，非旧耗时日志重放。相对Greedy的实际配对总碳净收益如下；正号仅表示该条件情景下减少排放，还需结合H原始完成率/违反率。单分支Spatial节省未使用分支计算，不改变空间模型。

{table(hview)}

算术净碳收益与服务可接受性分开判定：

{table(hqualification)}

{h_boundary}

若业务要求每个请求都满足硬SLA，任何 all_six_zero_violations=false 的条件都不能被认证为可部署的正收益方案。假设0.1%预算下的经验合格与严格硬SLA是不同标准，不事后替换服务质量要求。

## 固定空闲功率与等服务解释

所有策略共同12h空闲底座250.56kWh、41140.3512g；Spatial固定空闲碳占 {100*41140.3512/meanS.total_g:.6f}%。总百分比受该大底座稀释，不能把增量百分比当总设施减排。六主机、48安装GPU、每主机1可调度槽位保持基准；本阶段不重新调参改变利用率。对照使用同一安装资源，未用S/L池仍计空闲。

等服务辅助口径（事后交集有选择偏差）：

{table(equaldata.groupby('strategy').mean(numeric_only=True).drop(columns=['seed']).reset_index())}

正式判断始终以全部输入和共同观察窗为主，额外交付及失败传输耗能不删除。真实2025逐时life-cycle碳快照用于事后积分；合成请求、功率/网络假设、控制器10W和扰动模型提供条件仿真，其结果不是实际部署减排。

## 数值修复的影响与边界

完整固定整数LP可以修复纯连续恢复问题，但不能修复已无可行整数选择的解。已知原反例属后者；本轮拒绝及fallback保留。核心旧新差异逐项见 NUMERICAL_FIX_REPORT.md 和CSV；F/H使用新恢复协议重新执行，不能继续将旧H数字称作修复后结果。控制wall会随硬件/系统负载变化，跨环境逐位复现未验证。Native求解器工作量/线程/随机控制、真实硬件测量、前瞻业务预测与跨月份泛化仍未在本阶段验证。

图表来自实际CSV：results/core/figures（策略、配对净收益、逐种子时间消融与数值拒绝），results/F/figures（安全余量），results/H/figures（计算开销及净碳收益）。策略柱状图误差线为种子间1个标准差；配对净收益图误差线为95% t区间，统计口径不同。不在本轮重新宣称旧C功率敏感性已用新协议执行；旧C原始结果受保护，现实有效性仍有限。
''')
    spatial_ok=net.ci95_lower>0 and vsmpc.ci95_lower>0
    recommendation='保留 Frozen Pre-LightGBM + Spatial MILP + A2 DES 为本条件仿真的主模型；原 Full MPC 和 E-Full 保留作对照。' if spatial_ok else '本轮证据不足以确认 Spatial 同时优于两个主要对照；保留所有结果，不按预设排名冻结优胜结论。'
    write('FINAL_MODEL_RECOMMENDATION.md',f'''# 最终模型建议

{recommendation}

依据：Greedy−Spatial {sentence(net)}；原Full−Spatial {sentence(vsmpc)}。E受控比较 {sentence(temporal)}，不据此声称主动等待有独立收益。默认gamma=1.35，固定功率与服务目标不改。0.1%业务风险预算仅为显式敏感性假设；若要求所有请求零违反，则本轮任何有违反的模型都不能认证合格。

软件冻结的是“完整检查、不可恢复时明确拒绝、显式fallback、记录所有失败”的实现，不能冻结为“所有求解均成功”“所有请求准时”或“真实部署已证实”。纯时间端到端严格消融、真实硬件、原生固定工作求解协议及业务预测仍未验证。不得因排名而删除失败或修改参数。
''')
    write('CHANGELOG.md','''# 本次变更

全部新增文件仅在独立 final_release 内；旧目录保护哈希全部一致。

- v2/des.py：A2原样副本，字节相同；未改物理语义。
- v2/milp.py：原Full纯整数路径增加完整上下界/行校验；无连续自由度时拒绝非法取整。
- queue_milp.py、recovery.py：固定整数连续LP恢复，完整验证与证据保存；原目标、候选、容量、deadline与1e-5阈值不变。
- spatial.py：主Spatial调用同一修复后queue求解器；保留原控制/派发逻辑。
- policies.py：原E嵌套控制器副本；所有分支使用新恢复函数。
- experiments.py、configs/protocol.json：实际fresh120核心与F/H重跑、逐请求/逐事件/账本和恢复次数；独立种子与共同随机条件不改。
- tests/、run_tests.py：已有70测试只读执行，加13数值/保护测试；初始失败尝试保留。
- analyze_release.py、validate_release.py：种子级配对统计、旧新差异、独立逐小时碳积分、物理容量/能耗/请求守恒与重复验证。
- 报告、图表、运行入口、版本/哈希清单：用于固定本次实际证据和复现。

没有调整训练、S/L、真实碳、DGX功率、随机扰动或服务标准。第一次pilot仅修正汇总层holding_s缺省处理，记录保留。初次core汇总追上独立worker时禁止覆盖保护触发，core_attempt1.log、pipeline_attempt1.log保留；等待120个metrics全部完成后仅恢复汇总，没有重复创建独立样本。最终独立碳核算脚本最初误写gco2字段名，按实际metadata改为gco2e后重新检查，validate_release_attempt1.log与finish_checks_attempt1.log保留；原仿真使用的碳字段及结果未改。报告脚本增加显式UTF-8声明以解决本地Python3.9文件读取报错，reports_release_attempt1.log保留；报告阶段未重算或更改实验数值。测量CPU的H结果可因恢复开销和本机wall变化而改变，未强制匹配旧数值。原正式baseline及此前E/F/G/H均未覆盖。
''')
    write('README.md','''# Final numerical maintenance release

先读 FINAL_RESULTS_zh.md，再读 FINAL_MODEL_RECOMMENDATION.md、FINAL_VALIDATION_REPORT.md 和 NUMERICAL_FIX_REPORT.md。数学定义在 FINAL_MODEL_FORMULATION.md，运行方式在 REPRODUCIBILITY.md。全部新文件位于当前 C: worktree 的 final_release，未自动复制到D:。

FINAL_COMPARISON.csv为120条核心策略-种子结果；results/core 为24独立配对种子的统计、数值日志汇总、等服务辅助归因和新旧逐请求/事件对照。results/F、results/H 为原六种子范围重跑。各目录figures由实际CSV生成。runs保存每次完整请求、事件、账本、求解分支、配置及公共扰动；run_inventory.csv包含失败pilot，不隐藏样本。

冻结状态必须查 FREEZE_STATUS.json；存在文件或报告不代表已经冻结。freeze_manifest.json固定实际数据/源码/正式结果/图表与依赖信息，使用 run_all.py --verify 只读检查。run_all.py --reproduce final_release_reproduction_001在新目录重新运行，不覆盖交付件。

所有结果是冻结历史质量、合成到达、2025碳快照与DGX假设下的条件仿真，不是生产部署实测。软件正确性PASS、原始数值候选可能不可恢复、实际SLA可能违反、现实有效性未验证，是四个不同命题。
''')
    write('Results_and_Discussion_en.md',f'''# Results and Discussion

We reran five preserved policies on all 7,335 frozen requests and 24 paired
service-noise seeds. Spatial MILP reduced facility emissions relative to
Greedy by {net['mean']:.6f} g CO2e (paired 95% t interval
[{net.ci95_lower:.6f}, {net.ci95_upper:.6f}]), and relative to the original
candidate-time Full MPC by {vsmpc['mean']:.6f} g. These effects are conditional
on the fixed alignment, resource model and retrospective carbon snapshots;
they are not measured deployment savings.

The common idle base dominated facility electricity. Total percentage savings
were {pct:.6f}%, whereas the corresponding fraction of Greedy incremental
emissions was {incrementpct:.6f}%. Reporting both avoids denominator ambiguity.
Failures and already-incurred transfer emissions remained in all-input totals.
Quality labels were used only for post-simulation delivered-utility accounting.

Fixed-integer continuous recovery did not alter assignments. The saved real
counterexample was genuinely infeasible after rounding, with a 0.0037348 s
bound contradiction, and was correctly rejected with explicit fallback.
Recoverable synthetic cases established software capability, not evidence
that all real rejected incumbents could be recovered. Nonzero holding in
the controlled E comparison cannot be interpreted as independent temporal
carbon value under flat persistence forecasts, particularly when it rescues
a failed zero-hold solve.

Safety-margin and measured-control experiments were rerun on the original
six-seed scope. Their results are exploratory and do not establish calibrated
chance constraints, zero violations or hardware-independent runtime. Native
fixed-work solver determinism, physical power calibration, operational as-of
carbon forecasts and cross-window generalization remain unverified.
''')
    save(ROOT/'audit/reports_completed.json',{'status':'PASS','recommendation':recommendation,'numerical_totals':numeric.to_dict('records'),
        'check_counts':validation.status.value_counts().to_dict(),'preserved_failed_pilot_count':int(inventory.status.eq('FAILED_INCOMPLETE_RETAINED').sum())})
    print('Reports generated from actual completed runs')
if __name__=='__main__':main()
