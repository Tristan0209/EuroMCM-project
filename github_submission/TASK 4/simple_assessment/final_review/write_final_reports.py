from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SIMPLE = HERE.parent
sys.path.insert(0,str(SIMPLE.parent))
import numpy as np
import pandas as pd
from task4_core.sources import read

SEEDS = [2029101800,2029101801,2029101802]
SCENARIOS = ['Baseline','Difficult','Prediction Error','Demand Growth']
TASKS = ['General','Math','Code']
POLICIES = ['Greedy','Spatial_MILP']


def tab(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+
                    ['| '+' | '.join(str(v) for v in row)+' |' for row in rows])


def link(path,label):
    return '['+label+'](<'+str(path.resolve()).replace('\\','/')+'>)'


def new(name,value):
    path = SIMPLE/name
    assert not path.exists(),'Preserve prior final report; review before versioning'
    path.write_text(value.strip()+'\n',encoding='utf-8')


def fmt_range(series,decimals=0):
    a,b = series.min(),series.max()
    return f'{a:.{decimals}f}' if a==b else f'{a:.{decimals}f}–{b:.{decimals}f}'


def main():
    checks = read(HERE/'independent_checks.json')
    assert checks['status']=='PASS'
    original = pd.read_csv(SIMPLE/'Task4_Simple_Results.csv')
    flip = pd.read_csv(HERE/'prediction_flips.csv')
    fr = flip[flip.task_type.eq('All')].sort_values('seed')
    router = pd.read_csv(HERE/'router_vs_AllL.csv')
    service = pd.read_csv(HERE/'group_service.csv')
    energy = pd.read_csv(HERE/'energy_comparison.csv')
    ep = pd.read_csv(HERE/'energy_paired.csv')
    reg = pd.read_csv(HERE/'region_execution.csv')
    rb = pd.read_csv(HERE/'rebound_audit.csv').sort_values('seed')
    comp = pd.read_csv(HERE/'task_composition.csv')
    attribution = pd.read_csv(HERE/'active_carbon_identity.csv')
    om = original.groupby(['scenario','policy']).mean(numeric_only=True)
    em = energy.groupby(['scenario','policy']).mean(numeric_only=True)
    rm = router.groupby(['scenario','task_type']).mean(numeric_only=True)
    sm = service.groupby(['scenario','task_type','policy']).mean(numeric_only=True)
    def o(s,p,k):return float(om.loc[(s,p),k])
    def e(s,p,k):return float(em.loc[(s,p),k])
    baseline = ep[ep.scenario.eq('Baseline')]
    mean_delta = rb.cross_demand_delta_g.mean()
    rel_delta = rb.cross_demand_delta_fraction.mean()*100
    reference = rb.r_linear_reference.mean()*100
    flip_table = tab(['seed','L→S','S→L','总翻转/N','General/Math/Code翻转'],[
        [int(r.seed),int(r.L_to_S),int(r.S_to_L),f'{int(r.n_changed)}/7335 ({r.change_fraction*100:.3f}%)',
         '/'.join(str(int(flip[(flip.seed==r.seed)&flip.task_type.eq(t)].n_changed.iloc[0])) for t in TASKS)]
        for r in fr.itertuples()])
    flip_quality_table = tab(['seed','翻转前均分','翻转后均分','配对ΔScore','前/后达标%','明显下降/n_changed','新S失去L达标数'],[
        [int(r.seed),f'{r.quality_before_changed:.6f}',f'{r.quality_after_changed:.6f}',f'{r.quality_delta_changed:+.6f}',
         f'{r.pass_before_changed*100:.3f}/{r.pass_after_changed*100:.3f}',
         f'{int(r.material_drop_count)}/{int(r.n_changed)} ({r.material_drop_fraction_changed*100:.3f}%)',
         int(r.lost_L_pass_due_to_new_S)] for r in fr.itertuples()])
    rb_table = tab(['seed','c_inc基线 g/输入','c_inc增长 g/输入','增长/基线','线性r参考%','实际跨需求ΔC g','ΔC/原Greedy %','DES−线性 g'],[
        [int(r.seed),f'{r.c_incremental_baseline:.8f}',f'{r.c_incremental_growth:.8f}',f'{r.incremental_per_input_multiplier:.4f}',
         f'{r.r_linear_reference*100:.4f}',f'{r.cross_demand_delta_g:+.6f}',f'{r.cross_demand_delta_fraction*100:.6f}',f'{r.DES_minus_linear_g:+.6f}'] for r in rb.itertuples()])
    regrows = []
    for s in ['Baseline','Demand Growth']:
        for region in ['DE','FR','ES']:
            f = reg[reg.scenario.eq(s)&reg.policy.eq('Spatial_MILP')&reg.region.eq(region)]
            regrows.append([s,region,f'{f.completed.mean():.3f}',f'{f.active_energy_kwh.mean():.6f}',
                            f'{f.active_energy_share.mean()*100:.3f}',f'{f.effective_active_intensity.mean():.3f}',f'{f.active_carbon_g.mean():.6f}'])
    region_table = tab(['P1情景','执行地区','完成数均值','active kWh','active电量份额%','执行电量加权强度 g/kWh','active CO2e g'],regrows)
    quality_rows = []
    for s in SCENARIOS:
        for t in TASKS:
            f = router[router.scenario.eq(s)&router.task_type.eq(t)]
            a = rm.loc[(s,t)]
            quality_rows.append([s,t,fmt_range(f.n_input),fmt_range(f.unique_prompt_groups),
                f'{a.router_quality:.6f}',f'{a.AllL_quality:.6f}',f'{a.delta_quality:+.6f}',
                f'{a.router_pass*100:.3f}/{a.AllL_pass*100:.3f}',f'{a.delta_pass*100:+.4f}',
                f'{a.S_fraction*100:.3f}/{a.L_fraction*100:.3f}'])
    quality_table = tab(['情景','任务','每seed实例数','unique prompt组','Q_router','Q_AllL','配对ΔQ','Router/AllL达标%','达标差pp','S/L %'],quality_rows)
    service_rows = []
    positive_service = 0
    for s in SCENARIOS:
        for t in TASKS:
            a,b = [sm.loc[(s,t,p)] for p in POLICIES]
            x = service[service.scenario.eq(s)&service.task_type.eq(t)]
            aa = x[x.policy.eq('Greedy')].set_index('seed')
            bb = x[x.policy.eq('Spatial_MILP')].set_index('seed')
            positive_service += int(((bb.violation_rate-aa.violation_rate)>1e-12).sum())
            service_rows.append([s,t,f'{a.violation_rate*100:.4f}/{b.violation_rate*100:.4f}',
                f'{(b.violation_rate-a.violation_rate)*100:+.4f}',
                f'{a.completion_rate*100:.3f}/{b.completion_rate*100:.3f}',
                f'{a.completed_p95_s:.3f}/{b.completed_p95_s:.3f}',
                f'{a.valid_pass_rate*100:.3f}/{b.valid_pass_rate*100:.3f}'])
    service_table = tab(['情景','任务','G/S违反%','S−G违反pp','G/S完成%','G/S已完成P95秒','G/S有效交付达标%'],service_rows)
    assert positive_service==0
    energy_rows = []
    for s in SCENARIOS:
        for p in POLICIES:
            a = em.loc[(s,p)]
            energy_rows.append([s,p,f'{a.energy_kwh:.6f}',f'{a.carbon_g:.6f}',
                f'{a.idle_energy_kwh:.2f}/{a.idle_carbon_g:.4f}',
                f'{a.operational_energy_kwh:.6f}/{a.operational_carbon_g:.6f}',
                f'{a.network_energy_kwh+a.controller_energy_kwh:.6f}/{a.network_carbon_g+a.controller_carbon_g:.6f}'])
    energy_table = tab(['情景','策略','总kWh','总gCO2e','idle kWh/g','运行增量 kWh/g','其中network+ctrl kWh/g'],energy_rows)
    rates = tab(['同需求情景','节电率%（正为少用电）','设施减碳率%','扣idle运行减碳率%'],[
        [s,f'{ep[ep.scenario.eq(s)].energy_saving_fraction.mean()*100:+.6f}',
         f'{ep[ep.scenario.eq(s)].carbon_saving_fraction.mean()*100:.6f}',
         f'{ep[ep.scenario.eq(s)].operational_carbon_saving_fraction.mean()*100:.4f}'] for s in SCENARIOS])
    c = comp.groupby('seed')[['score_composition_component','score_within_component','pass_composition_component','pass_within_component']].sum().mean()
    composition_table = tab(['类别','Baseline权重%','40% Math/Code权重%','Baseline Score','Baseline达标%','新情景Score均值','新情景达标%'],[
        [t,f'{comp[comp.task_type.eq(t)].weight_baseline.mean()*100:.3f}',f'{comp[comp.task_type.eq(t)].weight_difficult.mean()*100:.3f}',
         f'{comp[comp.task_type.eq(t)].score_baseline.mean():.6f}',f'{comp[comp.task_type.eq(t)].pass_baseline.mean()*100:.3f}',
         f'{comp[comp.task_type.eq(t)].score_difficult.mean():.6f}',f'{comp[comp.task_type.eq(t)].pass_difficult.mean()*100:.3f}'] for t in TASKS])
    original_table = tab(['情景','策略','历史Score','历史达标%','P95秒','违反%','完成%','CO2e g'],[
        [s,p,f'{o(s,p,"mean_historical_quality"):.6f}',f'{o(s,p,"answer_pass_rate")*100:.3f}',
         f'{o(s,p,"p95_response_s"):.3f}',f'{o(s,p,"service_violation_rate")*100:.4f}',
         f'{o(s,p,"completion_rate")*100:.4f}',f'{o(s,p,"co2e_g"):.3f}'] for s in SCENARIOS for p in POLICIES])
    fig1 = str((SIMPLE/'figures/Figure1_Robustness.png').resolve()).replace('\\','/')
    fig2 = str((SIMPLE/'figures/Figure2_Carbon_Growth.png').resolve()).replace('\\','/')
    fig3 = str((SIMPLE/'figures/Figure3_Fairness.png').resolve()).replace('\\','/')
    baseline_general = rm.loc[('Baseline','General')]
    equal_fraction = (fr.quality_unchanged_count/fr.n_changed).mean()*100
    flip_abs = fr.changed_quality_abs_delta_mean.mean()
    evidence_table = tab(['证据层级','本项目实际使用','能够支持/仍缺什么'],[
        ['真实公开/历史电网数据','DE/FR/ES 2025逐小时、2026-02回溯快照，消费侧生命周期CO2e；来源归属按格式推断','支持该固定窗的历史归属核算；精确因子版本、当时发布vintage与独立来源链尚缺'],
        ['历史benchmark答案','RouterBench同ID的S/L归档score，阈值0.5，未重写','支持本cohort选模配对；不是新回答、不是代表性公共部门或人口属性样本'],
        ['公开测量转移代理','busy GPU 760W来自特定Qwen/B200配置转用','不是目标Mixtral/GPT-4服务的硬件/整机功率测量；需要目标模型独立遥测'],
        ['人为假设','shared1500W、GPU idle175W、PUE1.2、服务代理、origin/class、network/control','决定仿真边界，不能写成实测值；需要墙插/IT/设施电表及真实服务日志'],
        ['DES模拟结果','完整24执行及固定H账本、失败/完成记录','只支持这些情景的条件比较，不是部署认证或因果需求反弹'],
        ['现实缺失证据','代表性业务/质量、真实PUE/功率/排队、as-of预测、边际排放反事实','平均电网强度是归属核算，不等同改变负荷造成的边际电网因果减排']])
    new('Task4_Simple_Report_Final_ZH.md',fr'''
# 第四问最终结果：基于情景的鲁棒性与可持续性评估

日期：2026-10-18。**仅复用原24次执行，新增仿真、推断、训练均为0。**原Results、历史报告和三图保持不变；本报告及追加统计为新版本。复核未发现要求改写原始实验的数学或数据错误，补充解决了反弹解释、预测局部损害、双层类别审计及节电/减碳混淆。

## 1. 问题、方法和统计边界

题面第四条要求覆盖：更高难度请求、不可靠预测、意外用量增长、总排放反增、某类请求持续不利，以及环保效益证据。对应采用四情景：原Baseline；固定7335实例且Math+Code=40%；两模型预测各加独立N(0,0.10²)并clip；原7335追加floor(0.5×7335)=3667个唯一实例，总11002。旧Difficult情景的论文名称改为 **Higher Share of Math/Code Requests**，其为任务组成和计算负荷代理，不代表每条请求实际推理更难。

每情景3个预定seed {SEEDS}，两策略Greedy/P1最终Spatial MILP共享同world、路由、扰动、6host/48GPU/6准入slots、actual碳与12h窗口。保留冻结Task2 Pre_direct成本/阈值与Code/Math/unknown L回退；噪声后重新执行规则并切换对应事前服务代理。Task3主模型是最终Spatial MILP + DES，未修改目标、约束、容差、时间预算或账本。

真实两动作score只在原执行结束后评价；本轮只重读。质量比较逐base_request_id配对；复制实例保留group，不创造独立质量样本。报告所有均值先按seed计算，再平均3个值。每种子百分比之均值与均值之比不混用；这里节省率也先按seed取比值，细小末位差不得误称数据冲突。没有显著性检验、95%推断或新的风险覆盖保证。

设q_iS、q_iL为历史score，m_i为该情景的冻结规则选择，I_i表示完成且同时满足deadline与额外等待约束，则

$$
Q_{{router}}=\frac{{1}}{{N}}\sum_i q_{{i,m_i}},\qquad
P_{{hist}}=\frac{{1}}{{N}}\sum_i\mathbf{{1}}(q_{{i,m_i}}\ge 0.5),\qquad
P_{{valid}}=\frac{{1}}{{N}}\sum_i I_i\mathbf{{1}}(q_{{i,m_i}}\ge 0.5),\qquad
V=\frac{{1}}{{N}}\sum_i(1-I_i).
$$

完成率只要求completed；有效交付还要求合规。P95仅对每seed的已完成者计算，所以必须并列完成率/全输入违反率；不同完成集合会产生选择效应。24执行的842个失败是跨策略/重复cohort的终态计数，不是842个独立新题；截尾为0，不证明尾部永不截尾。固定窗口和部分能耗合同保持。

{original_table}

![原图1：情景鲁棒性](<{fig1}>)

图1保留原文件。图中“Difficulty 40% hard”仅指Math/Code份额40%；应按正文代理含义解读。点为seed，横杠为均值、竖线为观察范围，非置信区间。P95与全输入服务违反率是两个独立面板。

## 2. Higher Share of Math/Code Requests：为什么均分下降而达标率上升

原Baseline权重为General5539/7335、Math1710/7335、Code86/7335；新情景为4401、2794、140。Math:Code近似保留原比例，整数采用最大余数法。真实q没有变，任务及其样本权重发生变化。

{composition_table}

Math的平均Score约0.6624，低于General约0.8125；但Math的q≥0.5达标率约91.99%，高于General约81.51%。均值E[q]与阈值概率P(q≥0.5)描述分布不同部分，数学上没有同向变化要求；跨benchmark的Score刻度/部分分值也不是统一的真实推理难度量表。因此增加Math权重可同时拉低均分、抬高达标率，不能据此断言数据错误或系统质量全面改善。

令w_g、μ_g分别为任务权重和组内均值，D为新混合、B为基线，则有精确恒等式（不是新拟合模型）

$$
\Delta Q=\sum_g(w_g^D-w_g^B)\mu_g^B+\sum_g w_g^D(\mu_g^D-\mu_g^B).
$$

对达标指标以μ_g=P_g替换。逐seed分解后均值：Score变化中组成项{c.score_composition_component:+.6f}、组内重采样项{c.score_within_component:+.6f}，合计{o('Difficult','Spatial_MILP','mean_historical_quality')-o('Baseline','Spatial_MILP','mean_historical_quality'):+.6f}；达标率组成项{c.pass_composition_component*100:+.4f}pp、组内项{c.pass_within_component*100:+.4f}pp，合计{(o('Difficult','Spatial_MILP','answer_pass_rate')-o('Baseline','Spatial_MILP','answer_pass_rate'))*100:+.4f}pp。组内项包括有限重采样差异，不能解释为因果算法效果。

P1名义服务总时长从{o('Baseline','Spatial_MILP','nominal_service_sum_s'):.2f}s增至{o('Difficult','Spatial_MILP','nominal_service_sum_s'):.2f}s（+{(o('Difficult','Spatial_MILP','nominal_service_sum_s')/o('Baseline','Spatial_MILP','nominal_service_sum_s')-1)*100:.2f}%），L份额从94.124%增至{o('Difficult','Spatial_MILP','L_share')*100:.3f}%。P95从7.827s升至11.967s，平均服务违反仍为0.0182%，完成率99.9818%。可说此负载代理下维持较高完成水平但延迟恶化；不能说每条请求变难，也不能宣称严格零风险。

## 3. Prediction Error：少量翻转仍可能造成局部损害

### 3.1 方向与任务类别

{flip_table}

原Baseline每seed固定S431/L6904。噪声后S/L分别为428/6907、442/6893、434/6901。全部翻转属于General；Math与Code均0，既定L回退没有被误差突破。翻转比例以全部7335为分母，不以两策略重复记录翻倍计数。

### 3.2 只在同一历史请求的翻转集合配对

令F_b={{i:m_i^B≠m_i^E}}，δ_i=q_{{i,m_i^E}}−q_{{i,m_i^B}}。下表所有前后质量与达标率使用同一个F_b，不能拿不同请求组均值代替配对差。报告约定“明显下降”为δ_i≤−0.10绝对Score；同时另报所有下降，阈值是本轮预先声明的描述口径，非官方标准。

{flip_quality_table}

各seed下降/改善/不变数分别为19/18/100、22/18/99、18/21/94；这里所有实际下降也均达到0.10阈值。新S丢失L可达标的定义为原L→新S且q_L≥.5、q_S<.5，所以13/15/12是局部损害数，不是整个系统新增失败数，也不等于净达标变化。翻转集合达标数净变化分别为−2、−4、+2，存在质量改善抵消。

小总体变化的三个数据原因必须一起陈述：仅{fr.change_fraction.min()*100:.3f}%–{fr.change_fraction.max()*100:.3f}%翻转；翻转集合约{equal_fraction:.2f}%两模型同分；剩余请求的增益与损失抵消。不能笼统说“变化请求两模型质量都很接近”：翻转集合平均绝对质量差约{flip_abs:.6f}，局部损害并不小。精确关系为

$$
\Delta Q_b=\frac{{|F_b|}}{{7335}}\overline{{\delta}}_{{F_b}}.
$$

三个seed总体ΔScore为{', '.join(f'{x:+.8f}' for x in fr.population_quality_delta)}，均值{fr.population_quality_delta.mean():+.8f}；总体达标变化均值{fr.population_pass_delta.mean()*100:+.5f}pp。409次跨seed翻转对应290个distinct base ID，不能称409个新独立质量样本。改选影响对应的事前service已同步，P1平均服务违反仍0.0182%、P95约7.819s，不能用质量均值掩盖上述个体损害。

Code/Math/unknown固定回退在机制上隔离选模翻转，但没有取消guard消融，不能量化guard避免多少损害；没有重新confirm校准，原有限样本统计覆盖不能自动推广至噪声或新混合。此结论不需要新增Prediction Error实验。

## 4. Demand Growth与反弹：以实际跨需求总账为主

增长实得49.9932%，原7335实例全部保留、追加3667个ID、6host及H不变。P1服务违反0.0182%→0.1697%、P95 7.827→11.885s，完成率99.8303%；同增长Greedy服务违反1.6270%。增长提高负载，P1仍较Greedy少违反，但P95较Greedy更高，不能用单指标宣布全面优越。

### 4.1 三种不同问题

1. **同需求调度差**：C_G(N)−C_S(N)。N0时均值73.622355g；N1时101.442949g。两者始终保持相同请求世界/硬件，回答调度贡献。
2. **跨需求总排放**：C_S(N1)−C_G(N0)。本次均值+{mean_delta:.6f}g，三个seed都为正，回答外生增长后的总账是否反增；不能与第一种差混用。
3. **效率诱导行为反弹**：需要效率改善→需求增加的可识别因果通道。本次没有价格/采用率/需求因果数据，仅施加外生增长，未识别真实行为反弹。

### 4.2 线性公式本身正确，但条件不成立

在相同固定idle及常数每输入运行增量c_0>0下：

$$
C_S^{{lin}}(N)=C_{{idle}}+Nc_0,\qquad c_0=\frac{{C_S(N_0)-C_{{idle}}}}{{N_0}},\qquad
r_{{lin}}=\frac{{C_G(N_0)-C_S(N_0)}}{{C_S(N_0)-C_{{idle}}}}.
$$

逐seed把r_lin代回C_idle+(1+r_lin)N0c0，均精确恢复C_G(N0)，残差在数值容差内为0。这里c0必须包含active、network、controller全部增量，并以所有输入数为分母，不以完成数代替需求，也不能把总设施碳按N整体放大。若分母≤0或r=0已无优势，需另外说明；本次不属于这些退化情形。

{rb_table}

99.21%是三个seed **常数边际系数参考的均值**，不是DES求得的实际反弹阈值。真实增长点的每输入增量c_inc均升高约33.64%–36.27%，所以约50%增长已有总排放超过原Greedy，而线性参考仍预测保有优势。只测r=0与约.5，不能定位准确DES临界率、证明中间单调性或真实行为阈值；这不是公式算错。

### 4.3 用地区与小时账本解释失效

{region_table}

Spatial的active电量约1.742479→2.610571kWh（约×1.498），但active碳约73.598169→149.405761g（约×2.030）；执行电量加权碳强度逐seed均值42.237→57.231gCO2e/kWh。FR的active电量份额89.583%→79.632%，ES约9.855%→17.881%，DE约0.562%→2.487%；表中每地区执行加权强度仅小幅变化。所有world实际地区/小时碳表完全相同，变化来自执行时空电量权重，不能说电网外生强度被增长改写。

逐region×hour拆分active区间后独立重建C_active=ΣE_rh I_rh，与原总账一致。其精确差分恒等式为

$$
\Delta C_{{active}}=(E_1-E_0)\bar I_0+E_1(\bar I_1-\bar I_0).
$$

三个seed平均active增碳{attribution.active_carbon_change_g.mean():.6f}g，其中电量规模项{attribution.energy_volume_term_g.mean():.6f}g、执行加权强度项{attribution.execution_mix_intensity_term_g.mean():.6f}g。该拆分依顺序约定，是核算恒等式，不是新优化模型或因果归因。它与清洁池相对拥塞、更多计算溢向高碳地区的解释一致；不据此识别每个拒绝的因果贡献。network/controller也保留，均未被删掉。

### 4.4 差异很小，不能认证现实增排

原Greedy均值41288.178g、原Spatial41214.556g、增长Spatial41290.674g。+2.495812g占原总量的逐seed相对差均值约 **{rel_delta:.6f}%**；它小于现实中许多未校准参数的可能影响，但本轮没有运行参数敏感性，不能声称已经验证其符号稳健或必然反转。目标功率、地区/时段PUE、碳强度及服务配置可能改变量级与执行分配。

![原图2：固定需求对照与跨需求总排放](<{fig2}>)

图2同一需求下比较G/S是调度差；“Base Greedy”与“+50% Spatial”是跨需求总排放差。左侧总量含idle、纵轴从0开始；右侧另展示增量，不能用增量比例代替总量减排，也不能将图题当行为反弹已识别。没有新增增长仿真。

## 5. 双层任务类别审计

### 5.1 选模层：Router与离线All-L历史质量参照

$$
Q_{{router,g}}=\frac{{1}}{{N_g}}\sum_{{i\in g}}q_{{i,m_i}},\qquad
Q_{{AllL,g}}=\frac{{1}}{{N_g}}\sum_{{i\in g}}q_{{i,L}},\qquad
\Delta Q_g=\frac{{1}}{{N_g}}\sum_{{i\in g}}(q_{{i,m_i}}-q_{{i,L}}).
$$

达标差同样按逐请求阈值指示之差求均值。All-L仅为已保存两动作score的离线参照，不假定其必定按时交付，没有运行All-L DES，也不假定L逐题都优于S。

{quality_table}

Baseline General的配对ΔQ={baseline_general.delta_quality:+.6f}、达标差{baseline_general.delta_pass*100:+.4f}pp：模型选择损害在此总体上集中于允许选S的General；这是可见的质量/资源折衷，需要单列，而不是用调度层的低违反率掩盖。Code/Math全部L，因此相对自身All-L差为0，并不意味着这些组的绝对质量完美。

Baseline General的431条S中，41条比L历史Score高、74条比L低，其余316条相同；其余仍可存在“L不如S”的题却被选L。因此不能按单调性假定计算损害。全体General、Math、Code分别有235、187、6条q_S>q_L；这些是固定历史实例数，不能把跨seed复制计新样本。两类回退维持了选模参照差为0，也可能放弃S更好的个题，未重新优化此规则。

### 5.2 调度层：同world、同task的G/S服务对照

{service_table}

G/S分别为Greedy/Spatial，P95均仅完成者；有效交付达标率以全输入为分母。36个task×scenario×seed比较中，Spatial−Greedy服务违反差为正的有0个：保留“在本次类别比较中没有提高违反率”，不能扩大成不存在任何不公平。所有类别的Spatial P95通常更长；不同完成集合和等待是潜在解释，不能只用违反率声明全面服务改善。

绝对表现上Code的历史达标与尾响应较差，Math的mean Score更低却阈值达标高；这些不同口径不能排序成一个公平总分。增长情景P1违反率General约0.1404%、Math0.2482%、Code0.4919%，相对各自Baseline增量分别约+0.1223、+0.2287、+0.4919pp，Code冲击最大；但Code基线仅86实例、85 distinct prompt groups，增长只是复制到128–144实例，仍为85组。困难情景Code的140实例仅69–73组。重复不增加独立答案证据，组数也不是统计独立性的证明。

![原图3：同类别服务差](<{fig3}>)

图3纵轴为Spatial−Greedy违反率百分点差，负值表示Spatial较少违反，0表示相同；不是负的失败概率。图只评价调度层，选模层必须结合本节第一张表。真实人口属性公平性没有数据支持；“Math更慢”本身不等于歧视。

## 6. 节电与减碳、环境证据边界

{energy_table}

运行增量=总账−idle=active+network+controller；最后一列是其中一部分，不能再次加到增量上重复计费。所有数字保持原12h、host idle一次、IT PUE一次、network独立边界。总量和分项逐seed相加复核通过。

{rates}

Baseline Spatial比Greedy多用约{(-baseline.energy_saving_kwh.mean()):.6f}kWh，节电率{baseline.energy_saving_fraction.mean()*100:+.6f}%，却减碳{baseline.carbon_saving_g.mean():.6f}g。Spatial完成更多工作并改变执行时空碳权重，故 **Energy efficiency不等同Carbon efficiency**；本证据支持同需求减碳，不支持宣称其总电量更低。节省率定义均为(同seed G−S)/G，负节电率表示更多耗电；未完成/拒绝导致的少用电不是成功交付效率。

固定idle电量250.56kWh、碳41140.3512g，对Baseline Spatial约占总碳{e('Baseline','Spatial_MILP','idle_carbon_fraction')*100:.6f}%。其含义是六台已供电的8GPU主机在整个12h的固定基底占主导，不是GPU有99.82%时间空闲，也不是该比例可自动省去。忙碌增量的相对减碳约49.80%，设施总减碳仅0.1783%，两分母回答不同问题。第三问既有24seed的0.178144%/49.758542%与本问3seed略有差异来自种子范围，未改动原结果。

{evidence_table}

现有公开来源与项目出处沿用此前 {link(SIMPLE/'Task4_Environmental_Evidence.csv','Evidence and Limitations表')} 和 {link(SIMPLE/'audit/public_evidence_sources.json','来源核查记录')}；本轮未新增外部研究。NVIDIA整机规格上限不验证本系统busy/idle参数。PUE为假设、目标模型功率未直接测量、RouterBench不代表公共部门业务；平均电网碳核算不等同边际因果减排，条件仿真不等于实机认证。真实确认需要目标模型墙插/设施遥测、真实工作负载与质量、PUE/服务/预测日志及适当碳反事实，而不是删idle提高百分比。

## 7. 最终回答与论文就绪

| 题面第四条问题 | 本次可支持的回答 | 证据边界 |
| --- | --- | --- |
| 请求变难是否可靠 | 40% Math/Code代理下完成/违反仍较好，但负荷与P95上升 | 非逐题真实难度提升；质量与达标分开 |
| 预测不可靠如何影响 | 133–139翻转，局部明显损害及L达标丢失已量化，总均值很小 | 固定噪声/3seed，无guard消融或新覆盖 |
| 用量意外增长 | +49.9932%时SLA风险和尾延迟恶化，P1同需求下仍较G少违反 | 非普遍鲁棒保证，未以拒绝当节能 |
| 高效系统会否总碳反增 | 增长Spatial比原Greedy平均+2.495812g，约0.006045% | 条件总账变化；非行为反弹识别 |
| 是否损害某类请求 | General有相对All-L选模均值损失；未观察P1相对G违反率增量；增长Code脆弱 | 区分两个层级，样本/人口公平证据有限 |
| 环保主张需要什么证据 | 完整历史碳与执行账本支持归属比较，目标硬件/业务/因果反事实缺失 | 不能认证真实部署环保收益 |

线性公式正确且条件已明确；99.21%仅为失效条件下的理论参照，实际主结论来自两个需求水平的完整DES。新增统计均可逐seed、逐ID或逐ledger回溯。Critical未发现原始执行/数据/公式错误；Important解释与缺失统计已补齐；三张图保持原文件，通过正文明确标签含义与差值方向。

**TASK 4 READY FOR PAPER WRITING**。该状态表示按题面第四条和本轮授权足以撰写条件仿真论文，不是部署或公平/风险认证；不再自动编写英文全文或追加实验。

追加统计：{link(SIMPLE/'Task4_Additional_Analysis.csv','统一长表')}；单位、分母与缺失规则：{link(HERE/'statistics_schema.json','数据字典')}；审查修复与独立验证见本轮Final文件。旧报告、Results和24执行保持不变。
''')

    new('Task4_Final_Review_Fixes_ZH.md',fr'''
# TASK 4 最终审查修复记录

日期：2026-10-18。范围为原24执行的离线复核、补充统计和新的论文材料版本。新增DES、模型推断、训练均为0；不改原Results/报告/三图/运行配置/日志，Task2/Task3冻结产物只读。

## 1. Critical / Important / Minor

| 级别 | 原始错误、解释或缺失 | 审查证据与处理 | 当前结论 |
| --- | --- | --- | --- |
| Critical | 核对执行/数据/数学公式是否错误 | 24manifest与逐请求/总账重算一致；线性r逐seed代回原Greedy总碳，残差0；真实q按base ID对原outcomes核对 | 未发现要求修改原实验的Critical；无未解决项 |
| Important | 将线性99.21%过度突出，易读成实际DES阈值 | 保留公式并写明固定idle/常数c_inc；实际c_inc上升33.64%–36.27%；主结论改为实际跨需求+2.495812g、0.006045% | 已解决，属适用条件与解释，不是公式算错 |
| Important | 总体噪声质量均值掩盖局部损害 | 增加L→S/S→L、类别翻转；同ID翻转集score/pass、下降阈值与13/15/12新S丢失L达标 | 已解决，无取消guard消融或新风险保证主张 |
| Important | 原公平性主要是调度层 | 新增12个task×scenario的Router−离线All-L score/pass、S/L与unique groups；调度层保留违反/完成/P95并补有效交付 | 已解决，不能扩大为不存在不公平 |
| Important | “Difficulty”易误解为每题真实更难 | 论文命名Higher Share of Math/Code Requests；用权重×组内score/pass及精确分解解释相反方向 | 已解决，无新Difficulty模型 |
| Important | 原节电/减碳未充分并列表格 | 所有情景能量/碳/idle/增量/net+ctrl；Baseline P1多用0.004857kWh而减碳73.622355g | 已解决，未删idle、未双加net/ctrl |
| Minor | unique prompt组易被称为已证明独立样本 | 报distinct组数；复制和3seed重复不增加独立质量样本，Code85原组数明确 | 已澄清 |
| Minor | 运行记录legacy历史质量列全空可能误读为缺标签 | 权威q_s/q_l来自quality_harms且逐base ID核验；不使用空列，更不填0 | 原设计的分离评价字段，不是缺失实验真值 |
| Minor | 百分比聚合/单位/图标签 | 新CSV显式每seed单位/分母，报告mean of seed ratios；沿用旧三图以新caption解释代理、跨需求和负差方向 | 已澄清；不改旧文件 |

## 2. 关键补充数值

{flip_table}

翻转集质量下降19/22/18条、改善18/18/21条、同分100/99/94条。不能把小总体均值解释为每个翻转请求质量差很小；全体稀疏翻转、同分与正负抵消共同作用。All-L仅离线质量参照，不生成服务/碳反事实。

增量c_inc由每输入约0.010117g增至0.013663g，FR active电量份额下降约9.951pp；active加权强度42.237→57.231g/kWh。固定idle未变；实际地区×小时碳表未变，改变的是执行电量权重。线性r参考均值99.21%不能推翻真实50%增长点总账，亦不能据两个点反推准确实际临界率。

## 3. 新文件和审计链

五个新交付分别为Final中文报告、Additional Analysis CSV、Final英文提纲、本修复记录和Final Validation。计算脚本与逐seed支持表在final_review，包含翻转请求配对明细、两层组别、地区/小时active、反弹和组成分解。{checks['long_csv_values_checked']}条长表数值通过独立核对；空翻转子组均值保留NA，不填0。新增Score下降阈值0.10仅为报告约定。

旧报告和原数值不覆盖；首次review前hash快照与收尾hash对照证明保持。未运行任何原C1–C8、CVaR、DRO、LHS、大型联合测试或All-L DES。没有为了美观调整原数据。

## 4. 未解决问题与论文停点

没有阻止**条件仿真论文写作**的Critical或Important问题。目标硬件/PUE/真实请求/边际排放/人口公平证据仍不足，作为明确研究局限保留，不伪装成已验证。3seed不支持显著性或可靠95%推断，真实行为反弹未识别。

**TASK 4 READY FOR PAPER WRITING**。本轮到审查与材料修订止，不生成英文论文全文。
''')

    new('Task4_Paper_Outline_Final_EN.md',fr'''
# Task 4 Final Paper Outline — Scenario-Based Robustness and Sustainability Assessment

Status: **TASK 4 READY FOR PAPER WRITING**. This is a 2–3 page outline, not final prose. All new statistics reuse the same 24 saved executions; no new DES, training, All-L simulation or large study. Four scenarios × three paired seeds × Greedy/Spatial; keep seed-level means and observed ranges only.

## 1. Scope, scenarios and quality definitions (~0.35 page)

Freeze Task 2 Pre-LightGBM/risk-guard choices, rerunning its rule after perturbation, and use the final Task 3 Spatial MILP + DES. Greedy shares routing, exogenous world and six powered hosts. H=43,200s. Name the composition stress **Higher Share of Math/Code Requests**: it is a task/computational-load proxy, not proof of higher difficulty for every query.

Scenarios: Baseline N0=7,335; Math+Code share 40% with original relative ratio; independent S/L prediction noise epsilon~N(0,0.10²) before clipping, shared within prompt groups; demand N1=N0+floor(0.5N0)=11,002. Preserve base IDs/groups and never treat replay copies as independent quality labels.

Definitions to include: Q_router=mean(q_selected), P_hist=mean(1[q_selected≥0.5]); I=completed AND deadline/wait compliant, P_valid=mean(I·1[q_selected≥0.5]), V=mean(1−I). P95 uses completions only and must be shown with V/completion.

## 2. Robustness results and prediction flips (~0.65 page; Figure 1)

Use the existing Figure1_Robustness.png with a caption correcting the old “Difficulty” shorthand. Baseline Spatial V=0.0182%, completed P95=7.827s. Higher Math/Code: Q 0.776142→0.748728, historical pass 83.817%→84.872%, service estimate sum 8,930.01→13,114.63s, P95 11.967s, V remains 0.0182%. Explain that Math has lower mean score but higher threshold-pass probability than General; composition can move the two metrics in opposite directions.

Prediction flips by seed: L→S=67/75/68, S→L=70/64/65, totals137/139/133 (1.868%/1.895%/1.813%), all General. On the same changed-request subset, mean paired score change=−0.016423/−0.026978/+0.016917; score declines≥0.10 affect19/22/18, and new S loses an L-passing answer for13/15/12. Approximately{equal_fraction:.2f}% of flips have identical S/L scores; remaining gains/losses offset. Small population mean change is not evidence of harmless individual decisions. No no-guard ablation or recalibration was run.

Growth: Spatial V rises to0.1697%, P95 to11.885s, completion99.8303%; same-demand Greedy V=1.6270%, P95=6.998s. Spatial offers fewer violations but a higher completed-sample P95; mention different completed sets/queue trade-offs rather than claiming universal improvement.

## 3. Full-facility carbon and growth (~0.5 page; Figure 2)

Keep C_total=C_idle+C_active+C_net+C_ctrl with common H and no idle removal. Baseline G=41,288.178369g, S=41,214.556014g; saving73.622355g (0.1783% total,49.80% operational increment). Baseline Spatial uses0.004857kWh MORE, so energy savings and carbon savings differ. Fixed idle=250.56kWh/41,140.3512g, about99.82% of baseline Spatial carbon.

Main growth result: S(N1)=41,290.674181g; S(N1)−G(N0)=+2.495812g, approximately{rel_delta:.6f}% of the original total. Same-demand growth saving remains101.442949g. These contrasts answer different questions. This tiny cross-demand increase is conditional on power/PUE/carbon assumptions and is not a real-world significant or causal estimate.

If space permits, add only a subordinate theoretical reference:

$$
c_0=\frac{{C_S(N_0)-C_{{idle}}}}{{N_0}},\qquad
r_{{lin}}=\frac{{C_G(N_0)-C_S(N_0)}}{{C_S(N_0)-C_{{idle}}}}.
$$

Do NOT headline99.21% as the DES rebound threshold. It is an algebraically correct constant-marginal reference, but actual per-input incremental carbon increases from about0.010117 to0.013663g. FR active-energy share falls89.583%→79.632%; execution-weighted intensity rises42.237→57.231g/kWh, so the linear assumption fails. Only r=0 and≈0.5 were simulated: no accurate nonlinear crossing, monotonicity, or efficiency-induced behavioral demand was identified. Figure2_Carbon_Growth.png compares G/S within demand and growth Spatial against original Greedy across demand.

## 4. Two-layer task audit (~0.5 page; Figure 3 + compact quality table)

Routing layer: ΔQ_g=mean(q_selected−q_L), paired within the same historical request; show Router/All-L pass difference and S/L share, unique groups. All-L is an OFFLINE quality reference, not an executed service counterfactual. Baseline General Q_router=0.812466, Q_AllL=0.818424, ΔQ=−0.005958 and pass difference−0.5055pp. Math/Code always choose L, giving exactly zero routing-reference differences, not perfect absolute quality. L is not better on every query: among431 selected General S answers,41 score above L and74 below L.

Scheduling layer: Figure3_Fairness.png is Spatial−Greedy violation in percentage points; negative favors Spatial. No positive violation difference occurs in the36 task×scenario×seed comparisons. Still show completion, completed P95 and valid-pass rate: Spatial has higher P95, and absolute growth risk is worst for Code (Spatial violation≈0.4919%). Code has86 baseline instances but85 distinct prompt groups, so duplication never strengthens independent quality evidence. Do not equate slower Math, low absolute benchmark scores, or no observed relative service loss with demographic fairness or absence of all harms.

## 5. Evidence/limits and conclusion (~0.35 page)

Use one compact Evidence and Limitations table, separating historical public grid observations, RouterBench paired scores, transferred public Qwen/B200 measurements, assumed idle/shared/PUE/service parameters, DES outputs and missing deployment evidence. No target-model isolated power, real site PUE, representative public-sector trace, as-of forecast vintage or marginal-grid causal counterfactual was measured. Average carbon accounting is attributional; environmentally beneficial deployment is not certified.

Conclude: the stated task-mix and prediction stresses retain high conditional completion, while demand growth exposes congestion and a small total-carbon reversal; aggregate averages conceal local routing quality loss. Scheduling reduces carbon at matched demand without necessarily reducing electricity or completed P95. Claims remain tied to fixed assumptions and three seed-level observations; no significance,95% coverage or universal robustness/fairness claim.

Trace every number to Task4_Additional_Analysis.csv (unit/denominator dictionary in final_review/statistics_schema.json), the Final Chinese report and Final Validation. Existing three figures remain unchanged; do not add plots or a new research chapter. The linear reference is optional supporting algebra, not the paper's principal empirical conclusion.
''')
    print('Wrote new Final Chinese report, fixes and English outline; original files untouched')


if __name__=='__main__':main()
