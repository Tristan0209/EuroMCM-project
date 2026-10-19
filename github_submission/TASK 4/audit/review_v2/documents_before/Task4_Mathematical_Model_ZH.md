# 第四问数学模型说明

日期：2026-10-18。**本文全部第四问模型、分布、参数区间、阈值与实验均尚未验证；仅为实施规范。**第三问继承约束与本轮手算检查另有标识，不把评价指标说成已经加入调度器的优化约束。

实施基线与实际接口见[中文实施方案](<D:/EuroMCM/EuroMCM-project/TASK 4/Task4_Implementation_Plan_ZH.md>)；执行门槛见[清单](<D:/EuroMCM/EuroMCM-project/TASK 4/Task4_Execution_Checklist_ZH.md>)。

## 1. 样本单位、变量和实验层级

| 符号 | 定义 |
|---|---|
| g / i | 规范化prompt group／执行实例。重复同一g不创造新的独立质量观测 |
| i0 / instance_id | 历史base_request_id／本情景实例唯一ID，保留二者映射 |
| k / b / π | 参数或固定压力场景／独立外生世界／调度策略 |
| θ / ω | 不确定参数／在给定参数下实现的到达、误差、服务与故障随机轨迹 |
| N, a_i, D_i, W_i | 世界内输入数、到达、完成截止、额外等待上限（秒） |
| m_i, p_i, Z_i | 重执行第二问后本场景固定的S/L、事前名义服务时长、隐藏的均值1服务扰动 |
| o_i, r_i, d_i, v_i, f_i | origin、实际执行region、dispatch、服务start、completion |
| C_r(t), Ĉ_s(t,r) | 环境actual碳强度／在s发布的预测，单位gCO2e/kWh |
| H | 公共观察窗口43200秒，UTC锚点默认2025-01-02T00:00:00Z |
| q_iS, q_iL | 既有benchmark score，只用于事后评价，不是调度特征 |
| A_rm(t), K_rm | 当前可准入槽数／安装并配置的准入上限，故障不删除idle |
| I_i, V_i | 截至H完成且合规交付指标／全输入服务违反指标 |
| C_πb, E_πb | 世界b在固定H内策略π的总设施碳与电量 |

分层估计对象为 `Y(π,θ,ω)`。M2的八个逆境配置是固定场景，不能将其最大值称为概率95%的尾风险。M3的world独立抽取才用于MC分布估计。M6的LHS是对参数假设分布的分层探索，不能与M3按同一概率总体混合。重复seed、重复配置引用、同prompt复制、同seed跨月份都不额外增加独立样本量。

## 2. 冻结Pre推断与合法预测误差实验

### 2.1 继承的第二问规则（实际实现已核对）

设两个冻结Pre预测为 `q̂_iS,q̂_iL∈[0,1]`，冻结训练成本模型为

\[
\widehat c_{im}=(1,\operatorname{chars}_i/1000)\beta_m,
\quad \Delta\widehat c_i=(\widehat c_{iL}-\widehat c_{iS})/c_0.
\tag{4.1}
\]

当前最终家族为 Pre_direct，General的决策为

\[
h_i=(\widehat q_{iL}-\widehat q_{iS})-.5\Delta\widehat c_i,
\quad m_i=\begin{cases}L,&h_i\ge -.617099253690552,\\S,&h_i<-.617099253690552.\end{cases}
\tag{4.2}
\]

Code、Math和未注册benchmark先回退L；Post阈值为null，不执行S→L cascade。式(4.2)是实际冻结决策，不是第四问拟合的新模型。`q_hat`与真实`q`必须不同列、不同权限。

### 2.2 拟议Pre失准机制（尚未验证）

对每个group预生成误差，例：

\[
\widetilde q_{im}=\operatorname{clip}_{[0,1]}
\{\widehat q_{im}+\beta_m+\sigma_q(\sqrt{\rho_q}u_g+\sqrt{1-\rho_q}\epsilon_{gm})\},
\quad u_g,\epsilon_{gm}\sim N(0,1).
\tag{4.3}
\]

独立group用独立误差；重复实例同一g共用预测误差，服务仍可不同。主失准方向用gain bias `β_Δ=β_L−β_S`，实现时取`β_L=β_Δ/2, β_S=−β_Δ/2`；负bias可能增加S调用。另报告正bias及对称噪声。clip使边界附近误差不再均值零，必须报告实得gain bias、RMSE和clip比例。误差仅加到预测，真实q和历史答案不变。

重新执行式(4.1)–(4.2)或同义`routing.apply`，保留原fallback，得到情景action，再按新action取对应服务与事后质量。`score`、`mu`必须同步记录；对Pre_direct而言std=0，不额外创造置信区间。零误差逐ID应恢复S431/L6904。直接改写冻结标签或按最终损失挑选flip目标不属于合法实验。

### 2.3 风险guard的适用边界

继承的组级损失定义为

\[
X_g=\max_{i\in g}(q_{iL}-q_{im_i})_+,
\quad H_g=\max_{i\in g}\mathbf1\{q_{iL}\ge .5,\ q_{im_i}<.5\}.
\tag{4.4}
\]

confirm按任务分组，有n个独立组、h个harm组时使用Clopper–Pearson上界与经验Bernstein上界，加声明的重复质量限d_t：

\[
U_t^{harm}=\min\{1,\operatorname{Beta}^{-1}(1-\alpha;h+1,n-h)+d_t\},
\quad(h=n\Rightarrow U=1),
\]
\[
U_t^q=\min\{1,\bar X+\sqrt{2s_X^2\log(2/\alpha)/n}
+7\log(2/\alpha)/(3(n-1))+d_t\}.
\tag{4.5}
\]

n<2按实现保守返回1。零harm不是零风险。原实验个别alpha=.05/(10×3×2)，总量损失预算.02、任务均值损失.05、任务pass损失.02；这些是研究协议，不是题目法定门槛。对非fallback任务要求任务上界达标，且声明混合集合中的最大加权总量上界达标。

修改预测函数、任务比例或重复质量后，原校准覆盖不自动成立。拟议新校准若事前固定K个扰动政策、F个家族、T任务与2类损失，统一预算至少按`.05/(KFT×2)`分配；不能看完test再挑扰动/预算。权重和阈值固定在train/select侧，confirm只计算上界/选已覆盖的fallback；未知来源仍L。共同全局偏差可作为固定场景条件化，group独立性需在该条件下说明。任意对手根据confirm/test真实q挑选错误的压力场景不享有该保证，只作攻击性描述。当前不执行式(4.5)的新校准。

## 3. 困难请求、需求和联合外生世界（尚未验证）

### 3.1 困难性必须可观测或明确为事后分析

主要困难混合用task组成：`π_hard=π_Code+π_Math`，原比例1796/7335，新增压力点.4/.6/.8；内部Code:Math比例保持原比例。General内部可用train分位数定义的低`q̂_S`或大预测gain分箱辅助审计，不用test q定义在线困难特征。task只是困难性的代理，Math并不意味着每题都更难。

实现时按test组构造固定N的分层重采样/重加权回放，并追踪同一group复制；Code仅86行，不能通过复制宣称新增独立Code证据。真实q定义的“难题子集”只允许事后、明确选择条件的辅助描述。真正未见的新困难题和未知领域质量需新标签，不允许把原q减常数模拟成真实观测。

### 3.2 三种负载变化分别定义

到达压缩：

\[
a_i'=a_i/\ell,\quad D_i'=a_i'+(D_i-a_i),\quad N'=N.
\tag{4.6}
\]

deadline相对预算与class保持，等待预算另有压力因子时才变化。`load_inputs(load=ell)`属于此类。它改变高峰拥塞但不增加12h内的总请求量。

burst：令原相对到达分位数u映射为`a'=F_burst^{-1}(u)`，密度在预定15min段乘3/5/10后归一化；同N、同首末时段、固定H。用于突发形状，不称总需求反弹。

数量增长：在同一到达支持区间`[0,T_A]`，`T_A≈15087.638s`，

\[
N(r)=N_0+\lfloor rN_0\rfloor,
\quad \lambda_r(t)=(1+r)\lambda_0(t),\quad r\ge0.
\tag{4.7}
\]

r=0保留全部原ID/顺序；r>0追加唯一实例ID，base_id/group用于连接已有两动作结果。额外到达按原经验形状抽样，可预先生成r_max的实例全集再取固定前缀；各r共享已有实例及其潜在随机量，不按策略重新抽。全体按arrival/稳定ID排序。机制A保持task/origin/class分布；机制B新增请求来自预定偏移混合，二者单独报告。历史回答score复用只代表同任务回放，不代表新质量样本。

### 3.3 随机服务与碳预测边界

\[
P_{ib}=p_i s_{m_i}Z_{ib},
\quad \log Z_{ib}=-\sigma_T^2/2+\sigma_T\eta_{ib},
\quad E[Z]=1.
\tag{4.8}
\]

p_i已含原L倍数3，`s_m`是新敏感性倍率，不再乘3。`s_m`若表示名义服务整体变化，规划与环境都用`p_i s_m`；若表示规划失准，必须另用`P=p_i s_m^actual Z`、规划p_i不变的显式实验，禁止混同。Z仅在服务准入时读取。尾部可选task/时段共同冲击应另列模型，不能声称原lognormal已经包含相关慢服务。

碳持久化从最近完整小时取值：

\[
h_s=\max\{h:t_h+3600\le s\},\qquad \widehat C_s(t,r)=C_{r,h_s}.
\tag{4.9}
\]

拟议误差场按外生发布网格`j=floor(s/3600)`固定，不按策略issued精确值重抽：

\[
\widetilde C_s(t,r)=\widehat C_s(t,r)
\exp\{b_r+\sigma_C(\sqrt{\rho_C}u_j+\sqrt{1-\rho_C}\epsilon_{jr})-\sigma_C^2/2\}.
\tag{4.10}
\]

误差场与未来actual独立生成；actual用于记账，不能从actual未来减出“预测残差”再喂入调度器。设`ρ_C=.5`为假设并保留0/.9附加检查。偏差b_r按区域设置，FR正bias/DE负bias可破坏空间排序；common-only乘法在纯active目标中可能不改排序，不能把无影响误称鲁棒。Persistence仍无同次issue的未来下降信息，第四问不新增时间优化收益主张。

### 3.4 M1精确定义：22个配置

以下七因素各取三个非基线水平，其他固定基线，再加单独基线，共22个设置；全部是拟议压力范围。

| 因素 | 基线 | 三个非基线水平 |
|---|---|---|
| 困难份额π_hard | 1796/7335 | .4/.6/.8，固定N重采样 |
| 到达压缩ell | 1 | .5/2/4，N不变 |
| burst形状 | 无 | 15min密度倍率3/5/10，归一化保持N |
| Pre误差 | βΔ=0, σq=0 | (-.05,.05)/(-.10,.10)/(+.10,.10)，ρq=.5 |
| 碳预测误差 | b=0, σC=0 | σC=.1/.2/.4，b_FR=+σC、b_DE=−σC、b_ES=0 |
| 服务波动 | σT=.15 | .30/.50/.80，均值仍1 |
| 容量故障 | normal | FR-L在t3600停用5min／30min；或全部地区L池t3600同时停用30min；均保持供电并恢复 |

M2的八组合按实施方案的顺序确定；严重非基线采用hard=.6、ell=2、burst5、Pre(-.1,.1)、碳σC=.4及区域bias、σT=.5、FR-L30min故障；第七组合用所有L池同时30min；第八用这些严重参数并取所有L池相关故障。组合结果只解释这组设定。

### 3.5 M3概率分布：可审查的假设，而非拟合现实

先抽共同regime `R~Bernoulli(.2)`，再在regime条件下抽参数；每个world使用同一R驱动多类压力，产生联合相关性。`.2`是设计权重，**没有实测发生率依据**，需同时报告逆境权重0/.1/.2/.4的重加权敏感性（若使用已生成分层样本，应保留设计抽样概率与目标权重）。

| 随机量 | R=0（设计权重.8） | R=1（设计权重.2） |
|---|---|---|
| ell | U(.8,1.2) | U(1.2,2) |
| π_hard | U(原比例,.4) | U(.4,.6) |
| βΔ, σq | U(-.03,.03), U(0,.05) | U(-.10,0), U(.05,.10) |
| σC；b_FR,b_DE | U(0,.15)；+σC,−σC | U(.15,.4)；+σC,−σC |
| σT | U(.10,.25) | U(.30,.50) |
| 15min burst | 概率.1，倍率3 | 概率.5，倍率5 |
| outage | 概率.05，单FR-L | 概率.5；其中.25为所有L池共同停用 |
| outage起点/时长 | U(1800,10800)/U(300,900)s | U(1800,10800)/U(900,1800)s |

N固定7335，数量反弹另在M4；这样M3首先隔离失准/结构/突发的系统风险。条件内未声明的随机量独立；服务同实例配对、碳区域相关和共同regime相关按上述保留。采用概率分布只是定义“在此假设下的风险”，不能升级为现实频率。跨月作为固定分层另报，不把挑选的极端月份当独立随机抽样。

## 4. 继承Spatial与DES：评价扩展不改冻优化模型

每个决策s，已到达EDF批次不超过64；m_i在该情景固定。地区与slot候选必须满足允许链路、模型兼容、当前容量及名义latest。最终Spatial的ready、tail、latest、非抢占顺序约束、量化碳与未分配罚项完全继承第三问(3.8)–(3.15)。u是名义排队证书，执行FCFS可提前或推后；z=1进入fallback，不自动从输入集合删除。

执行恒等式为

\[
ready_i=d_i+n_{ir_i},\quad w_i^{pre}=d_i-a_i-h,\quad
w_i^{queue}=v_i-ready_i,\quad
f_i-a_i=h+w_i^{pre}+n_{ir_i}+w_i^{queue}+P_{ib}.
\tag{4.11}
\]

`w_pre`含聚合、计算和持有，不全部叫主动碳等待。额外等待门槛是`w_pre+w_queue<=W_i`，完成门槛是`f_i<=D_i`。故障仅改变准入，运行不中断；同刻先处理容量变化。

### 固定H的溢出处理（新观察器尚未实现）

给每实例状态`completed / failed / censored_at_H`。若H时仍运行/排队/未交付，标censored，服务违反=1；已开始服务以`[v_i,min(f_i,H))`记busy，不能只因未完成就删active。完整未截尾f_i仅可由环境保存为私有事件信息，不回传调度器。所有输入事前应`a_i<H`；不满足的场景设计本身拒绝，不能计为已到达需求。对本来可在H内排空的世界，新观察器必须逐事件和旧账本相等。

冻结引擎当前会在未来事件超过H时报错，不能直接充当该观察器；开发通过前不产出完整MC/CVaR。不能删除溢出world或按每策略扩展H。若需要24h辅助观察窗口，应事前作为独立协议对全部配对策略与同一网格统一使用，不与12h结果拼接。

## 5. 电量、碳与交付损失的统一边界

设物理host集合为\(\mathcal H\)，host h位于r(h)，安装GPU数G_h，每host共享功率P_sh,h、GPUidle P_id,h、GPUbusy P_bu,h及PUE φ_h。相同host内b_h(t)个GPU执行时

\[
P_h^{IT}(t)=P_{sh,h}+G_hP_{id,h}+b_h(t)(P_{bu,h}-P_{id,h}).
\tag{4.12}
\]

\[
E_{idle}=\sum_h\frac{\phi_h(P_{sh,h}+G_hP_{id,h})H}{3.6\times10^6},
\quad E_{active}=\sum_h\frac{\phi_h(P_{bu,h}-P_{id,h})}{3.6\times10^6}\int_0^H b_h(t)dt.
\tag{4.13}
\]

碳用同一分时积分在各项内乘`C_r(t)`，不拿全年度均值代替。baseline六host、8GPU/host、1500/175/760W、PUE1.2得到idle250.56kWh，busy增量585W；这是已核对的输入口径，非硬件测量。

\[
E_{net}=\sum_{i:dispatch_i<H}e_{io_ir_i},\quad
C_{net}=\sum_i\frac{e_{io_ir_i}}2[C_{o_i}(d_i)+C_{r_i}(d_i)],
\]
\[
E_{ctrl}=\frac{P_{ctrl}\phi_{ctrl}}{3.6\times10^6}
\sum_{k}\left|[s_k,s_k+L_k)\cap[0,H)\right|,
\quad E_{total}=E_{idle}+E_{active}+E_{net}+E_{ctrl},
\]
\[
C_{total}=C_{idle}+C_{active}+C_{net}+C_{ctrl},\qquad C_{op}=C_{total}-C_{idle}.
\tag{4.14}
\]

网络输入电量是既定设施代理，不再乘PUE；本地两端相同也只合成一笔e。已dispatch后失败仍记network，已计算仍记controller，额外等待不是额外计算。`estimated_energy_kwh`仅为名义busy增量，绝不加到实际账本。

旧`evaluate`用`controller_w×1.2`，LHS改变PUE时需要TASK4显式φ_ctrl。若所有设施统一φ，应同步规划成本、服务器核算和controller；网络依其独立边界。只在固定轨迹上重计PUE称“accounting-only敏感性”，不能替代改变成本系数后的闭环DES重算。

### 交付与质量指标

\[
I_i=\mathbf1\{completed,\ f_i\le D_i,\ w_i^{pre}+w_i^{queue}\le W_i\},
\quad V_i=1-I_i,
\quad v_\pi=N^{-1}\sum_iV_i.
\tag{4.15}
\]

另报completion、failure、censoring各率，防止把晚完成误记成未完成。只报completed响应分位数会选择性忽略失败，故配全输入损失

\[
\ell_i^{time}=\begin{cases}
\min\{(f_i-a_i)/(D_i-a_i),2\},&截至H完成,\\
2,&failed\text{或}censored,
\end{cases}
\quad Q_\pi=N^{-1}\sum_i(q_{iL}-I_iq_{im_i})_+,
\]
\[
H_\pi=N^{-1}\sum_i\mathbf1\{q_{iL}\ge .5,\ I_iq_{im_i}<.5\},
\quad U_{valid}=\sum_iI_iq_{im_i}.
\tag{4.16}
\]

时间截断常数2是预先定义的效用损失，不把失败伪造成实际2倍延迟；同时单报实际完成延迟和失败数。Q衡量有效交付相对“假设All-L答案可用”的质量机会损失，包含未交付惩罚；若需比较可执行All-L系统，用P2的实际I_iL另报配对效用差，不把P2永远按时交付作为事实。

纯模型选择质量也另报`mean(q_L−q_m)`、positive loss和pass harm，以区分路由质量与排队失败。历史score冻结不等于响应交付质量恒定，因为I_i可以变化。

## 6. CVaR极端风险：独立于调度目标的评价模型（尚未验证）

主报损失向量：

\[
L^V_{\pi b}=v_{\pi b},\quad L^Q_{\pi b}=Q_{\pi b},\quad
L^C_{\pi b}=\frac{C_{\pi b}-C_{P0,b}}{100\,\mathrm{g}}.
\tag{4.17}
\]

正的碳损失表示相对同world、同需求、同硬件P0增排；另报其正部和绝对C_total的尾部，不能只报增量。参数改变idle时比较仍要完整facility总量。M4反弹的固定需求基线另定义，不能混用(4.17)。

辅助复合损失可预注册为

\[
L^{mix}=.4\,v/.001+.3\,Q/.02+.3\,(L^C)_+.
\tag{4.18}
\]

`.001`、`.02`、100g及.4/.3/.3都是建议的研究量纲/权重，**尚未验证且非题目标准**。分别报告各分量及等权/服务优先权重的敏感性；不依测试结果挑一组权重宣布胜出。SLA和质量可采用门槛式排序，不能让碳收益自动抵消不合格服务。当前不把CVaR项加进冻结Spatial优化目标；任何风险厌恶调度器另立未验证改进实验。

对已声明世界分布，定义

\[
\operatorname{VaR}_{\alpha}(L)=\inf\{x:F_L(x)\ge\alpha\},
\quad
\operatorname{CVaR}_{\alpha}(L)=\min_{\eta\in\mathbb R}
\left[\eta+\frac{E(L-\eta)_+}{1-\alpha}\right].
\tag{4.19}
\]

有限情景概率`w_b>=0, sum w_b=1`下以

\[
\min_\eta\left\{\eta+\frac1{1-\alpha}\sum_bw_b(L_b-\eta)_+\right\}
\tag{4.20}
\]

计算，自动包含VaR处所需的分数质量。不能简单平均所有`L>=VaR`，因为离散重复值和分数尾部会改变尾质量。手算`[0,1,2,3], α=.625`的CVaR为8/3；本轮只验证了此类公式实现，不产生系统CVaR结果。

主水平α=.95，1000独立world约50尾部当量；α=.99的1000world约10尾部当量，只作不稳定诊断，拟议5000world才约50当量。所有报告同时显示B、尾质量、分布假设和配对bootstrap区间。50不是统计保证，零损失样本不代表零尾部风险。

比较策略时分别算`CVaR(L_P1)−CVaR(L_P0)`，每次bootstrap按world成对重采样并重算两边CVaR；它**不等于**`CVaR(L_P1−L_P0)`，后者只作为配对损失差的另一指标。固定θ多服务seed时先估条件CVaR；若外层θ也随机，按其真实抽样层级重采样。预定逆境/不完备运行不从风险分布删除；窗口/软件未支持导致账本缺失时该批不通过发布门槛，不用0替代未知损失。

## 7. 需求反弹与容量扩展（尚未验证）

### 7.1 两种比较对象

在相同r、硬件配置k和世界b下，

\[
\Delta C_{sched}(r,k,b)=C_{P0}(r,k,b)-C_{P1}(r,k,b)
\tag{4.21}
\]

识别调度差异。固定需求反事实基线为

\[
C_{ref,0,b}=C_{P0}(r=0,k_0,b),\quad
G_C(r,k)=E_b[C_{ref,0,b}-C_{P1}(r,k,b)].
\tag{4.22}
\]

这是“采用后增长能否抵消相对于既有系统的环境收益”。如果比较整个路由系统，则另外用All-L+Greedy的固定需求基线P2，明确更换比较对象；不可把调度0.178%的改善等同整个路由的效率改善。电量G_E同理。已知第三问P1电量略高，不能预设存在正的“节电反弹空间”。

### 7.2 条件线性诊断，不当作DES结论

仅在固定idle、相同地区／服务混合、无拥塞／失败变化且边际碳恒定时，

\[
C_{P1}(r,k_0)\approx C_{idle,P1}+(1+r)C_{op,P1}(0),
\quad
r_C^{lin}=\frac{C_{ref,0}-C_{idle,P1}}{C_{op,P1}(0)}-1.
\tag{4.23}
\]

这个公式只能作一致性/数量级检查，不能把所有C_total按请求数线性乘，因为idle固定。若r=0已经增排，则不存在正起点的收益空间；若分母为0需单独处理。新增host、碳时段转移、fallback与控制成本都可破坏该近似。

闭环阈值定义为从r=0的可接受区间开始首次失去环境收益的边界：

\[
r_C^*=\inf\{r\ge0:G_C(r,k(r))\le0\}.
\tag{4.24}
\]

同时要求服务预算`v<=v_max`及有效质量门槛；服务先失效时报告`r_SLA`与环境边界分别存在，不以增排阈值代替业务可用界。若r=0不合格则不存在已证实的起始可用区间。

先跑r=0/.25/.5/1/2/4并绘配对绝对碳差，若区间端点有稳定异号再细化。DES存在离散扩容/拒绝/排序变化，不假设单调，不盲目二分。报“在r_a与r_b之间出现采样支撑的交叉”、置信不确定区间或“测试范围内未定位”；网格之外不外推。用新seed确认探索出的候选根区间。

### 7.3 slots与新增host的区分

配置k为：原6host、48安装GPU，每host允许1/2/4/8slots；或者每地区L有2/3host、S仍1host、每host1slot。后两者总host=9/12，总安装GPU=72/96，总slots=9/12。host增加使idle随位置变化，slots增加保持原idle。每host需满足

\[
0\le A_h(t)\le K_h\le G_h=8,\quad
P_{sh,h}+8P_{bu,h}\le14300\mathrm W.
\tag{4.25}
\]

现有`load_inputs`是一host一pool；新多host适配器需另建host/pool表。相同硬件可聚合pool的idle_base为各host之和、capacity为slots之和、每job增量保持一GPU；不能聚合完再乘capacity收费。安装容量与实际可用容量不同。

先对固定配置逐一比较，再研究假设扩容规则`k(r)`。规则可以按训练/校准时段估计的利用率目标或合同容量配置，不能看测试实际未来完成时刻选择最低碳配置。

若额外引入价格／等待反弹弹性，可用

\[
1+r=(c_{new}/c_{old})^{-\epsilon_c}(T_{new}/T_{old})^{-\epsilon_T},
\quad \epsilon_c\in[0,2],\ \epsilon_T\in[0,1]
\tag{4.26}
\]

作**未验证假设情景**。这些不是估计值；用事前预测/校准的T或显式有滞后的需求反馈模型，不用同一测试窗真实T瞬时决定其自身到达。费用USD API与EUR自建不直接相加，无实测价格弹性不能报告因果反弹。主实验先用外生r，避免未识别的反馈参数掩盖调度结果。

## 8. 分组公平性审计（尚未验证）

预定组：Code/Math/General、benchmark、service_class、模拟origin，train-only长度与Pre难度分箱；主交叉组为task×class，其他交叉组先列样本量再决定只描述或估计。q、执行region、是否完成等结果变量不能定义事前主组。region公平性用origin；按实际执行region分组属于策略选择后的机制分析。

对组g，以**所有输入**为分母：

\[
v_g=\frac{\sum_{i\in g}V_i}{N_g},\quad c_g=\frac{\sum_{i\in g}\mathbf1(completed_i)}{N_g},\quad
u_g=\frac{\sum_{i\in g}I_iq_{im_i}}{N_g},
\quad h_g=\frac{\sum_{i\in g}\mathbf1(q_{iL}\ge.5,I_iq_{im_i}<.5)}{N_g}.
\tag{4.27}
\]

另报组内模型选择positive loss，分清原L本就不通过与路由新增损害。拒绝/失败同时产生低碳时必须呈现其u_g和c_g。

\[
D_{abs}=\max_g v_g-\min_g v_g,\quad
R_{valid}=\min_g (1-v_g)/\max_g(1-v_g),\quad
\delta_g=v_{P1,g}-v_{P0,g}.
\tag{4.28}
\]

分母0记NA，不能用任意epsilon伪造“高比率”。比例极小时同时报百分点差及置信区间，不仅报夸大的relative ratio。对等待用`(w_pre+w_queue)/W_i`；W_i=0则任何正等待直接违反，归一化量另记NA。完成样本P95与全输入time-loss/失败率并列。

为控制不同任务的时效组成，使用预先规定的共同class权重w_c：

\[
v_g^{std}=\sum_c w_c v_{g,c},\qquad \sum_cw_c=1.
\tag{4.29}
\]

空单元不能填0；标不可估计／仅在共同支持集上标准化并说明权重改变，不默默借邻组数据。报告原始差与标准化差，避免辛普森效应。

推断按world配对并对prompt group作聚类处理：固定历史cohort下的服务随机性用world bootstrap；希望推广至题目总体时另加group层重采样，且原有旧test不能因此称新holdout。仅86条Code、更多交叉组更稀疏，优先效应与区间；进行多重组检验用预注册Holm校正。建议unique group<30的交叉组只描述，阈值30本身为未验证的报告约定，不能证明统计功效。缺少人口属性时不下人口公平结论；不同task的equal S/L rate不是公平目标，因其能力需求不同。

## 9. LHS参数范围、环境收益敏感性（全部范围尚未验证）

### 9.1 分层抽样

对参数维度j、n点LHS：

\[
U_{kj}=(\pi_j(k)+\xi_{kj})/n,\quad\xi_{kj}\sim U(0,1),
\quad\theta_{kj}=F_j^{-1}(U_{kj}),\ k=0,\ldots,n-1.
\tag{4.30}
\]

π_j是独立随机排列。2套独立scramble各128点，每点相同8个服务seed；主参数LHS与服务随机子流独立。`Tri(a,mode,b)`表示三角分布，`LogU(a,b)`表示对数均匀；均为研究者假设，不是测量置信分布。

| 维度／单位 | 基线 | 拟议分布与范围 | 如何作用／数据依据边界 |
|---|---:|---|---|
| host shared W | 1500 | Tri(750,1500,2500) | 沿旧敏感性范围作为假设；改host idle及busy aliases共同平移 |
| GPU idle W | 175 | U(100,250) | 改idle与busy增量，须闭环重算；旧范围不是最终置信区间 |
| GPU busy W | 760 | Tri(500,760,1000) | 新扩展范围，非S/L实测；核对busy>=idle及峰值 |
| PUE | 1.2 | Tri(1.1,1.2,1.6) | 同步服务器、controller与规划；network边界独立 |
| S服务倍率s_S | 1 | Tri(.7,1,1.5) | 乘已存在的S服务代理，假设 |
| L服务倍率s_L | 1 | Tri(.7,1,1.5) | 乘已含3倍的L代理，不能再乘3，假设 |
| log服务sigma_T | .15 | U(.05,.5) | 只改变均值1扰动分布，非真实误差估计 |
| actual碳共同尺度a_C | 1 | U(.8,1.2) | 同比例处理历史环境并由其过去重建预测，假设会计/系统情景 |
| 地区对比度d_C | 1 | U(0,1.5) | 用下式改变空间差异，不称新历史数据 |
| 网络就绪延迟倍率 | 1 | LogU(.5,3) | 乘完整(.05+latency_ms/1000)，与原load_inputs只缩放ms不同，需新适配器 |
| 网络电量倍率 | 1 | LogU(.1,10) | 两端分摊，不再乘PUE，缺乏真实传输能耗 |
| controller W | 10 | LogU(5,50) | 增量计算功率，非另一主机idle |
| router seconds | .01 | Tri(.005,.01,.05) | 控制延迟与计算工作同步，固定proxy而非实测wall |
| 碳预测sigma_C | 0 | U(0,.4) | 使用外生时空误差场；区域bias按第3节另记录 |
| Pre sigma_q | 0 | U(0,.1) | 改预测后重执行TASK2，原guard覆盖未建立 |

\[
C'_r(t)=a_C\max\{0,\bar C(t)+d_C[C_r(t)-\bar C(t)]\}.
\tag{4.31}
\]

clip会改变三地区平均值，需记录clip比例；d_C=0消除同小时空间差异。式(4.31)是环境反事实，预测只能重新读取C'过去；只改预测属于另一类失准实验，不能混为一谈。CO2e species/boundary不变。

gamma是决策安全参数，主LHS固定1.35；host数与slots是容量政策，留在M4，不伪装成连续物理测量误差。可选策略对照gamma1.2/1.5/1.75属新的**尚未验证**消融，不根据测试最优值回写冻结模型。

### 9.2 重算与分析原则

共同shared仅平移同host的idle与busy aliases、保持busy−idle时，规划成本不变，理想控制轨迹可复用，按新idle重新计账；应先用小样例证明轨迹不变。PUE、GPUidle、GPUbusy、服务、网络、Pre/碳预测误差可能改变分配，必须闭环重算。仅调controller W且无温度/功率影响服务的模型内可固定轨迹重计，需明确条件。不得把所有LHS点都当“已有轨迹乘比例”。

每点报配对ΔC_total、ΔC_op、ΔE、SLA、Q、U_valid及是否净收益/服务都合格。先Spearman/PRCC描述主要方向，再用分箱或局部曲线检查非单调、交互与阈值；PRCC的单调假设失败就不解释为全局因果重要性。两个scramble比较排序与符号稳定性；仅两套设计不足以给稳健随机化置信区间，若需此类区间，应追加更多独立LHS设计并单列预算。不能从LHS的普通行bootstrap声称严格独立MC推断。

参数区间变宽会改变重要性排序，需同时报告分布与范围、相关性假设。无相关数据时主LHS假设参数独立；如后续加入相关性，用显式条件/潜在变量设计并验证覆盖，不在独立LHS完成后随意排序破坏边际分层。16个极端候选点仅作探索，再加独立seed确认；确认不是重新拟合上游模型。

## 10. 统计和结论发布条件

固定场景的配对差`Δ_b=Y_P0,b−Y_P1,b`可按独立world给均值、SD和配对区间；同一个world两策略不拆开抽样。需求曲线的增长实例嵌套、月份重复seed和重复prompt分别追踪依赖。区间说明条件于历史cohort、碳日期和假设参数；不能宣称覆盖真实部署全部不确定性。

环境优势声明至少同时检查：完整总账下碳差方向、服务违反预算与组别损害、有效质量交付、尾部风险、对反弹和扩容的稳健性，以及所用数据的现实代表性。任何未跑情景不写数值；软件失败不变成节能；选择性交付不变成公平。只有完成对应实验并通过执行清单，才将该模块从“尚未验证”更新为具有明确样本范围的实证结论。
