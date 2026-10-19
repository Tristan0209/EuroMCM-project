# 第四问数学模型说明

日期：2026-10-18，审查修订v2。四类状态：**已实现并验证、已实现但未验证、仅设计、环境阻断**。独立Pre_direct全7335逐ID、窗口/核算toy和原始1000/7335单世界等价、质量拆分、公平性及CVaR离散/Bootstrap函数已实现并在注明范围验证。真实扰动压力、多seed/跨月7335新DES、多host/PUE正式场景尚未验证；世界发生器、增长与正式LHS运行仅设计；原完整IntegratedRouter仍环境阻断。全部M1–M8正式实验未执行，所有新分布、预算和部署阈值仍为未验证研究假设。

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

### 3.4 M1扩展范围：22个配置；Core只取8个配置

以下七因素各三个水平构成Extended M1的22设置。Core只取每因素一个代表值加基线：hard=.6、ell=2、burst5、Pre(-.1,.1)、碳sigma=.4及区域bias、服务sigma=.5、FR-L30min故障，共8设置×3seed。全部是尚未执行的压力范围。

| 因素 | 基线 | 三个非基线水平 |
|---|---|---|
| 困难份额π_hard | 1796/7335 | .4/.6/.8，固定N重采样 |
| 到达压缩ell | 1 | .5/2/4，N不变 |
| burst形状 | 无 | 15min密度倍率3/5/10，归一化保持N |
| Pre误差 | βΔ=0, σq=0 | (-.05,.05)/(-.10,.10)/(+.10,.10)，ρq=.5 |
| 碳预测误差 | b=0, σC=0 | σC=.1/.2/.4，b_FR=+σC、b_DE=−σC、b_ES=0 |
| 服务波动 | σT=.15 | .30/.50/.80，均值仍1 |
| 容量故障 | normal | FR-L在t3600停用5min／30min；或全部地区L池t3600同时停用30min；均保持供电并恢复 |

M2的八组合按精简协议C2的顺序确定（ell压缩不是数量增长）；严重非基线采用hard=.6、ell=2、burst5、Pre(-.1,.1)、碳σC=.4及区域bias、σT=.5、FR-L30min故障；第七组合用所有L池同时30min；第八用这些严重参数并取所有L池相关故障。组合结果只解释这组设定。

### 3.5 M3概率分布：可审查的假设，而非拟合现实

目标分布使用共同regime R，基准逆境概率p=.2；条件参数范围沿下表保持。正式采样改为**分层独立抽样**：R=0和R=1各1000个独立world，共2000。每层内到达、误差、服务及故障的world种子独立；同一world跨策略配对。它们是独立条件样本，不能称2000个IID混合分布样本。目标混合分布用w_b=(1-p)/n_0或p/n_1估计。

p=0/.1/.2/.4属于预定敏感性，复用这2000个world，仅重加权，不重新运行DES。p是未验证的设计权重，没有现实发生率依据。该分层分配较原1000world方案更重视各概率下的尾部有效样本，而削减其他网格及LHS来降低总成本。Bootstrap在每层内成对重采样、保持n_0/n_1固定，不把Bootstrap重复当新world。

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

### 固定H的溢出处理（已实现并验证toy与原始1000/7335单世界无截尾等价）

给每实例状态`completed / failed / censored_at_H`。若H时仍运行/排队/未交付，标censored，服务违反=1；已开始服务以`[v_i,min(f_i,H))`记busy，不能只因未完成就删active。完整未截尾f_i仅可由环境保存为私有事件信息，不回传调度器。所有输入事前应`a_i<H`；不满足的场景设计本身拒绝，不能计为已到达需求。对本来可在H内排空的世界，新观察器必须逐事件和旧账本相等。

冻结引擎原行为保留不变。TASK4 `WindowedSpatial/WindowedGreedy`使用显式本地事件循环在H停止，继承最终Spatial.optimize、预测、准入和fallback方法。先以临时初始化边界满足父类构造器，再将实际执行边界与Spatial的H恢复为43200；未来deadline不再迫使延长观察窗。H时刻事件按原优先级处理，连续功耗按[0,H)积分，H的瞬时dispatch保留其network费用。窗口关闭后不允许在同一对象继续run。

已验证同一原始世界在1000及7335规模、seed2027101800下两策略的新旧执行事件/control完全一致，共同记录/账本字段及非wall指标atol=1e-8。全7335路由也已逐ID核对；以上只支持该单世界原始情景，不能推广为24seed、跨月或压力网格已验证。新控制器实测wall平移与复查尚无已验证适配，接口显式拒绝非零measured_wall_fraction，避免悄悄改变Spatial语义。

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

旧`evaluate`仍用`controller_w×1.2`；TASK4 `evaluate_window`已显式支持按地区或统一的φ_ctrl，并完成不同PUE合同检查。若所有设施统一φ，应同步规划成本、服务器核算和controller；网络依其独立边界。只在固定轨迹上重计PUE称“accounting-only敏感性”，不能替代改变成本系数后的闭环DES重算。

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

式(4.17)的Q是端到端正损害；模型选择与交付增量的独立CVaR按式(4.32)另报。正的碳损失表示相对同world、同需求、同硬件P0增排；另报其正部和绝对C_total的尾部，不能只报增量。参数改变idle时比较仍要完整facility总量。M4反弹的固定需求基线另定义，不能混用(4.17)。

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

主水平α=.95；Core为正常/逆境各1000个条件独立world，目标p=.2，敏感性p=0/.1/.2/.4。逐次显示tail ESS、正尾权重world数、distinct tail losses、VaR处ties与分数权重、分层配对Bootstrap区间。按w_max<=1/1000推得尾ESS理论最低约50，式(4.35)给出精确计算；该门槛只是报告约定，不是尾部估计精度保证。若实际有效ESS低于50，保留结果但降级为不充分证据，按预注册规则向不足层增加独立world。

CVaR99仅Optional，不从这2000world宣称稳定99%尾部。常数/全零损失可使Bootstrap区间退化，即使ties平摊后的ESS很高，也不能据此声称未来风险为零；须同时显示distinct loss数、事件数及独立风险上界的适用性。p是情景假设，不是拟合现实频率。

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

Core先跑r=0/.25/.5/1/2并绘配对绝对碳差，r=4放Optional，若区间端点有稳定异号再细化。DES存在离散扩容/拒绝/排序变化，不假设单调，不盲目二分。报“在r_a与r_b之间出现采样支撑的交叉”、置信不确定区间或“测试范围内未定位”；网格之外不外推。用新seed确认探索出的候选根区间。

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

## 8. 分组公平性审计（算法toy已验证；正式审计未执行）

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

## 9. LHS参数范围、环境收益敏感性（范围尚未验证；先筛查再扩展）

### 9.1 分层抽样

对参数维度j、n点LHS：

\[
U_{kj}=(\pi_j(k)+\xi_{kj})/n,\quad\xi_{kj}\sim U(0,1),
\quad\theta_{kj}=F_j^{-1}(U_{kj}),\ k=0,\ldots,n-1.
\tag{4.30}
\]

π_j是独立随机排列。Core为2套独立scramble各16点，每点相同3个服务seed，共192次执行。Extended才考虑2×32点×4seed，再将8个选定点补至12seed。16点与32点的普通LHS不自动嵌套，扩大设计不能直接扣除旧192次；只有完全相同参数点/seed/完整配置才能复用。主参数LHS与服务随机子流独立。`Tri(a,mode,b)`表示三角分布，`LogU(a,b)`表示对数均匀；均为研究者假设，不是测量置信分布。

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

每点报配对ΔC_total、ΔC_op、ΔE、SLA、Q、U_valid及是否净收益/服务都合格。Core只作Spearman/分箱和两个scramble的方向稳定性筛查，不在每个仅16点的15维设计上作PRCC显著性推断；Extended有足够参数点且单调性合理时才作描述性PRCC。两个scramble比较排序与符号稳定性；仅两套设计不足以给稳健随机化置信区间，若需此类区间，应追加更多独立LHS设计并单列预算。不能从LHS的普通行bootstrap声称严格独立MC推断。

参数区间变宽会改变重要性排序，需同时报告分布与范围、相关性假设。无相关数据时主LHS假设参数独立；如后续加入相关性，用显式条件/潜在变量设计并验证覆盖，不在独立LHS完成后随意排序破坏边际分层。Extended最多8个极端/边界点先探索再补独立seed确认；确认不重新拟合上游模型。扩大触发条件见正式协议，不以未达到显著性就无限加样本。

## 10. 统计和结论发布条件

固定场景的配对差`Δ_b=Y_P0,b−Y_P1,b`可按独立world给均值、SD和配对区间；同一个world两策略不拆开抽样。需求曲线的增长实例嵌套、月份重复seed和重复prompt分别追踪依赖。区间说明条件于历史cohort、碳日期和假设参数；不能宣称覆盖真实部署全部不确定性。

环境优势声明至少同时检查：完整总账下碳差方向、服务违反预算与组别损害、有效质量交付、尾部风险、对反弹和扩容的稳健性，以及所用数据的现实代表性。任何未跑情景不写数值；软件失败不变成节能；选择性交付不变成公平。只有完成对应实验并通过执行清单，才将该模块从“尚未验证”更新为具有明确样本范围的实证结论。


## 11. 审查补充：可加性质量拆分、未完成工作、尾部支持与原因审计

### 11.1 模型选择与服务交付损害分开（已实现并验证公式合同）

令m_i为情景路由，q_im为其历史答案score，I_i为式(4.15)的合规交付指标。signed拆分为

\[
A_i=q_{iL}-q_{im_i},\quad B_i=(1-I_i)q_{im_i},\quad
T_i=q_{iL}-I_iq_{im_i}=A_i+B_i.
\]

若希望只计正损害，应定义**交付增量正损害**，而不是直接把B_i加到A_i的正部：

\[
A_i^+=(q_{iL}-q_{im_i})_+,\quad T_i^+=(q_{iL}-I_iq_{im_i})_+,\quad
B_i^{inc}=T_i^+-A_i^+\ge0,\quad T_i^+=A_i^++B_i^{inc}.
\tag{4.32}
\]

例如S质量高于L但未交付，B_i可能大于T_i^+，因为原S有质量优势；直接把A_i^++B_i当综合损害会失真。分别报告mean(A)、mean(A^+)、mean(B)、mean(B_inc)、mean(T)、mean(T^+)和pass harm的同样增量拆分。CVaR分别对模型选择损害、交付增量和端到端损害评估，不把三者再次相加收费。当前`quality_harms`已检查加性与S优于L的例子；系统结果尚未运行。

### 11.2 H时刻backlog（已实现并验证部分执行/未开始队列合同）

定义U_H为H时非终结实例，R_H为其中已开始服务实例，Q_H为尚未开始实例，包含queued/inflight/pending/planned/routing。failed是已终结失败，不在U_H中：

\[
N_{unfinished}(H)=|U_H|=N_{censored},\quad
N_{not\ completed}=N_{failed}+N_{censored},\quad
N_{waiting}(H)=|Q_H|,\quad N_{running}(H)=|R_H|.
\]
\[
W_H^{nom}=\sum_{i\in Q_H}\bar p_i+
\sum_{i\in R_H}\max\{0,\bar p_i-(H-v_i)\},\qquad
W_{H,running}^{actual,eval}=\sum_{i\in R_H}\max\{0,P_i-(H-v_i)\}.
\tag{4.33}
\]

bar p为本情景已确定的名义service。实际量仅针对已经开始、环境已抽取service的任务，明确为事后评价字段，不输入调度器。未开始实例的实际剩余service未知，报告unknown数量，不读取其Z去填真值。名义剩余可能在真实超时时降为0，不表示运行已结束，因此必须并列running数与事后实际剩余。

控制器工作也存在截尾：

\[
W_{ctrl,H}=\sum_k|[s_k,s_k+L_k)\cap[0,H)|.
\tag{4.34}
\]

它与完整计划计算时长不同；跨H未完成solver产生pending截尾，已经消耗的控制电量照计。队列等待不计CPU工作。部分active、network已发费用及idle同时保留。

### 11.3 尾部有效样本量与Bootstrap（已实现并验证离散/非等权合同）

对VaR阈值eta，先给L_b>eta的world全部原概率w_b，余下尾质量按L_b=eta的所有world原权重比例分配为t_b，使sum(t_b)=1-alpha。归一化a_b=t_b/(1-alpha)，定义

\[
ESS_{tail}=\frac1{\sum_ba_b^2},\quad ESS_{world}=\frac1{\sum_bw_b^2},\quad
CVaR_\alpha=\sum_ba_bL_b.
\tag{4.35}
\]

ties平摊避免ESS随文件排序改变，但常数损失的distinct_tail_losses=1，即使ESS高也不提供未见极端事件的信息。Core每层1000时任意p∈[0,1]有w_max<=.001；sum(a_b^2)<=max(a_b)sum(a_b)<=.001/.05，故95%尾ESS>=50（浮点容差除外）。这解释为何保留2000条件独立world，而不是把小样本压力网格拼成尾样本。

`paired_mixture_cvar`在正常/逆境各层内有放回抽相同world索引，P0/P1共享抽样；保持n_0/n_1不变，重新赋目标p/n_r权重，再重算CVaR。默认2000次Bootstrap报告逐指标点态95% percentile区间；正式图表同时显示尾ESS与原world数。p=0/.1/.2/.4只改变评价分布，不进入在线调度，不重新跑DES。

多指标和多概率的点态95%区间不能称整体95%联合覆盖。离线选择需在独立确认阶段预定检验族并校正，或明确仅是条件探索性审查。Bootstrap区间和重复种子不替代新独立质量标签或生产尾部数据。

### 11.4 故障原因和途径（观测标签已实现并验证；因果归因未识别）

`failure_decomposition`输出以下可重叠标签：上游模型选择positive harm（需评价真值且标assessed）、服务器零容量暴露、队列/预测限时过期、MILP非法候选不可恢复拒绝、solver timeout、fallback使用、已经开始执行而实际超过deadline、窗口截尾。执行超时仅在实际deadline已经越过或完成时间确实晚于deadline时标记；deadline在H之后的running截尾不能提前叫执行超时。

主服务结局按窗口截尾→执行超时→有效完成→队列/预测上限→观察容量暴露→其他服务失败的固定优先级互斥分组，其数量之和=N。可重叠机制标签不可相加成100%。例如数值拒绝触发fallback但最终按时完成，应保留numeric/fallback两个途径，同时主结局仍是有效完成；不能把一次可恢复服务过程同时算成两个失败。

零容量暴露由事后已发生事件区间与等待区间的交集识别，不把所有队列失败自动归因容量。上游S/L是否**导致**排队超时需另做改变动作的反事实执行，当前不据标签宣称因果归因。

## 12. 三类环境问题与预定义资格规则（机制/阈值尚未实证验证）

### 12.1 同需求、外生增长与真实反弹机制的区别

式(4.21)固定同r/同fleet比较P0/P1，回答调度是否减碳。式(4.22)将采用后r>0的总设施碳与固定需求基线比较，回答增长后是否增排。两者都不证明增长由效率提升引起。

真实效率诱导反弹需要可识别的价格/等待→需求因果通道。令同需求效率改进为s，观察到的需求相对变化为r，只有当r由该改进诱导而不是外生业务增长时，才讨论反弹率。式(4.26)弹性只是未验证假设；主Core对外生r=.0/.25/.5/1/2给条件排放和服务曲线，再用假设弹性映射可能r区间，图题必须含“假设效率诱导”。没有价格/等待弹性、采用率或独立业务数据时不能报告真实已识别反弹。

固定idle、增量电量、地区碳分配、扩容idle和服务拒绝分别核算，线性式(4.23)只是诊断。若同需求电量已经上升，则不宣称节电空间被反弹吞噬；仍可研究碳空间因总需求增长而消失。

### 12.2 离线资格审查预算（函数已实现并验证toy；预算本身仅设计）

以下数值预先固定为研究预算，不是官方标准。各upper/lower表示未来正式证据在其声明置信方法和总体下的界，缺失/稀疏则判证据不足：

| 条件 | 预算 |
|---|---:|
| 全输入service violation upper | <=.001 |
| CVaR95(service violation) upper | <=.005 |
| 模型选择positive harm upper | <=.02 |
| 模型pass harm upper | <=.02 |
| 端到端positive harm upper | <=.02 |
| 最差主组模型positive harm upper | <=.05 |
| 最差主组P1相对P0的violation增量upper | <=.005（0.5百分点） |
| 同需求总设施碳节省lower | >0 gCO2e |
| 所需压力概率下最小tail ESS | >=50 |

须先满足源/软件合同、全world工件完整与预注册统计协议；不允许碳收益补偿服务/质量不合格。全部过关但仍缺新独立质量、实机功率/PUE、as-of forecast、真实业务轨迹或代表性分组时，最多输出`CONDITIONAL_SIMULATION_CANDIDATE_ONLY`。合同失败/缺失区间/尾ESS不足输出`INSUFFICIENT_EVIDENCE`；超预算输出`NOT_QUALIFIED_IN_TESTED_SCOPE`。当前正式实验未执行，不能给任何真实候选“通过”结论。

`qualify_offline`是报告阶段的审查函数，不调用任何调度函数；`router/window`不导入它、`metrics`或`risk`。在线只能用保存的路由/名义服务、已发布预测、已到达队列与当前容量，不能读取未来真实碳或本窗最终CVaR来挑选策略。若部署时需要限流或故障fallback，须事前定义只依当前可观察状态的规则并独立验证，不能把离线审查门槛直接变成能预知未来的在线策略。
