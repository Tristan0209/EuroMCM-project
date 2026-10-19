# TASK 4 联合世界生成器规范 v3

日期：2026-10-18。实现为 `task4_core/world.py`、`protocol.py`；本文件解释工程实现，不更改 Core Protocol v2 的模型、参数分布或样本数。软件验证范围与实跑数量以 Readiness v3 和 Pilot 报告为准，全部正式实验仍未执行。

## 1. 数据与权限边界

`ScenarioBuilder`读取冻结规范化7335请求、Task2 prompt/benchmark/group/split、两动作事前服务估计、训练侧长度和Pre预测分位数、物理host与网络、实际小时碳及元数据。历史两动作答案仅计算来源哈希；发生器不解析评分，也不按评分选择样本。`Task4Runner.quality_after_simulation`在该策略DES结束后才解析两动作score并按base ID连接。

每个实例保留 `instance_id`、`base_request_id`、`prompt_group`、原phase=test、task、benchmark、origin、class、arrival、deadline相对预算、wait、priority、两动作服务代理和train-only分箱。prompt group与原splits逐条核对。旧test被再次使用，不是新holdout；复制实例不是新的独立质量观测。

正常r=0且不改任务混合时，instance_id保持原request_id；混合重采样使用 `mix_序号_源ID`，增长追加 `growth_序号_源ID`。两类ID命名空间不重叠。世界中禁止出现重复instance_id。

## 2. 任务混合与三种流量机制

- 困难份额按Code+Math定义，Code:Math保持86:1710。固定N用最大余数法确定三个任务数，再在各任务历史请求池有放回抽样、随机排列，放到原有arrival槽位。预算、class、origin与服务代理来自抽中的源请求。只用事前task标签，不用真实q挑“困难题”。抽样后实际比例因整数取整存在至多一个请求误差。
- Compression令arrival除以ell，保持N及每请求deadline相对预算；不是需求增长。
- Burst按数学式的分段逆CDF变换。未另指定时用t=3600–4500秒的15min区间；密度倍率为参数3/5/10，归一化且保持原完整到达支持。与Compression组合时先burst后compression，因而原15min段在最终时间轴会同时压缩；顺序写入世界审计，不将两者描述成独立不变的最终时长。
- Growth保留全部原实例，追加floor(rN0)个实例；源请求及arrival从同一原cohort经验分布抽样。所有r使用同一增长子流的前缀，extra ID、arrival和潜变量可嵌套核对。增量复制保留历史任务分布；偏移混合增长仍为Optional。固定H43200，不延长窗口为更多输入排空。

默认小规模cohort沿用已审计的Task3确定性分层子样本，覆盖原完整时段。正式N0仍为7335；Pilot的N0=800只验证工程。世界参数包含N0，反弹比较必须核对相同N0，不能把800基线与500基础量上的增长混比。

## 3. Pre误差与合法重路由

对于每个prompt group，用独立子流生成u_g、epsilon_gS、epsilon_gL，按数学式(4.3)计算两个加性误差。同组的所有复制实例使用相同误差；两个模型共享sqrt(rho_q)u_g，模型独立成分为sqrt(1-rho_q)epsilon_gm。支持两个独立质量bias，以及beta_L=beta_gain/2、beta_S=-beta_gain/2的收益bias。

`routed_requests`调用真实冻结Pre LightGBM，然后clip预测、执行冻结成本及阈值规则。Code/Math和未注册benchmark仍强制L。没有改写原selected_model，也没有翻转旧标签。新动作选用冻结的 `estimated_service_time_S_seconds` 或 `_L_seconds`，再乘该动作的敏感性倍率；L的原3倍已包含，不能再乘3。

路由工件保存两原预测、两扰动预测、score/threshold、动作/calls、fallback、base/group、两模型实得扰动和clip标记、选中服务代理。指标保存实得gain bias、相对原预测的扰动RMSE和clip率；这些不是用真实q测得的预测准确率。原finite-sample guard不自动覆盖新混合/失准/重复量，持续标记未重新评估；未重新训练或校准。

## 4. 服务潜变量与容量

每个instance预生成Z=exp(-sigma_T²/2+sigma_T eta)，均值1；相同world所有策略共享同一Z。调度不读取Z，冻结 `_admit` 在服务真正开始时才揭示一次。H时尚未开始任务的实际剩余工作仍未知；只报名义剩余和未知数，不预读其Z填入调度记录。已开始而未完成的实际剩余仅供事后评价。

正常每地区S/L各一host。`l_hosts=2`为每地区增加一个L物理host，共9host、72GPU；slots只增加单host的准入槽，不增加host idle。pool内PUE和边际busy功率须同质，异质输入显式拒绝，不能平均后冒充host-aware优化。

Availability保存t=0初值、单FR-L或全部L池的故障事件、恢复事件，支持较长持续时间和容量保留比例。保留容量=floor(安装slots×比例)。环境在同刻优先处理容量变化；优化器收到当前capacity字典，不能查未来事件。运行任务不抢占；恢复后才可观察并重新准入。不可用主机始终计idle，未建立关机机制。

## 5. 碳误差场和环境反事实

actual使用2025-01-02锚点前24h及窗内13h真实历史生命周期CO2e；这是历史数据回放，不是当前生产预测。LHS的共同尺度及地区对比度严格按式(4.31)变换，并保存clip率和假设反事实标记。

世界在j=0..12的UTC相对整点预生成三个地区的乘法因子，按式(4.10)设置rho_C、地区bias及-lognormal均值修正。每个j共享一个公共正态量，另有各地区独立量；不同j独立。世界保存实际因子和seed，完全可复现；生成过程不用未来actual反推残差。

`FieldForecast`先调用冻结ForecastView的无噪声persistence，只获得issued以前已完整结束的小时历史，再乘world[j,region]。随机键不含policy、精确issued或调用序号。相同小时不同issued共享外生误差来源；跨小时可观察历史更新是合法信息差异。forecast audit保存issued、observed cutoff、grid hour；完整issued预测表也落盘。

没有改变Spatial的优化目标、约束、候选、30s solver budget、1e-5数值接受或恢复协议。只替换预测输入和环境，保留最终优化函数身份；不调用旧Full MPC/旧DES。

## 6. 世界对象、种子及哈希

`ExogenousWorld`完整保存参数、seed、anchor、请求实例、Pre误差、碳误差场、service shocks、容量事件、host/network、actual carbon与元数据、来源哈希和发生审计。world.json为完整可重建规范载体，另有压缩CSV便于检查；两者都被执行manifest校验。当前双份保存有存储成本，不能在报告中漏计。

子流使用SHA256(seed|stream)的128bit前缀初始化PCG64，流名为task_mixture、growth、pre|group、service|instance、carbon|hour及conditional_parameters。请求/组键稳定，不依赖策略调用顺序或Python hash随机化。所有随机量在执行策略之前形成；策略过程中不抽外生随机数。

world ID为规范化完整外生payload的SHA256；策略运行前后重新核对，检测原地污染。执行键另外纳入全部Task4模块代码、冻结上游来源、实际路由/服务、H/anchor、资源/数组、controller/solver配置、policy及Python/库/平台版本。研究模块名不进执行键。改变任一影响执行的量会失去复用资格。

正式种子、Pilot种子和冻结兼容性种子分离：Pilot为2028101800/1801；Core固定配置保留2030101800..1811；MC两层各1000使用2040101800起及额外10000偏移；LHS scramble种子20311018/19。跨固定情景使用公共seed以减少差分噪声，因此Pilot14个世界不是14个互相独立IID样本；每个情景只有2个独立服务seed。

## 7. 编排与当前实现边界

`protocol.core_plan()`只产生4558执行引用：C1=48、C2=192、C3=4000、C4=120、C6=192、C8=6；C5复用，C7为Optional。相同参数/seed/policy的名义键去重为4540，实际能否复用仍需完整执行签名和工件。MC条件分布、p敏感性、LHS15维范围均保持v2；dry-run生成计划不执行任何DES。

CLI只有dry-run、pilot、validate，当前硬性执行上限为1000实例，正式门关闭。pilot配置还限制最多8情景、每情景最多3个不同整数seed、只运行两主策略；当前配置7情景×2seed×2策略。断点续跑需要同一配置及显式--resume，失败attempt、异常、部分轨迹保留；同签名仅可引用完整有效记录。后续正式授权需要单独审查开闸，本轮不解锁。

实现尚不能据Pilot认证完整7335压力下的wall、内存、尾部事件或部署资格；真实硬件、真实需求反弹及人口公平仍需新数据。Optional的真实wall平移、异质pool、偏移增长和新月份未实现或未验证，不能作为本轮已完成结果。
