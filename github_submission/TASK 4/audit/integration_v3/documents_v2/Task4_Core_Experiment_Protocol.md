# Task 4 Core Experiment Protocol — 精简正式协议 v2

日期：2026-10-18。**当前状态：仅设计，正式M1–M8尚未执行。**基础路由/观察器/核算及分析函数已完成注明范围的合同测试，不能把合同结果记作本协议的性能结果。

实施与指标分别见[实施方案](<D:/EuroMCM/EuroMCM-project/TASK 4/Task4_Implementation_Plan_ZH.md>)、[数学模型](<D:/EuroMCM/EuroMCM-project/TASK 4/Task4_Mathematical_Model_ZH.md>)。本协议覆盖旧文件中较大的预设运行规模，原范围保留为Extended/Optional。

## 1. 主要问题和比较对象

1. 在困难混合、预测失准、随机慢服务、突发和容量故障下，整个既定系统的服务/质量损害、tail risk和组别差距是多少？
2. 固定同一世界、同一S/L路由时，最终Spatial相对Greedy的调度净减碳能否保留？
3. 同窗请求数量增长和扩容后，总排放是否越过固定需求基线？没有需求弹性数据时，哪些“效率诱导反弹”主张仍不能识别？

P0=本情景重新执行Task2 Pre_direct后固定路由＋final DES Greedy；P1=相同路由＋final Spatial MILP；P2=All-L＋Greedy，仅用于所测正常参考范围的端到端参照。P2必须重选L服务与q_L，不能继承原S任务服务。主比较保留共同router proxy .01s；All-L零router开销可作Optional假设下界，不能称真实决策CPU为零。

禁止把原Full MPC恢复为主模型。最终Spatial的optimizer与数值接受原样继承；TASK4只扩展观察窗口、核算和环境/评价。配置只允许固定proxy控制，当前不支持未经验证的实测wall平移模式。

## 2. 全部实验共同合同

- 数据来自冻结7,335历史请求、对应两动作benchmark结果、train-only服务统计；原始权重/数据/论文/结果只读。这个旧test不是新独立holdout。
- H=43200s，原UTC锚点2025-01-02；所有输入arrival在[0,H)，deadline可超过H。H时处理完同刻事件，尚未终结者censored；continuous busy/control记[0,H)交集，已发送network不退款。
- 一世界两策略共享路由、输入ID顺序、服务潜变量、碳/误差场和容量事件；参数/seed键不包含policy。扰动后动作由同一个冻结决策函数计算，不能flip旧标签。
- 允许调度读取：已到达请求、当前观察容量、名义服务/预约、已完整观察的过去小时所发预测。实际未来碳、未揭示Z、未来arrival/outage和事后q只在环境/评价侧。
- host idle每物理host一次；slots不加host；新增host必须加idle/GPU；IT PUE一次，network不再乘PUE，controller使用显式PUE。不同pool可不同PUE；同pool异质边际功率/PUE当前明确拒绝。
- 每个world保存全输入终态、events、solver/recovery/fallback、controls、ledger和来源/随机数组哈希。软件异常不删除world、不填0、不覆盖失败attempt。

## 3. Core Experiments：必须完成

### C1 / M1：最小单因素解释表

基线加下列7代表点，共8配置；每配置3服务seed、P0/P1、N7335，最多48执行引用。

| 代表因素 | 非基线值；其他保持基线 |
|---|---|
| 困难混合 | Code+Math份额.6，内部比例固定；固定N的历史组回放 |
| 拥塞压缩 | ell=2，仅压缩arrival，不增加N |
| burst | 预定15min段密度倍率5，归一化保持N和支持时间 |
| Pre预测失准 | gain bias=-.1，sigma=.1，group公共成分rho=.5 |
| 碳预测失准 | sigma=.4，FR bias=+.4、DE=-.4、ES=0；外生整点误差场 |
| 服务长尾 | 均值1lognormal，sigma=.5 |
| 单区故障 | FR-L在t3600停用1800s，再恢复；保持供电 |

这是机制筛查，n=3不用于尾部认证。完整22点网格移入Extended。

### C2 / M2：联合压力

8配置，每配置12独立服务seed、P0/P1、N7335，最多192引用：正常；困难+Pre失准；ell2+burst5；碳失准+FR-L故障；sigmaT.5+FR-L故障；困难+ell2；所有L池共同故障30min；所有代表压力共同作用并采用共同L故障。参数取C1代表值。

这里只是固定逆境，不赋等权现实发生概率。这里的ell压缩不能称数量反弹。每配置报告worst/mean、配对差、全部终态和组别；不能把8×12写成每配置96seed。

### C3 / M3：MC、CVaR95及概率敏感性

使用数学说明第3.5节两个条件regime分布，**每层1000个独立world，共2000world×2策略=4000次**。每world含完整条件参数、group误差、到达/容量/服务随机轨迹；同层种子独立，同world两策略配对。总N固定7335，数量增长留在C4。

目标逆境概率p=.2；敏感性p=0/.1/.2/.4复用同样执行记录，权重分别为(1-p)/1000和p/1000。不重跑DES，不把4个p当8000独立world。

主要tail损失为全输入违反率、模型选择positive harm、交付增量positive harm、端到端positive harm、同世界碳regret；分别报告CVaR95、VaR、tail ESS、正尾world数、distinct tail loss数和区间。绝对设施总碳的tail也保留，不能只报扣idle后的比例。复合损失为辅助，权重预定而不据结果调整。

每层Bootstrap重采样原层大小、同一索引同时抽P0/P1，重复2000次，报告点态95% percentile区间。CVaR差与“配对差的CVaR”分开命名。每层1000使p∈[0,1]的单world概率<=.001，因此95%尾ESS理论下界约50；同时仍须报告事件数和ties，零事件退化区间不能认证零风险。

若工件不全或实际tail ESS<50，停止资格结论、定位缺口，必要时在不足层独立补样；不能用重复旧seed充数。CVaR99不在Core。

### C4 / M4：增长、容量与条件反弹边界

r=0/.25/.5/1/2，6seed，两资源配置（原6host、48GPU、slot1；各地区L增加到2host、S不变，共9host、72GPU、每hostslot1），P0/P1，共120引用。N(r)=7335+floor(r×7335)，原arrival支持和H固定；额外实例嵌套复用，group ID追溯原prompt。

同时计算：同r同fleet的P0-P1；固定需求/固定原fleet基线与采用后总设施排放差；SLA、有效质量、backlog与未完成工作。新扩容host即使闲置也计电。

先报告外生增长的条件排放曲线；若用弹性公式映射r，明确图题为“假设效率诱导”，弹性未估计。没有真实价格/等待/采用率因果数据就不能称真实反弹机制已识别。未出现有支持的交叉就报告未定位；不假定单调、不凭拒绝更多请求保住低碳。

### C5 / M5：公平性和原因审计

直接复用C1–C4记录，不新增DES。主组为task、benchmark、service_class、模拟origin及train-only长度/Pre难度分箱；主要交叉task×class。每组显示N、unique prompt groups、失败/截尾、模型/交付/端到端损害、全输入违反、完成分位数和时限标准化等待。

class标准化用共同固定权重；空单元标不可估，不填0。按world配对统计；想推广到prompt总体时另有group层不确定性，但不能把重复历史prompt当独立新质量数据。多组选择事先定义检验族；Code样本少时结论降级。

原因审计保留上游模型损害、容量暴露、队列限时、MILP数值拒绝、fallback、执行超时和窗口截尾的重叠标签，以及加总=N的互斥主服务结局。不把fallback或numeric rejection自动算作最终服务失败，也不把观测暴露当因果归因。

### C6 / M6：LHS小规模筛查

保留数学说明的15维范围，但先做2个独立scramble×16点×3服务seed×2策略=192次。其范围为研究假设，不是实测置信分布。每点物理合法，服务/功率/PUE/网络/Pre/碳变化中影响决策的部分闭环重算；仅核算变更的复用须有明确证明。

先比较Spearman方向、分箱图及两个scramble的top-5重叠，不对每个16点、15维设计作PRCC显著性推断。若top-5重叠少于3、重要方向不一致、出现资格边界/符号变化或未解释tail，触发Extended LHS；若方向一致且无边界迹象，可停止于描述性筛查并保留“低功效、未排除交互”的限制。不能为了得到显著结果无上限增加点数。

### C8 / M8：端到端正常参考

P2在6个对齐正常world运行，共6次；P0/P1引用C2正常子集。相同fleet、H、输入世界；用L对应服务和评价score。只支持这6个正常world的端到端比较。若论文要主张全压力下整个系统优于All-L，则需要Extended对齐所主张范围，不能拿这6条推广到MC尾部。

## 4. Extended：条件触发的确认

| 扩展 | 触发及规模上限 | 复用要求 |
|---|---|---|
| M1细网格 | 发现代表点失效，补到22配置；优先沿失效因素补水平 | 原完整同键点复用，若新seed则不能扣旧执行 |
| M2独立确认 | 8配置中关键边界用6个事先预留新seed确认，最多96执行 | 不把旧12seed挑选出的最差seed再称独立确认 |
| M4边界 | 有交叉/服务失效迹象时，最多2fleet×4新r×12新seed×2=192执行 | 新r产生新实例集合，完整键核对；非单调可多区间，不能盲二分 |
| M6扩展 | 2×32点×4seed×2=512新执行；随后最多8点补到12seed | 普通16→32点LHS不自动嵌套；512不可直接减192。8点确认192引用若已有4seed可只新增128 |
| M8失准参照 | 事前固定100个分层world加入All-L，额外100次 | 只支持该子集均值/分组对照，不据100world认证95%极端尾部 |

这些上限不是承诺全部运行。预留确认seed与探索数据分开；改变核心分布或损失应新协议版本，不能改完仍沿用原资格结论。

## 5. Optional：需要额外主张或预算时

M7四预定月份最终Spatial复验；CVaR99独立加强；r=4；固定6host的slots2/4/8；L每地区3host；新增任务偏移增长；有滞后需求反馈/弹性数据；实测wall；Full/E机制对照；真实新质量/硬件/人口数据验证。每项单独计算预算，不塞回Core的默认运行。

## 6. 严格去重与独立性

执行键包括模型/观察器/核算代码、完整路由与request、世界数组、host/network/carbon/forecast/capacity、anchor/H、controller与solver协议、policy。analysis模块名、图表名和引用序号不进入执行键。同键仅可复用`COMPLETE_VALIDATED`且所有工件哈希有效的记录；不完整/失败/不同PUE/host/数组/时间边界/solver版本都不能混用。

Core最多4558执行引用；M1/M2正常基线共享3对、M4的r0原fleet共享6对时，可扣18执行，约4540个新执行上界。实际扣除必须有台账，不因同seed或同N就假定相同世界。M3独立条件world仍是每层1000；同一次执行被多表引用不增加样本。历史Task3记录只能当历史基线；未经完整语义、工件与核算合同验证，不能直接冒充新TASK4已执行world。旧B/C归档绝不当最终Spatial敏感性。

## 7. 离线判定和放行边界

预定义风险/质量/公平/环境预算见数学说明第12节和`qualification.py`。点态CI不作联合覆盖承诺；独立确认时预定多概率/多指标检验族及校正。缺界、全world不完整、ESS不足判证据不足；服务或质量超限不能用减碳补偿；缺真实部署数据最多为条件仿真候选。

这些是离线证据审查条件，不能把未来actual、未来完成或本窗最终CVaR作为在线策略输入。当前基础接口可做有界pilot，联合世界生成/完整runner/台账尚未实现；**本协议尚未进入正式运行**。
