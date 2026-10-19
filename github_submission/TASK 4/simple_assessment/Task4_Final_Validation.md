# TASK 4 Final Validation — 最终离线验证

日期：2026-10-18。**PASS；TASK 4 READY FOR PAPER WRITING**。本轮只对原24次执行的已保存工件离线计算、独立核对和撰写新材料；新增DES、预测推断、训练、All-L仿真均为0。此状态适用于条件仿真论文写作。

## 1. 文件保护和实验来源

| 验证对象 | 本轮实际核对 | 结果 |
| --- | --- | --- |
| 原TASK 4全部文件 | 开始快照中2540个文件逐一SHA-256对照，含历史报告/Results/三图/原代码/日志 | PASS，无修改或删除 |
| D盘上游及题面 | 6402个文件完整快照再比，包括task2、TASK 3、Work 1、latency_environment、原cohort和PDF | PASS，无增加、修改或删除 |
| 实际采用的C盘第三问最终release源码和输入 | 58个文件路径、集合、SHA-256对照 | PASS，无变化 |
| 24次原执行 | completion仍24条COMPLETE_VALIDATED；record集合及所有原工件哈希保持；此前独立检查逐record核验manifest可复用性 | PASS，没有重跑 |
| 策略配对 | 4情景×3seed=12对；每对Greedy/Spatial共享world_id | PASS，无伪增独立样本 |
| 新文件范围 | 所有新文件均在TASK 4/simple_assessment下 | PASS |

原失败日志全部保留。哈希基线为final_review/before.json，结果为final_review/preservation_after.json；本轮脚本未调用任何执行器、模型predict或训练。静态解析新增Python脚本通过；复核用旧接口只读读取、签名校验，不修改上游。

## 2. 独立数值和公式检查

| 项目 | 独立方法与来源 | 验证结果 |
| --- | --- | --- |
| 原汇总一致性 | 原Results的24行逐项对保存metrics/质量/状态/能源账本重算 | 全PASS，最大绝对差1.78e-15（浮点序列化精度；容差1e-10），未修原CSV |
| 真实S/L质量 | quality_harms的q_s/q_l逐base ID对task2/inputs/paired/outcomes.jsonl；全部7335原ID覆盖 | PASS；未用requests中设计为空的legacy质量列，未将NA填0 |
| 冻结选模规则 | 从保存的perturbed_S/L、cost_delta与threshold逐ID重建规则；fallback全部L | 24记录PASS；只是保存数值的代数核对，没有重新推断 |
| 预测翻转 | 独立Python字典/标量求和，配对同base_request_id；三个seed逐项复核 | L→S=67/75/68，S→L=70/64/65；总137/139/133；Math/Code=0 |
| 局部质量损害 | 同翻转集合前后真实Score和阈值pass；下降定义delta≤−0.10；另报所有下降 | 下降19/22/18；新S失去L可达标13/15/12；配对均差−0.016423/−0.026978/+0.016917 |
| Router−All-L | 每任务逐请求Score差、阈值指示差；独立标量累加核对全部24记录 | PASS；Math/Code差严格0，L不逐题支配S，All-L仅离线质量参照 |
| active能源与碳 | 逐interval跨小时拆分，按原world地区×小时碳表重新积分，与原ledger及地区汇总对照 | 24记录PASS；active电量误差≤1e-10kWh、碳≤1e-8g |
| 完整能源边界 | total=idle+active_increment+network+controller；运行增量含net/ctrl，后者不重复相加 | PASS；固定12h/六host，idle与原第三问口径一致 |
| 线性反弹 | 每seed独立检查c0与r_lin；代回idle+(1+r)N0c0恢复原Greedy；核对退化分母 | 三seedPASS；公式成立于固定idle与常数增量，99.21%是理论参考 |
| 实际增长 | 原增长Spatial总账减原需求Greedy，逐seed比率后取均值 | +2.495812g、约0.006045%；两增长水平不能定位实际非线性阈值 |
| 组成恒等式 | 各seed任务权重和组内Score/pass的组成项+组内项恢复总体差 | PASS；Math/Code份额变化不是逐题真实难度标签 |
| 新统计长表 | 全5175行按来源表、scenario/seed/policy/task/region/hour/metric重新定位逐值对照，检查唯一键和单位 | PASS，48个空翻转子组均值为NA；其余5127值有限 |

独立检验结果为final_review/independent_checks.json，计算来源/真值归档核对为calculation_audit.json，单位与分母为statistics_schema.json。长表5175行不是5175个新世界；409次跨seed翻转涉及290个distinct base ID，不是409条独立新题。逐seed结果、逐ID翻转明细、两层分组、地区/小时账本与反弹分解都保存在final_review支持表。

## 3. 三图、报告和数值口径

三张原PNG已逐张视觉检查，并逐SHA-256证明未改。图1点为seed、横线为均值、竖线为观察范围；P95仅完成者，右下为全输入违反率，右上为有效交付达标率。图2左轴kg、右轴g，二者都保留固定idle边界的解释，跨需求差不冒充同需求差。图3为Spatial−Greedy违反率百分点，负值表示更少违反，n是实例数而不是独立prompt数。Difficult旧图文字在新caption中明确解释为Higher Share of Math/Code Requests代理。

六个中文显示公式及一个英文提纲公式已检查数学排版；7个图/数据/来源链接目标存在；Markdown没有控制字符或残缺公式定界。没有新绘图或新增图。

原报告的active加权强度42.238与本轮42.237在第三位小数不同：旧值为合并能量/碳的比，新值为三个seed比率之均值；均不是原总账错误。新报告显式统一先算seed再汇总。第三问24seed与本问3seed减排百分比稍异也来自汇总cohort，不用修改第三问。原24行Results与账本没有实质数值冲突；本轮只补统计和说明，全部原数据保留。

## 4. 结论边界与八项验收

| 验收问题 | 是否回应及定位 |
| --- | --- |
| 原文第四问要求 | 已回应难度代理、预测误差、用量增长、总排放、类别不利及环保所需证据；Final中文报告第1–7节逐项对应 |
| 公式正确与条件 | 已核对；组成/配对/碳恒等式正确，线性临界仅限固定idle/增量条件，非实际阈值 |
| 预测错误具体影响 | General发生双向翻转，18–22条局部质量下降/seed；总体小来自稀疏、同分与抵消，无guard消融因果或新覆盖保证 |
| 任务组成和增长的服务影响 | Math/Code份额提高使Spatial P95约7.827→11.967s；增长使其违反0.0182%→0.1697%、P95→11.885s；均带完成率 |
| 同需求减排为何不保增长减排 | 增长改变需求量、地区执行权重及单位增量；同需求减排与跨需求总账比较分开，行为诱导未识别 |
| 类别不利是否存在 | General相对All-L均值损失已展示；36配对未观察Spatial提高违反率，但P95较长、Code增长风险最高，非无不公平证明 |
| 环保证据和假设 | 六层证据明确；公开历史碳/benchmark与转移功率、假设PUE、条件DES及缺失部署证据分开 |
| 论文阻断 | 未发现原始数学/数据/执行Critical；Important解释与新增统计已解决，余下真实部署证据是声明的局限 |

所有汇总先逐seed计算，n=3仅描述均值/观察范围，不宣称显著性、95%置信或总体风险保证；重复请求与复用记录不冒充新独立样本。P95带完成/违反指标，不将少交付的低耗电称成功节能。没有在决策时使用本轮事后真实质量或未来碳；本轮事后分析不构造可部署oracle策略。+2.496g的现实符号稳健性未验证，目标功率/PUE/碳假设可能影响该细小差异；没有追加敏感性试验。

## 5. 状态与停止点

| 状态 | 本轮范围 |
| --- | --- |
| 已实现并验证 | 已保存数据的追加统计、独立配对/规则代数/小时积分/公式/单位检查、五份交付和原工件保护 |
| 已实现但未验证 | 无本轮新增未验证实验实现；本轮不重跑既有合同测试或仿真 |
| 仅设计/仍缺数据 | 目标模型实测功率/PUE、真实业务及行为反弹、人口公平与边际电网反事实；不是本轮已完成结果 |
| 环境阻断 | 本轮离线审查无环境阻断；不据此声称所有未用上游接口的兼容问题已解决 |

Critical：0项待修复。Important：0项待修复。Minor：数值聚合、NA/单位、图注和数学排版均已澄清。可进入约2–3页第四问英文论文写作；不等于实机部署、普遍鲁棒/公平或环保认证。**TASK 4 READY FOR PAPER WRITING**。按授权到审计和材料完善止，不生成英文全文，也不运行原大型Monte Carlo/CVaR/DRO/LHS/复杂联合实验。
