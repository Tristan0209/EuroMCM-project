# MILP 数值解恢复报告

## 路径审计

旧 queue_milp.py 和原 Spatial 均对 y/z 取整但保留 u，随后检查矩阵；原 Full 的 candidate-time MILP 全为整数变量。新 Spatial 与 E 共用 final_release/queue_milp.py，原 Full 使用独立 v2/milp.py。所有完整向量还检查变量上下界，不能仅检查选中的任务。单请求枚举没有此整数/连续取整路径，保持原实现。

## 已有反例与不能恢复的原因

原 rejected_3621.npz 的原始线性残差约 7.78e-13，整数偏差 5.3448414e-7，取整后违反 0.0037347603 秒。固定整数后 LP 返回 infeasible，而非成功恢复。独立约束证据：u10>=6987.347812924047，u1-u10>=0.370336900494294，故 u1>=6987.718149824541；但 u1<=6987.714415。差 0.003734824541 秒。只把未选 u 置零无法消除该矛盾。更改二元选择或放宽容差会改变问题，本轮不采用。

## 恢复协议

先检查整数偏差与完整向量。仅在必要时固定所有整数变量，用剩余原 wall budget 重求连续 LP；不改变任何分配/拒绝。连续目标只作正比例缩放，LP 原始/对偶容差1e-7；原完整模型复检容差保持1e-5。完整恢复后追加名义容量、网络允许、时间和队列非重叠检查。不可恢复返回 status4，进入原显式 fallback。保留失败、超时、未选 E 分支和证据矩阵。纯整数 Full 无连续自由度，不伪造修复。

## 完整核心重跑统计（每策略24种子；次数含未选分支）

| strategy | recovery_illegal_rounded_candidate_count | recovery_attempted_count | recovery_recovered_count | recovery_unrecoverable_count | status4_count | fallback_count | native_timeout_count | recovery_lp_wall_seconds | actual_native_milp_calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E-Full | 14 | 14 | 0 | 14 | 14 | 29 | 0 | 0.0165906 | 194496 |
| E-Spatial | 14 | 14 | 0 | 14 | 14 | 46 | 0 | 0.0184577 | 194496 |
| Greedy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| MPC | 0 | 0 | 0 | 0 | 0 | 24 | 0 | 0 | 48753 |
| Spatial_MILP | 14 | 14 | 0 | 14 | 14 | 46 | 0 | 0.0178323 | 48624 |

核心+F/H去重后的 234 次正式执行统计如下；重复使用的零开销/默认gamma配置只计一次。按阶段引用的计数另列 numerical_recovery_by_phase_references.csv，不可把其总和当唯一调用数。

| strategy | recovery_illegal_rounded_candidate_count | recovery_attempted_count | recovery_recovered_count | recovery_unrecoverable_count | status4_count | fallback_count | native_timeout_count | recovery_lp_wall_seconds | actual_native_milp_calls |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E-Full | 68 | 68 | 0 | 68 | 68 | 242 | 0 | 0.0721552 | 534864 |
| E-Spatial | 64 | 64 | 0 | 64 | 64 | 298 | 0 | 0.0681649 | 534864 |
| E-Spatial_single_branch | 1 | 1 | 0 | 1 | 1 | 17 | 0 | 0.0008711 | 24312 |
| Greedy | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| MPC | 0 | 0 | 0 | 0 | 0 | 24 | 0 | 0 | 48753 |
| Spatial_MILP | 14 | 14 | 0 | 14 | 14 | 46 | 0 | 0.0178323 | 48624 |

修复前日志的真实统计：

| strategy | status4 | timeout | numeric_round_rejections |
| --- | --- | --- | --- |
| E-Full | 14 | 0 | 14 |
| E-Spatial | 14 | 0 | 14 |
| Greedy | 0 | 0 | 0 |
| MPC | 0 | 0 | 0 |
| Spatial_MILP | 14 | 0 | 14 |

成功恢复数若为零，表示该实际样本中遇到的固定整数均不可行，不能写成恢复成功。新增可恢复合成例、LP超时、LP假成功、无剩余预算、纯整数无自由度、不可恢复原反例等测试全部通过。这是软件恢复能力的证据，不是实际数据已发生可恢复事件的证据。

受影响批次见 results/core/numerical_affected_batches.csv；完整原矩阵、边界、原解与恢复向量见 audit/recovery_candidates。逐请求改动见 before_after_changed_requests.csv.gz，逐事件类别对照见 before_after_event_counts.csv，按事件/请求ID/出现次序对齐的完整事件变化见 before_after_changed_events.csv.gz。不同调用可复用相同证据哈希，但调用次数分别统计；不能把调用或请求数当独立统计样本。actual_native_milp_calls按实际子问题计数并排除singleton枚举；旧账本highs_calls按顶层引擎标签计数，二者不可混用。

各阶段 numerical_affected_request_outcomes.csv.gz 逐个列出非法候选所在批次请求的新旧地区、派发、开始、完成、fallback及失败原因，并区分该分支是否实际被选择。未选分支的数值失败也保留，不能把其全部请求称为实际被迫回退。

## 时间与结果变化

完整子问题 wall 包括恢复与验证，LP wall/iterations 单列，原预算不增加。理想控制核心部分采用独立 worker 并行，因此其测得 wall 只作描述，不用于声称前后速度因果变化；H 独占串行重算其实际控制耗时。新旧计时不同是预期现象。下表是核心各指标最大绝对差：

| metric | max_absolute_new_old_difference |
| --- | --- |
| co2e_g | 0 |
| completion_rate | 0 |
| controller_co2e_g | 0 |
| controller_kwh | 0 |
| energy_kwh | 5.68434189e-14 |
| fallback_count | 0 |
| idle_co2e_g | 0 |
| incremental_co2e_g | 7.27595761e-12 |
| mean_intentional_delay_s | 7.57306469e-29 |
| mean_queue_wait_s | 0 |
| mean_solver_seconds | 0.00249789703 |
| p50_response_s | 4.4408921e-16 |
| p95_response_s | 0 |
| p99_response_s | 0 |
| service_violation_rate | 0 |
| solver_nonoptimal_count | 0 |
| solver_timeout_count | 0 |
| solver_wall_seconds | 11.0157259 |

核心新增/更改终态请求总计 0 个策略-种子-请求记录（非独立样本）。即使部分数值相同，也属于新恢复协议的实际重跑，不称作旧版本的精确复现。F/H另有各自完整前后差异CSV。首轮pilot出现统计字段holding_s缺省的KeyError，日志及失败目录保留；汇总适配后重跑，未改模型，未纳入正式样本。
