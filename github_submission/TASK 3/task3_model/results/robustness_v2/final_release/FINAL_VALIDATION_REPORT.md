# 最终验证报告

| item | status | evidence |
| --- | --- | --- |
| Existing 70 and new 13 tests | PASS | audit/tests.xml; 83 passed; initial failed test attempt retained |
| All original versions and inputs protected | PASS | 11698 protected file hashes unchanged |
| A2 DES physical semantics unchanged | PASS | v2/des.py byte-identical to A2; same-time capacity tests pass |
| Full rounded/recovered model feasibility | PASS | original absolute threshold 1e-5; LP 1e-7; reject on any failure |
| Saved counterexample reproduction and safe rejection | PASS | LP infeasible at fixed integers, independently derived 0.0037348245s contradiction, rejected; not reported as successful recovery |
| Every raw numerical incumbent recoverable | FAIL | genuinely infeasible rounded assignments remain; counts in numerical_recovery_totals.csv; no such candidate executed |
| Core 7335 x 24 x 5 new executions | PASS | 120 new core cases, paired IDs and common shocks |
| Deterministic same-environment repeats | PASS | 5 policies x preselected seed 2027101806 x 3 total executions; terminal digests identical, numeric atol=1e-8 |
| Request/capacity/shock/energy/carbon/causality audits | PASS | 249 unique selected runs; audit/artifact_invariants.csv |
| Affected F/H reruns | PASS | F 72 references and H 72 references; first six seeds; shared configurations reused without increasing sample size |
| No failed execution hidden | PASS | run_inventory.csv includes all attempts, with 1 failed/incomplete attempts preserved and excluded from independent-seed counts |
| Pure temporal carbon benefit independently established | NOT VERIFIED | persistence nominal future is flat; nonzero hold can rescue rejected zero branch; no claim of temporal dominance |
| All requests satisfy realized hard SLA | FAIL | nonzero observed violations retained; nominal feasibility is not a stochastic guarantee |
| Strict nominal-to-actual queue equivalence | NOT VERIFIED | nominal reservations/FCFS uncertainty and existing in-flight reservation approximation retained |
| Native fixed work/thread/random-seed enforcement | NOT VERIFIED | SciPy1.9.1/HiGHS1.2.2 interface limitation; ample budget and empirical repeats instead |
| Exact measured-H wall/trajectory reproduction | NOT VERIFIED | CPU duration depends on local platform/load; H is fresh serial closed-loop simulation |
| Full fresh one-command reproduction in another directory | NOT VERIFIED | individual pipeline stages actually executed; --verify checked at delivery; full reproduction entry provided but not redundantly rerun end-to-end |
| F/H 24 independent seeds | NOT VERIFIED | six-seed original scope explicitly retained |
| Physical DGX/service/network measurements | NOT VERIFIED | frozen DGX proxy; not calibrated operational hardware |
| Operational as-of forecast / real deployment savings | NOT VERIFIED | 2025 retrospective actual snapshots and causal persistence baseline, synthetic arrivals |
| Unrestricted global Oracle optimum | NOT VERIFIED | not claimed; finite candidate/batch strategy retained, not rerun here |

软件准入及账本门槛 PASS。上述 FAIL 指“所有数值候选均可恢复”和“所有实际请求均满足硬 SLA”两项事实命题不成立，不能改成通过。它们不表示非法候选已执行。最终冻结仅覆盖正确执行、显式拒绝与失败统计完整的条件仿真软件，不认证零风险 SLA 或现实减排。
