# 第三问验证报告

上游21项接口测试通过，系统测试25项，失败/错误0项。完整Phase0见results/audit/phase0.json；pytest明细见results/audit/tests.xml。

最终表引用336个已完成配置/重复运行，每次执行了输入ID守恒、S/L冻结、地区允许、执行不早于到达、单次非抢占执行、物理副本不重叠、容量准入、响应时间分解、公共随机条件、能量分解、MPC已到达集合与过去观测边界核查。passed表示软件和账本一致，不表示每个情景没有SLA失败。

新版手算例：2个10秒L任务，DE/FR/ES碳强度300/30/200 gCO2e/kWh，单槽位；两任务在FR顺序执行，增量设施功率为585×1.2=702W，增量碳为20×702/3.6e6×30=0.117g，增量电量0.0039kWh；200秒内6台空闲底座为1.16kWh。跨小时3590至3610秒、100至200强度积分为3000，增量碳0.585g。原75W增量手算测试仍作为legacy小型接口算例保留，不用其参数生成新版实验。另验证一台主机增加到2个准入槽位时不会重复计空闲，以及不可用不等于断电。

完成12小时共同窗口检查和100/500/1000/7335四种规模门槛；Oracle是批次候选受限基准，不具备全问题全局最优证书。求解器真实状态：

```json
{
  "completed_unique_runs_used": 336,
  "solver_decisions": 130217,
  "status_counts": {
    "0": 130026,
    "2": 188,
    "1": 3
  },
  "max_variables": 1354,
  "max_constraints": 2434,
  "max_reported_relative_gap": 6.547364336001783e-07,
  "restriction": "finite candidates; chronological offline Oracle batches; singleton enumeration separately logged"
}
```

当前版本上游563项保护哈希未变。原327项历史保护记录与旧功率实验已归档，不能将旧哈希宣称为当前版本核验。真实碳值26280行，关键列无缺失/重复/非小时；DE/FR/ES提供方估计标志12/5/27小时原样保留。生命周期与direct独立运行，使用提供方CO2eq，不将它转换或声称为纯CO2。完整请求除estimated_energy_kwh外与旧版字段完全一致；全部保存Pre预测、S/L路由、训练长度服务代理重放通过。新输入指纹覆盖当前上游配置、源码、权重、路由、功率合同及物理主机表；上游变更时必须先audit，不能复用旧功率缓存。

pending/not_verified：精确历史发布版本和快照GWP因子版本、匹配硬件/网络/控制功率、独立外部质量留出。没有缺失碳强度时静默填0、免费真实回答、未来actual进入MPC或未来请求进入MPC。Perfect-future只允许Oracle路径。已有SLA失败如实保留，不能当作测试跳过。
