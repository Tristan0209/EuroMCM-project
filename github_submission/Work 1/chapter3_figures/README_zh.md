# 第三章三张结果图

按用户最终要求，只交付改后图的 PDF，不修改文章。

- `fig_3_2_evidence_value.pdf`：回答评价的收益与调用代价。
- `fig_3_3_cost_structure_boundary.pdf`：成本构成、价格响应与折旧边界。
- `fig_3_4_latency_energy_carbon.pdf`：响应时间、电量构成与地区排放。

同目录保留 SVG、300 dpi PNG、重建代码及 `data/`，便于自行使用。

运行 `python plot_task1_figures.py`（有 python3 的环境亦可用 `python3`）。脚本只读取旁边的 data 目录，不重新训练或执行仿真。依赖见 requirements.txt。

质量数据取自用户指定的正文汇总值；VC11 AUC 留空，不补误差条或显著性结论。

成本数据来自本地 `Work 1/outputs/cost_summary.csv` 的 27 个 760 W 情景，关键费用与用户请求一致，导出保留六位小数。整机折旧由原份额折旧乘 `8/c` 推算；价格响应是固定轨迹、零维护下的解析计算，不是功率扫描，结果受源表舍入精度限制。采购价和电价扰动大小不同，不能直接作为等比例敏感性比较。

环境 summary、runs 和 manifest 来自 `Work 1/task1_response time, and environ mental impact/outputs`，未使用电量口径不同的 `latency_environment/outputs`。18 个配置各五次运行；碳因子的重复记账行不作为独立运行。始终使用同一台通电的八卡服务器，共享 1500 W、GPU 空闲 175 W、忙碌 760 W、PUE 1.2。空心点按平均 P95 和晚响应比例共同判定，排放面板仅使用 Ember CO2e 年度因子。

电量构成用各次时长及总电量核算，再按平均服务时间独立验证。数据与恒等式检查见 validation_report.json，来源及哈希见 data/provenance.json。这些是三部分已有结果，不是新执行的联合实验。
