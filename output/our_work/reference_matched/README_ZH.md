# 三张样图逐一对应的重绘版

本轮按三张原图的具体结构重画，保留每张各自的布局语法，不沿用上一版的通用四任务卡片。图中内容同时来自 EuroMCM 论文和四个工作文件夹，按主线挑选模型、分析和实验。

| 输出 | 样图与保留结构 | 重点材料 |
|---|---|---|
| 01_based_on_2400996 | 第5页 Figure 3；左边两块纵向模型框、右边错位 Task 框、空心 apply 箭头、底部整宽 Task 4；粉色标题、绿色节点、蓝色指标与黑色虚线 | Pre-LightGBM/IRT/MIRT、V/VC2、CP+Bernstein、Q/C/L/E、Spatial+DES与压力/反弹 |
| 02_based_on_2025_O_award | 第5页 Figure 2；上方总标题、左侧三列主体、右侧上下分析栏、底部综合和应用栏；米白/淡黄/浅橙及金色箭头 | 质量与路由、成本折旧与服务评估、Spatial MILP+DES、压力/公平/反弹/适用边界 |
| 03_based_on_2300348 | 印刷第4页（PDF物理第5页）Figure 2；左侧贯穿全高链条、上中粉色分析、上右紫色关系、下中蓝色模型、下右绿色评价；绿色虚线分区 | 质量预测/风险路由、四类度量、压力/反弹、区域调度、服务公平及证据审计 |

每张均含矢量 PDF、文字可编辑 SVG、320 dpi PNG。带 comparison 的 PNG 左边为样图截取、右边为对应新图，便于逐一检查相似程度。插入正文时使用不带 comparison 的 PDF / SVG / PNG。

## 工作材料选取

- Work 1/task1_quality_pre：LightGBM、LightGBM+IRT、LightGBM+MIRT；图中写为基线与测试变体，不把所有模型并入最终流程。
- Work 1/task_1_quality_post：V 为回答有效性评估，VC2 为额外回答的一致性证据；属于已研究的回答后评估与级联比较。
- Work 1/task1_cost.py、cost_dgx_engine.py、响应时间/环境 outputs：历史模型调用成本和自建服务器折旧分开，排队/功率/PUE/碳信号用于相应核算。
- task2/runs/work1_integrated_20261016/RESULTS_zh.md、routing.py：质量收益-成本阈值、prompt分组与OOF、Clopper-Pearson (CP) 与经验Bernstein上界、分任务强模型回退。最终策略为直接Pre-LightGBM；Math/Code与未注册来源回退L，级联和反馈更新作为比较，不能当作最终已开启的模块。
- TASK 3/task3_model/task3_final_paper/academic_revision_20261018/Task3_Paper_EN_Revised.md：固定S/L后，以Spatial MILP分配区域，用DES评估；比较Greedy、Full MPC及受控时间调度。使用已观测碳信号、可用容量和deadline，不让未来实际值进入决策。
- TASK 4/simple_assessment/academic_revision_20261018/Task4_Paper_EN_Revised.md：任务构成、预测噪声、需求增长；任务/服务类别的质量与时延风险；反弹和设施总碳排放；部署证据、开销和适用边界。

## 边界与缩写

历史回放和参数化仿真不等于生产系统实测。未把API成本、自建成本和碳核算混成一个加权最优目标。没有把MIRT、VC2或时间延迟调度写成最终选择；没有填入未经验证的数值结论。三个图的任务编号对应你的四问，而非范文任务的原义。

CP = Clopper-Pearson；OOF = out-of-fold；CI = carbon intensity；PUE = power usage effectiveness；MILP = mixed-integer linear programming；DES = discrete-event simulation；S/L = lightweight/more capable model；Q/C/L/E = quality/cost/latency/environmental impact。

build_reference_matched.py 为重绘代码；check_reference_matched.py 为文字边界/导出检查与打包代码。正文图中模型和证据的细节按不同样图的容纳量有所取舍。
