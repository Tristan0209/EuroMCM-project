# Task 4 三图重绘交付

日期：2026-10-19。**这是已有验证结果的展示重绘，新增实验、训练和模型推断均为 0。** 原论文、原 PNG、冻结数据和失败日志均保留；没有修改附件 PDF。附件图和 PDF 只作为布局参照，未从像素读数或手工输入结果。

## 1. 输出与使用

| 图 | 400 dpi 白底 PNG | 可编辑 SVG | 矢量 PDF | 原生尺寸 |
| --- | --- | --- | --- | --- |
| Scenario Robustness Dashboard | [PNG](png/Fig1_Scenario_Dashboard.png) | [SVG](svg/Fig1_Scenario_Dashboard.svg) | [PDF](output/pdf/Fig1_Scenario_Dashboard.pdf) | 7.2 x 4.85 in；2880 x 1939 px |
| Facility Carbon and Demand-Growth Contrast | [PNG](png/Fig2_Carbon_Accounting.png) | [SVG](svg/Fig2_Carbon_Accounting.svg) | [PDF](output/pdf/Fig2_Carbon_Accounting.pdf) | 7.2 x 2.9 in；2880 x 1160 px |
| Group-Level Service Trade-offs under Demand Growth | [PNG](png/Fig3_Group_Tradeoffs.png) | [SVG](svg/Fig3_Group_Tradeoffs.svg) | [PDF](output/pdf/Fig3_Group_Tradeoffs.pdf) | 7.2 x 2.65 in；2880 x 1060 px |

三图统一 Greedy = 蓝色 `#2878B5`、Spatial MILP = 橙色 `#E6862A`。共同选模的质量指标及政策差值使用中性色，不给它们附加虚假的单一调度策略归属。英文标题采用中性指标名称。

SVG 保留可编辑 text 和矢量图元，没有嵌入栅格图片；编辑时建议安装 DejaVu Sans，避免替换字体改变排版。PDF 使用嵌入 TrueType 字体和矢量图元。PNG 为白底，不是透明底；满足白底／透明底二选一要求。

建议在 12pt 正文中按整行宽度使用，优先采用矢量 PDF；避免缩成窄小半栏。原生主要标签约 9–10.5pt，数值与图内说明约 8–8.8pt。三图高度分别约 12.3、7.4、6.7 cm；用图3替代原公平性表，减少重复占页。此轮没有重排整篇论文，也不把单图检查声称为整章一定恰好三页。

[英文图注](Figure_Captions_EN.md)包含分母、P95、三 seed、Code 小样本与因果边界说明。[LaTeX 插入片段](figures_for_paper.tex)使用自动图号与唯一 label，不固定章号。用户说明对应 5.2–5.4；所附合稿 PDF 中相关内容已排为 6.2–6.4，插入时随总稿引用即可，不需改变图数据。

## 2. 唯一数值来源和计算

| 图／面板 | 已有来源 | 重绘计算及定义 |
| --- | --- | --- |
| 图1(a) | `../Task4_Simple_Results.csv` 的 `mean_historical_quality` | 同 seed 情景−Baseline 的 Score 差；标签是该情景三个 seed 的绝对均值。逐 seed 确认两策略质量一致后仅绘一次，不重复计为两组质量证据。 |
| 图1(b) | 同表 `answer_pass_rate` | 历史阈值达标率，Score >= 0.5，分母为全部输入；乘100后取同 seed 基线差，单位为百分点，标签为绝对百分比。没有误用 `valid_answer_pass_rate`。 |
| 图1(c,d) | 同表 `p95_response_s`、`service_violation_rate` | 分别取三个 seed 的完成者 P95 均值、全输入违反率均值；违反率乘100后绘对数轴。原始值全为正，没有加入 epsilon、伪计数或删去零值。 |
| 图2(a) | 同表 `co2e_g` | 总设施碳／1000 得 kgCO2e，保留全窗口固定 idle；零起轴。接近的均值用分行、对应颜色的标签表示，避免标签重叠。 |
| 图2(b) | 同表 `incremental_co2e_g` | `total - idle`，包含 active、network 和 controller 原账本增量，不再额外相加；单位 gCO2e，零起轴。该量不是设施总减排百分比。 |
| 图2(c) | 同表逐 seed 的 `co2e_g` | 先计算同需求 Spatial−Greedy、增长 Spatial−原需求 Greedy，再取三 seed 均值。显示精度为三位小数，不手动指定结果。 |
| 图3(a,b,c) | `../final_review/group_service.csv` 的 `violation_rate`、`completed_p95_s`、`valid_pass_rate` | 仅选择 Demand Growth，按 General／Math／Code 和 Greedy／Spatial 分组，先保留 seed 原比率／P95再平均；类别比率分母为该类别全部输入，P95只看完成者。 |

所有标记均为 **3 个配对 seed 的指标均值**，没有把请求合并后重算 P95，也没有计算或绘制置信区间。连接线仅连接政策均值，或从零基线连到情景差值，不是观察范围。完整逐 seed 值、均值和 min/max 仍保存在 [plot_manifest.json](audit/plot_manifest.json)，但 min/max 未画为区间。其 61 组量及 [183 行绘图数据](plot_data.csv)是重表达，不是新增世界或新增独立样本。

图1(a,b) 历史质量变化不能解释为共同选模下的调度质量收益；Math/Code 40% 是任务组成代理。图2(c) 的微小跨需求反转来自外生增长，不代表识别出效率诱导的真实行为反弹。图3注释中的 Code 86 个实例／85 个 prompt 组专指 baseline，不冒充增长情景样本数。

## 3. 复现

绘图代码：[redraw_task4_figures.py](redraw_task4_figures.py)。只依赖 Python 标准库和 matplotlib，不导入第二、第三、第四问模型或实验执行器。附有两份原表的逐字节副本 `source_tables/`；在解压目录可直接运行：

```powershell
python -B redraw_task4_figures.py
```

也可明确指定现有结果目录和新的输出目录：

```powershell
python -B redraw_task4_figures.py --source-dir "D:/EuroMCM/EuroMCM-project/TASK 4/simple_assessment" --output-dir "D:/EuroMCM/EuroMCM-project/TASK 4/simple_assessment/figure_redraw_20261019"
```

默认优先读本目录附带的字节一致副本；没有副本时读上一级原结果。输出始终是绘图目录，不覆盖原 `simple_assessment/figures`。无需重新训练、推断、路由或 DES。

## 4. 验证和保护

状态：**已实现并验证**。独立 [verify_figures.py](verify_figures.py) 不调用绘图代码，使用原 CSV 和 Decimal 重算全部 61 组量，并与前一轮数值审计核对图2三个差值的显示方向和数值；同时检查 PNG 像素／DPI、SVG 可编辑文字与无嵌入栅格、PDF 矢量内容及字体嵌入、三图视觉检查记录和原文件哈希。

独立验证在当前项目运行；它需要上一级冻结表、既有审计和开始快照。解压包可独立重绘，但不把脱离原项目的运行称为完成全部原文件保护核验。

原 TASK 4 的 2669 个既有文件逐 SHA-256 复核，并检查原文件集合没有新增或删除。附件 PDF／可访问的示例图仅作参照，内容保持不变。结果见 [validation.json](audit/validation.json)、[visual_qa.json](audit/visual_qa.json)。全程保留日志；初次布局检查发现图3标题超出画布，已通过换行和调整留白修正，未改数据。

没有尚未验证的新模型或新实验。本轮验证仅支持图形忠实呈现原条件模拟结果，不扩大原研究的可靠性、公平性或环境收益结论。
