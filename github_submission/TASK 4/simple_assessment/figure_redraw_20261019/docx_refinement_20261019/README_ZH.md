# 按 DOCX 说明完成的图表美化版本

本目录保留上一轮要求：第二张图仍分为 a+b 合图、c 单图；所有图片底部不放图注或解释长句。英文图注另见 `Figure_Captions_EN.md`。四份 PDF 同时提供 400 dpi 白底 PNG 和可编辑 SVG。

## 修改对照

| 位置 | 完成修改 |
| --- | --- |
| 图1图例 | 仅保留 Greedy 蓝色、Spatial MILP 橙色；共同选模解释移到单独图注 |
| 图1(a,b) | 坐标仍为基线差，数字也统一为变化量；Score 四位小数，达标率变化三位小数，后者单位为百分点；完整历史阈值达标率名称保留 |
| 图1(c,d) | 配对点保持，P95 仅完成者、违反率全输入；保留明确标注的 log scale |
| 图2(a,b) | 保留总碳零起轴、共同12小时 idle；b 改名 Operational carbon increment；加宽两面板以便阅读 |
| 图2(c) | 改为 Key carbon differences；中性横轴 Difference in total carbon (gCO2e)；逐行区分同需求 Spatial−Greedy 与 Growth Spatial−Baseline Greedy；差值三位小数、零线与中性色保持 |
| 图3 | 三指标名称、Rate (%)／Response time (s) 统一；P95 两位、违反率三位；c 移除拥挤数字，所有点与连接线保留；单独图注明确 under 50% demand growth、各类别全输入／完成者分母 |

数值标注直接由未舍入的 seed 均值格式化，不对旧标签再次舍入，不改变任何坐标、柱高、原始数据或比较方向。三 seed 均值仅支持原有描述性比较，没有增加置信区间或新实验。

## 文件与复现

- `pdf/`：四份矢量 PDF。
- `png/`：四份 400 dpi PNG。
- `svg/`：四份保留可编辑文字的 SVG。
- `Figure_Captions_EN.md`：单独英文图注，交由论文统一编号。
- `refine_figures.py`：本轮修改代码；`plotting_basis.py` 为未修改的原绘图代码副本。
- `source_tables/`：原结果表的逐字节副本，仅供重绘，不是新实验记录。
- `reference_plot_manifest.json`：上一轮61组绘图量的原验证记录。
- `verification.json`：本轮数据、格式、文字与视觉核验。

在解压目录运行 `python -B refine_figures.py` 即可重绘，依赖 matplotlib 和 Python 标准库。不导入或执行上游训练、推断和 DES。DOCX 只作为本轮展示修改说明；原始模型、数据、结果和原版本图表不修改。
