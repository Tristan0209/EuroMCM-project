# Our Work 使用说明与内容核对

已提供英文正式版、英文紧凑版、中文译文、LaTeX 插入片段，以及三套流程图的 PDF / SVG / 320 dpi PNG。推荐先比较 comparison.png，再用 Our_Work_Review.pdf 检查实际排版。

## 中文对照（对应英文正式版）

我们面向 SWITCH 构建质量感知路由与碳感知调度框架。流程图概括了本研究相互衔接的四个阶段。

首先，结合公开基准、请求轨迹、能耗证据和电网碳强度数据，评估回答质量、推理成本、时延及碳排放。推理前质量预测与回答后评估为路由策略的比较提供互补证据。

其次，在质量与成本约束下比较直接路由和顺序级联。最终选择的 Pre-LightGBM 路由器采用分任务的有限样本风险保护与强模型回退；最终策略未启用回答后升级。

随后，固定模型选择，将区域分配建模为 Spatial MILP，通过离散事件仿真评估执行结果。与 Greedy 及时间调度备选策略的比较用于分析碳排放、尾部时延及服务违规之间的权衡。

最后，通过任务构成变化、预测噪声与需求增长情景检验稳健性，并分析请求类别间的公平性、反弹效应及设施总排放。结合敏感性与开销分析，识别额外路由或调度复杂度何时值得采用，为 SWITCH 提供建议。

## 三个版本怎么选

| 版本 | 版式与参考 | 适用情况 |
|---|---|---|
| A_dependency_roadmap | 借鉴 2400996 第5页 Figure 3 的模型依赖关系、虚线分区、柔和色块；采用顺时针任务链 | 希望详细说明四问之间的先后依赖，可分配约半页到三分之二页 |
| B_four_column_framework | 借鉴 2502617（2025美赛O奖(1)）第5页 Figure 2 的并列栏目与底部综合输出 | 最接近常见美赛 Our Work，总览均衡，建议作为正文首选 |
| C_core_and_validation | 借鉴 2300348 印刷第4页（PDF物理第5页）Figure 2 的异形模块组合，突出中间路由和执行核心 | 希望突出论文技术贡献，清楚区分决策核心、评估基础和风险检验 |

三版重绘为本论文的原创技术路线，未复制范文的具体文字、公式、模型或结论。颜色分别表示任务1、2、3、4；实线箭头表示主要信息传递，C版内部虚线箭头仅表示 DES 已观测状态用于下一次调度。

## 内容依据与取舍

- `C:/Users/吴连杰/Downloads/EuroMCM.pdf`：第3页任务定义和 Our Work 空缺；第3、4、5章的术语与模型连接关系。其 Writing guide 仅视为现有稿件占位说明，不当作用户的新指令。
- `Work 1/chapter3_figures/README_zh.md`、`Work 1/task1_response time, and environ mental impact/outputs/RESULTS_zh.md`、质量/成本脚本及 EuroMCM 第3章：预测、后验回答评估、成本、排队与功率模型。ML.ENERGY 是独立能耗证据，不声称已校准目标服务的整机功率。
- `task2/runs/work1_integrated_20261016/RESULTS_zh.md`、`task2/routing.py` 和 EuroMCM 第4章：最终方案为直接 Pre-LightGBM + 分任务风险回退。级联与反馈更新有比较，但最终策略不启用级联，在线更新默认关闭。
- `TASK 3/task3_model/task3_final_paper/academic_revision_20261018/Task3_Paper_EN_Revised.md`：固定上游 S/L，Spatial MILP + DES 为主；Greedy、Full MPC、E-Spatial/E-Full 为比较。模型选择、时间、地点不应写成已完成全局联合优化。
- `TASK 4/simple_assessment/academic_revision_20261018/Task4_Paper_EN_Revised.md`：实际完成的三类压力情景为任务构成、预测噪声、需求增长；公平性限于请求类别；反弹为情景分析，不是识别出的实际需求行为。
- 仿真估计、API历史成本、全设施总碳排放使用各自边界；本节不新增数值、不声称已测得生产部署减排。
- EuroMCM 第6、7章仍有占位文字，任务4概述来自文件夹中的最终修订材料。正式合稿时应相应补全章节；本次仅交付 Our Work 文稿和流程图。

## 插入论文

推荐 B 版 PDF，保留矢量文字与箭头；SVG 适合继续编辑；PNG 适合预览或直接插入 Word。英文正式版约 180 词，图放不下时使用约 100 词的紧凑版。A版所需纵向空间较大。以正文实际图号替换 Markdown 的 Figure X，LaTeX 通过引用自动编号。

Our_Work_EN.tex 默认引用 figures/B_four_column_framework.pdf。改用 A 或 C 时替换文件名即可。建议使用 graphicx，以 linewidth 为图宽。图中字母 S/L 与论文一致，分别表示轻量和较强模型。MILP = mixed-integer linear programming；DES = discrete-event simulation。

## 可复现与校验

build_figures.py 生成三版矢量图、PNG 和英文文稿；build_review.py 生成四页审阅 PDF。图中不含数值结果，无需重跑训练或付费 API。已检查所有图中文字在画布内、三个矢量 PDF 可以解析，最终审阅 PDF 和各图经过渲染检查。
