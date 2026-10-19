from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
FIG = ROOT / 'figures'
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'Arial', 'pdf.fonttype': 42,
                     'ps.fonttype': 42, 'svg.fonttype': 'none'})
INK = '#263947'
GRAY = '#677782'
LINE = '#667986'
COLORS = ['#436F91', '#3E7B6A', '#AA7C37', '#7A658B']
TINTS = ['#EDF3F8', '#EDF5F1', '#FBF4E8', '#F4EFF7']

class Chart:
    def __init__(self, height):
        self.h = height
        self.fig, self.ax = plt.subplots(figsize=(7.5, 7.5 * height / 1000))
        self.fig.subplots_adjust(0, 0, 1, 1)
        self.ax.set(xlim=(0, 1000), ylim=(height, 0))
        self.ax.axis('off')
        self.labels = []

    def text(self, x, y, s, size=8.5, weight='normal', color=INK, ha='center', va='center'):
        artist = self.ax.text(x, y, s, fontsize=size, fontweight=weight, color=color,
                             ha=ha, va=va, linespacing=1.35, zorder=5)
        self.labels.append(artist)
        return artist

    def rect(self, x, y, w, h, fill='white', edge=LINE, radius=7, dashed=False, lw=.8):
        p = FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                          facecolor=fill,edgecolor=edge,linewidth=lw,
                          linestyle=(0,(4,3)) if dashed else '-', zorder=2)
        self.ax.add_patch(p)

    def node(self, x, y, w, h, s, i=0, size=8.5, fill=None, dashed=False, weight='normal'):
        self.rect(x,y,w,h,fill or 'white',COLORS[i],dashed=dashed)
        self.text(x+w/2,y+h/2,s,size,weight)

    def panel(self,x,y,w,h,i,title,subtitle=None):
        self.rect(x,y,w,h,TINTS[i],COLORS[i],dashed=True)
        self.rect(x+9,y+9,w-18,42,COLORS[i],COLORS[i],radius=4)
        self.text(x+22,y+30,f'{i+1:02d}',9.5,'bold','white',ha='left')
        self.text(x+w/2+12,y+30,title,9,'bold','white')
        if subtitle:
            self.text(x+w/2,y+72,subtitle,8,color=COLORS[i])

    def arrow(self, points, color=LINE, dashed=False, lw=1.05):
        for a,b in zip(points[:-2],points[1:-1]):
            self.ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,
                         linestyle=(0,(4,3)) if dashed else '-',zorder=3)
        p=FancyArrowPatch(points[-2], points[-1],arrowstyle='-|>',mutation_scale=8,
                          linewidth=lw,color=color,linestyle=(0,(4,3)) if dashed else '-',zorder=3)
        self.ax.add_patch(p)

    def title(self, subtitle):
        self.text(500,25,'Our Work',14,'bold')
        self.text(500,53,subtitle,8.5,color=GRAY)

    def save(self,name):
        self.fig.canvas.draw()
        renderer=self.fig.canvas.get_renderer()
        # Confirm every label remains inside the exported canvas.
        bounds=self.fig.bbox
        for artist in self.labels:
            b=artist.get_window_extent(renderer)
            if b.x0 < -1 or b.y0 < -1 or b.x1 > bounds.width+1 or b.y1 > bounds.height+1:
                raise ValueError(f'Text outside canvas: {artist.get_text()}')
        for ext in ['pdf','svg','png']:
            self.fig.savefig(FIG/f'{name}.{ext}',dpi=320,facecolor='white')
        plt.close(self.fig)


def dependency():
    c=Chart(885)
    c.title('A unified path from performance evidence to deployment decisions')
    c.node(25,83,950,55,'Public benchmarks  |  Workload traces  |  Energy evidence  |  Grid-carbon data',size=8.8,fill='#F3F6F8')
    c.arrow([(252,138),(252,168)])
    c.panel(25,168,455,258,0,'Performance evaluation','Task 1  /  Construct the evidence base')
    c.node(44,262,206,70,'Pre-inference quality\nPre-LightGBM',0)
    c.node(263,262,197,70,'Post-response evidence\nValidity + agreement',0)
    c.node(44,348,416,53,'Cost model + queue / power simulation\nQuality  |  Cost  |  Latency  |  Carbon',0)
    c.panel(535,168,440,258,1,'Adaptive routing','Task 2  /  Select the model')
    c.node(555,262,400,55,'Predicted quality gain + calling cost',1)
    c.arrow([(755,317),(755,336)],COLORS[1])
    c.node(555,336,400,65,'Finite-sample risk guard + task fallback\nSelected policy: direct Pre-LightGBM',1)
    c.arrow([(480,296),(535,296)],COLORS[0])
    c.text(507,274,'inputs',7,color=GRAY)
    c.arrow([(755,426),(755,476)],COLORS[1])
    c.text(839,451,'Freeze S/L choices',7.8,color=COLORS[1])
    c.panel(535,476,440,259,2,'Carbon-aware scheduling','Task 3  /  Allocate execution resources')
    c.node(555,570,400,53,'Observed carbon + slots + deadlines',2)
    c.arrow([(755,623),(755,643)],COLORS[2])
    c.node(555,643,400,66,'Spatial MILP  ->  DES execution\nRegion, queue, response time and emissions',2)
    c.panel(25,476,455,259,3,'Robustness & sustainability','Task 4  /  Evaluate risks and net benefits')
    c.node(44,570,416,53,'Task-mix shift  |  Prediction noise\nDemand growth',3)
    c.arrow([(252,623),(252,643)],COLORS[3])
    c.node(44,643,416,66,'Task-level fairness + rebound analysis\nFacility-total carbon + evidence audit',3)
    c.arrow([(535,607),(480,607)],COLORS[2])
    c.text(507,582,'replay',7,color=GRAY)
    c.arrow([(252,735),(252,766),(500,766),(500,790)],COLORS[3])
    c.node(110,790,780,55,'Adoption conditions, trade-offs and recommendations to SWITCH',3,size=9,fill=TINTS[3],weight='bold')
    c.text(500,869,'Compared separately: model cascades, feedback updates and temporal scheduling.',7.8,color=GRAY)
    c.save('A_dependency_roadmap')


def columns():
    c=Chart(650)
    c.title('Assess performance  ->  Route requests  ->  Schedule execution  ->  Verify benefits')
    c.node(20,80,960,46,'Shared evidence: benchmarks, request traces, resource assumptions and regional carbon signals',size=8.5,fill='#F3F6F8')
    xs=[20,265,510,755]
    titles=['Performance','Routing','Scheduling','Validation']
    subs=['Task 1 / Evidence','Task 2 / Model choice','Task 3 / Region & time','Task 4 / Net impact']
    content=[
        [('Quality estimation','Pre-LightGBM\nResponse assessment'),('Resource accounting','Calling cost\nQueue + power model'),('Evaluation outputs','Quality and cost\nLatency and carbon')],
        [('Candidate comparison','Direct S / L\nSequential cascades'),('Decision safeguards','Quality-cost threshold\nTask-specific risk guard'),('Selected policy','Direct Pre-LightGBM\nFallback to L')],
        [('Frozen model choices','Observed grid carbon\nCapacity + deadlines'),('Spatial MILP + DES','Regional allocation\nQueues and execution'),('Policy comparisons','Greedy\nTemporal alternatives')],
        [('Stress scenarios','Task mix + quality noise\nDemand growth'),('Service and fairness','All-input violations\nTask-level quality / delay'),('Sustainability','Total facility carbon\nRebound + evidence')]
    ]
    for i,x in enumerate(xs):
        c.panel(x,158,225,365,i,titles[i],subs[i])
        for j,(head,body) in enumerate(content[i]):
            y=258+j*83
            c.text(x+112.5,y-9,head,8.25,'bold',COLORS[i])
            c.node(x+10,y+7,205,49,body,i,size=8.2)
            if j<2:c.arrow([(x+112.5,y+56),(x+112.5,y+66)],COLORS[i],lw=.85)
        c.arrow([(x+112.5,126),(x+112.5,158)],COLORS[i],lw=.9)
        if i<3:c.arrow([(x+225,509),(xs[i+1],509)],COLORS[i],lw=1.2)
    # A common bottom spine aggregates evaluation evidence without implying joint optimization.
    for x in xs:
        c.ax.plot([x+112.5,x+112.5],[523,548],color=LINE,lw=.8,zorder=1)
    c.ax.plot([132.5,867.5],[548,548],color=LINE,lw=.8,zorder=1)
    c.arrow([(500,548),(500,570)])
    c.node(110,570,780,45,'Decision support for SWITCH: adoption value, service trade-offs and environmental evidence',3,size=8.6,fill=TINTS[3],weight='bold')
    c.text(500,638,'Historical replay and parameterized simulation; deployment savings require measurement.',7.6,color=GRAY)
    c.save('B_four_column_framework')


def architecture():
    c=Chart(755)
    c.title('A protected routing core with carbon-aware execution and an independent assessment')
    c.node(25,85,950,48,'Inputs: registered request features, historical outcomes, workload arrivals and grid-carbon observations',size=8.5,fill='#F3F6F8')
    c.panel(25,172,245,455,0,'Evidence base','Task 1 / Evaluate')
    c.node(41,273,213,68,'Pre-inference\nquality prediction',0,size=9)
    c.node(41,366,213,68,'Post-response\nquality assessment',0,size=9)
    c.node(41,459,213,68,'Cost, service time\nand power models',0,size=9)
    c.text(147.5,575,'Quality / cost inputs\nResource accounting',8.3,color=COLORS[0])
    c.arrow([(147.5,133),(147.5,172)],COLORS[0])
    c.panel(310,172,380,206,1,'Protected model router','Task 2 / Select S or L')
    c.node(326,268,348,62,'Pre-LightGBM + cost threshold\nRisk guard + fallback to L',1,size=9)
    c.text(500,351,'Selected main path: one direct call',8.3,color=COLORS[1])
    c.arrow([(270,305),(310,305)],COLORS[0])
    c.arrow([(500,378),(500,421)],COLORS[1])
    c.text(610,399,'Frozen model choice',8.3,color=COLORS[1])
    c.panel(310,421,380,206,2,'Carbon-aware executor','Task 3 / Spatial MILP + DES')
    c.node(326,518,156,64,'Regional\nassignment',2,size=9)
    c.node(518,518,156,64,'DES\nexecution',2,size=9)
    c.arrow([(482,550),(518,550)],COLORS[2])
    c.arrow([(595,582),(595,607),(405,607),(405,582)],COLORS[2],dashed=True,lw=.8)
    c.text(500,618,'Observed state for the next decision',7.5,color=COLORS[2])
    c.arrow([(270,493),(289,493),(289,550),(310,550)],COLORS[0])
    c.panel(730,172,245,455,3,'Risk & impact audit','Task 4 / Assess')
    c.node(746,273,213,68,'Task-mix shift\nPrediction noise',3,size=9)
    c.node(746,366,213,68,'Demand growth\nService fairness',3,size=9)
    c.node(746,459,213,68,'Rebound effects\nTotal facility carbon',3,size=9)
    c.text(852.5,575,'Historical quality +\nservice / carbon ledger',8.2,color=COLORS[3])
    c.arrow([(690,550),(730,550)],COLORS[2])
    c.arrow([(852.5,627),(852.5,654),(500,654),(500,680)],COLORS[3])
    c.node(140,680,720,43,'Evidence-based adoption conditions and recommendations to SWITCH',3,size=9,fill=TINTS[3],weight='bold')
    c.text(500,741,'Alternative cascades, feedback updates and temporal policies are evaluated outside the selected core.',7.6,color=GRAY)
    c.save('C_core_and_validation')


TEXT = [
    'We develop a quality-aware routing and carbon-aware scheduling framework for SWITCH. Figure~\\ref{fig:our-work} summarizes four linked stages of our work.',
    'First, we combine public benchmarks, workload traces, energy evidence and grid-carbon data to evaluate response quality, inference cost, latency and emissions. Pre-inference quality prediction and post-response assessment provide complementary evidence for comparing routing policies.',
    'Second, we compare direct and sequential-cascade policies under quality and cost constraints. The selected Pre-LightGBM router uses task-specific finite-sample risk guards and fallback to the more capable model; post-response escalation is not activated in the selected policy.',
    'Third, we keep model choices fixed and formulate regional allocation as a Spatial MILP, with execution evaluated through discrete-event simulation. Comparisons with Greedy and temporal scheduling alternatives examine the trade-offs among carbon emissions, tail latency and service violations.',
    'Finally, we assess robustness under task-mix shifts, prediction noise and demand growth, and examine fairness across request categories, rebound effects and total facility emissions. Together with sensitivity and overhead analyses, these results identify the conditions under which additional routing or scheduling complexity is worthwhile and inform recommendations to SWITCH.'
]
SHORT = ('We organize our work into four linked stages (Figure~\\ref{fig:our-work}). '
         'First, we establish a common framework for quality estimation and cost, latency and carbon accounting. '
         'Second, we compare direct and cascade policies and select a Pre-LightGBM router protected by task-specific risk guards and fallback. '
         'Third, with model choices fixed, Spatial MILP and discrete-event simulation evaluate regional allocation against Greedy and temporal scheduling alternatives. '
         'Finally, stress tests, task-level fairness assessment and rebound analysis examine service risks and total environmental impact. '
         'Sensitivity and overhead analyses help identify when additional complexity is worthwhile and support recommendations to SWITCH.')


def write_text():
    def plain(s):return s.replace('Figure~\\ref{fig:our-work}', 'Figure X').replace('(Figure~\\ref{fig:our-work})','(Figure X)')
    md = '# 1.3 Our Work\n\n' + '\n\n'.join(map(plain,TEXT))
    md += '\n\n**Figure caption:** Overview of our framework: performance evaluation, protected model routing, carbon-aware scheduling, and robustness and sustainability assessment.\n'
    md += '\n## Compact version for a tight introduction page\n\n' + plain(SHORT) + '\n'
    (ROOT/'Our_Work_EN.md').write_text(md,encoding='utf-8')
    (ROOT/'Our_Work_EN.tex').write_text(
        '% Requires graphicx. Keep the figure PDF beside this fragment, or adjust its path.\n'
        '% Use subsection here only if the surrounding manuscript already contains Introduction.\n'
        '\\subsection{Our Work}\n\\label{sec:our-work}\n\n' + '\n\n'.join(TEXT) +
        '\n\n\\begin{figure}[htbp]\n  \\centering\n'
        '  \\includegraphics[width=\\linewidth]{figures/B_four_column_framework.pdf}\n'
        '  \\caption{Overview of our framework: performance evaluation, protected model routing, carbon-aware scheduling, and robustness and sustainability assessment.}\n'
        '  \\label{fig:our-work}\n\\end{figure}\n',encoding='utf-8')
    (ROOT/'content.json').write_text(json.dumps({'paragraphs':list(map(plain,TEXT)), 'short':plain(SHORT)},ensure_ascii=False,indent=2),encoding='utf-8')
    zh='''# Our Work 使用说明与内容核对

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
'''
    (ROOT/'README_ZH.md').write_text(zh,encoding='utf-8')


def comparison():
    names=['A_dependency_roadmap','B_four_column_framework','C_core_and_validation']
    labels=['A  |  Task dependency roadmap','B  |  Four-column framework','C  |  Core and validation']
    canvas=Image.new('RGB',(2500,930),'#E9EEF2')
    font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',24)
    for j,(name,label) in enumerate(zip(names,labels)):
        left=20+j*830
        ImageDraw.Draw(canvas).text((left+15,19),label,fill=INK,font=font)
        im=Image.open(FIG/f'{name}.png').convert('RGB')
        im.thumbnail((790,820),Image.Resampling.LANCZOS)
        canvas.paste(im,(left+(800-im.width)//2,75+(820-im.height)//2))
    canvas.save(ROOT/'comparison.png')


if __name__=='__main__':
    dependency();columns();architecture();write_text();comparison()
    print('Three flowcharts exported as PDF, SVG and PNG; text and comparison ready.')
