"""Rebuild three Our Work diagrams against their individually assigned references."""
from pathlib import Path
import json, zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from PIL import Image, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parent
ROOT.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})

class Diagram:
    def __init__(self,h,font='Arial',width=8.5):
        self.h=h;self.font=font;self.texts=[];self.node_labels=[]
        self.fig,self.ax=plt.subplots(figsize=(width,width*h/1000))
        self.fig.subplots_adjust(0,0,1,1)
        self.ax.set(xlim=(0,1000),ylim=(h,0));self.ax.axis('off')
    def text(self,x,y,s,size=10,bold=False,color='#171717',ha='center',va='center',family=None,italic=False):
        a=self.ax.text(x,y,s,fontsize=size,fontweight='bold' if bold else 'normal',
            color=color,ha=ha,va=va,fontfamily=family or self.font,
            fontstyle='italic' if italic else 'normal',linespacing=1.13,zorder=6)
        self.texts.append(a);return a
    def box(self,x,y,w,h,fill='white',edge='#303030',lw=1,rounded=0,dashed=False):
        if rounded:
            p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={rounded}',
               facecolor=fill,edgecolor=edge,lw=lw,linestyle=(0,(4,2)) if dashed else '-',zorder=2)
        else:p=Rectangle((x,y),w,h,facecolor=fill,edgecolor=edge,lw=lw,
                         linestyle=(0,(4,2)) if dashed else '-',zorder=2)
        self.ax.add_patch(p)
    def node(self,x,y,w,h,s,fill='#D7E8D2',size=10,bold=True,edge='none',rounded=5,color='#171717'):
        self.box(x,y,w,h,fill,edge,rounded=rounded)
        label=self.text(x+w/2,y+h/2,s,size,bold,color)
        self.node_labels.append((label,w,h))
    def line(self,points,color='#171717',lw=1,dashed=False,z=3):
        self.ax.plot([p[0] for p in points],[p[1] for p in points],color=color,lw=lw,
            linestyle=(0,(4,3)) if dashed else '-',zorder=z)
    def arrow(self,points,color='#171717',lw=1,dashed=False,head=8):
        if len(points)>2:self.line(points[:-1],color,lw,dashed)
        self.ax.add_patch(FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
            mutation_scale=head,color=color,lw=lw,linestyle=(0,(4,3)) if dashed else '-',zorder=4))
    def frame(self,x,y,w,h,title,tagw=None):
        self.box(x,y,w,h,'none','#262626',lw=1.4,dashed=True)
        tw=tagw or min(w-24,max(130,len(title)*6.3))
        self.node(x+15,y-15,tw,31,title,fill='#F7CFCF',size=10.5)
    def task(self,x,y,w,h,n):
        self.box(x,y,w,h,'none','#262626',lw=1.3,dashed=True)
        self.node(x+14,y-16,86,31,f'Task {n}',fill='#EF6963',size=11,rounded=0)
    def apply(self,x1,x2,y):
        self.ax.add_patch(FancyArrowPatch((x1,y),(x2,y),arrowstyle='simple,head_length=14,head_width=17,tail_width=7',
            facecolor='white',edgecolor='#243F68',linewidth=1.15,mutation_scale=1,zorder=5))
        self.text((x1+x2)/2,y-17,'apply',9.3,True)
    def gradient(self,x,y,w,h,colors):
        from matplotlib.colors import to_rgb
        cols=np.array([to_rgb(v) for v in colors]);positions=np.linspace(0,len(cols)-1,1000)
        a=np.stack([np.interp(positions,np.arange(len(cols)),cols[:,k]) for k in range(3)],axis=1)[None,:,:]
        self.ax.imshow(a,extent=(x,x+w,y+h,y),aspect='auto',zorder=1)
    def save(self,name):
        self.fig.canvas.draw();r=self.fig.canvas.get_renderer();b=self.fig.bbox
        xscale=b.width/1000;yscale=b.height/self.h
        for label,w,h in self.node_labels:
            q=label.get_window_extent(r)
            factor=min(1,w*xscale*.94/q.width,h*yscale*.90/q.height)
            if factor<1:label.set_fontsize(label.get_fontsize()*factor)
        self.fig.canvas.draw();r=self.fig.canvas.get_renderer()
        for t in self.texts:
            q=t.get_window_extent(r)
            if min(q.x0,q.y0)<-1 or q.x1>b.width+1 or q.y1>b.height+1:
                raise ValueError('Label exceeds page: '+t.get_text())
        for ext in ['pdf','svg','png']:
            self.fig.savefig(ROOT/f'{name}.{ext}',dpi=320,facecolor='white')
        plt.close(self.fig)


def template_2400996():
    d=Diagram(1200,width=8.1)
    g='#D7E8D2';blue='#D9E7FC'
    # The two tall model blocks and the offset task panels follow Figure 3.
    d.frame(155,45,420,355,'Quality Prediction and Assessment',375)
    d.box(166,77,398,115,'none','#262626',lw=1.2,rounded=5,dashed=True)
    d.node(183,91,133,84,'Pre-inference\nrequest\nfeatures',g,size=10.6,rounded=17)
    d.node(415,91,133,84,'Pre-LightGBM\nIRT / MIRT\nvariants tested',g,size=10.2,rounded=17)
    d.arrow([(316,133),(415,133)]);d.text(365,117,'predict',9.2,True)
    d.node(18,98,110,32,'Prompt length',g,size=9)
    d.node(18,146,110,32,'Task label',g,size=9)
    d.arrow([(128,114),(183,114)]);d.arrow([(128,162),(183,162)])
    d.node(290,213,226,47,'Predicted quality gain',g,size=10.2)
    d.arrow([(480,175),(480,205),(403,205),(403,213)])
    d.box(166,281,398,102,'none','#262626',lw=1.2,rounded=5,dashed=True)
    d.node(180,299,103,67,'Validity\nV',g,size=10.2,rounded=12)
    d.node(447,299,104,67,'Agreement\nVC2',g,size=10.2,rounded=12)
    d.arrow([(283,313),(447,313)])
    d.node(291,326,149,37,'Post-assessment',g,size=9.5)
    d.text(365,299,'extra response',8.3)
    d.text(366,375,'Evaluated for cascade candidates',8.5)

    d.task(675,234,300,204,2)
    d.node(691,267,268,35,'Quality-cost threshold',g,size=10.1)
    d.arrow([(825,302),(825,315)])
    d.node(691,315,268,48,'Risk guard: CP + Bernstein\nTask-specific fallback to L',g,size=9.6)
    d.arrow([(825,363),(825,376)])
    d.node(691,376,268,43,'Direct Pre-LightGBM selected\nCascades compared separately',g,size=9)
    d.apply(575,675,269)

    d.frame(155,441,420,491,'Quality-Cost-Latency-Carbon Framework',390)
    for x,w,t in [(174,91,'Quality'),(274,80,'Cost'),(363,90,'Latency'),(462,95,'Carbon')]:
        d.node(x,468,w,29,t,blue,size=10,color='#246EAD')
    d.node(168,509,116,29,'Indicators',fill='#CDCDCD',size=10.3)
    d.box(168,540,393,155,'none','#262626',rounded=8,dashed=True,lw=1.2)
    d.node(175,551,299,31,'Pre prediction and post assessment',g,size=9.5)
    d.node(511,551,32,31,'Q',blue,size=10.5,color='#287EC5')
    d.node(185,591,362,30,'LightGBM / IRT / MIRT; V and VC2',g,size=9.4)
    d.node(175,637,299,30,'Model-call cost / server depreciation',g,size=9.1)
    d.node(511,637,32,30,'C',blue,size=10.5,color='#287EC5')
    d.frame(168,718,393,196,'Latency and Emissions Accounting',344)
    d.node(180,746,365,39,'Azure trace + queueing / service model',g,size=9.2)
    d.arrow([(362,785),(362,799)])
    d.node(180,799,365,39,'P95 response time and service violations',g,size=9.1)
    d.arrow([(362,838),(362,850)])
    d.node(180,850,365,48,'Idle + active + network + control\nFacility electricity x grid carbon intensity',g,size=9.1)

    # The reference's left-side input branch is retained for trace and carbon evidence.
    d.node(11,556,74,34,'Trace',g,size=10)
    d.node(11,621,74,34,'Grid CI',g,size=10)
    d.arrow([(85,573),(124,573)],head=6)
    d.arrow([(85,638),(124,638)],head=6)
    d.arrow([(124,573),(124,761),(168,761)],head=7)
    d.text(55,704,'Resource\nand carbon\ninputs',8.2,True)

    d.node(724,481,207,31,'Scheduling alternatives',g,size=9.5)
    d.line([(730,522),(730,517),(922,517),(922,522)])
    d.node(706,530,104,34,'Greedy',g,size=10)
    d.node(842,530,113,34,'Spatial',g,size=10)
    d.line([(758,564),(758,585),(896,585),(896,564)])
    d.arrow([(827,585),(827,608)])
    d.node(646,608,330,43,'Frozen model choices + common DES',g,size=9.6)
    d.arrow([(825,438),(825,465)])
    d.task(675,694,300,96,3)
    d.node(689,721,271,47,'Spatial MILP + DES\nCarbon-latency-service trade-offs',g,size=9.5)
    d.apply(575,675,729)
    d.arrow([(827,651),(827,694)])
    d.task(675,832,300,100,1)
    d.node(690,859,175,51,'Evaluate and compare\nmatched policies',g,size=9.1)
    for j,(t,col) in enumerate([('Q',0),('C',1),('L',2),('E',3)]):
        d.node(888+j%2*35,855+j//2*32,29,27,t,blue,size=9.5,color='#287EC5')
    d.apply(575,675,869)
    d.arrow([(976,630),(989,630),(989,960),(216,960),(216,1011)])
    d.arrow([(989,960),(714,960),(714,1011)])
    d.text(612,949,'Request-level outcomes and complete facility ledger',8.7,True)

    d.task(29,987,946,158,4)
    d.line([(429,1004),(429,1131)],dashed=True,lw=.8)
    d.node(123,1011,250,31,'Robustness and fairness',g,size=9.6)
    d.node(48,1060,365,65,'Task mix + prediction noise + demand growth\nQuality and service risk across request types',g,size=9.2)
    d.node(592,1011,240,31,'Rebound and sustainability',g,size=9.6)
    d.node(451,1060,505,65,'Total facility carbon under increased demand\nSensitivity, overhead limits and deployment evidence',g,size=9.6)
    d.text(500,1173,'Our work',12,family='Times New Roman')
    d.save('01_based_on_2400996')


def template_2025():
    d=Diagram(570,font='Times New Roman',width=9)
    gold='#B58E37';border='#A29262';dark='#424242'
    d.gradient(12,10,976,43,['#FFF1E0','#F4E6B0','#EAF7F1'])
    d.text(500,32,'Quality-Aware AI Routing under Service and Carbon Constraints',13,True,color='#7B724E')
    d.box(12,54,976,442,'none',border,dashed=True,lw=1)
    d.text(380,72,'Quality, Resource and Carbon Modeling',11.7,True)
    d.text(870,72,'Robustness Analysis',11.7,True)
    d.arrow([(735,71),(763,71)],gold,lw=3,head=13)
    xs=[17,259,501];fills=['#FFFCF6','#FFF5CB','#FFECC2']
    headers=['Quality-Aware Model Routing','Cost and Service Evaluation','Carbon-Aware Execution']
    for x,f,t in zip(xs,fills,headers):
        d.box(x,92,237,354,f,'#8E8D7D',lw=1)
        d.text(x+118.5,110,t,9.6,True,color=gold)
        d.line([(x+7,127),(x+230,127)],border,dashed=True,lw=.9)
    for x in [252,494]:d.arrow([(x-11,96),(x+12,96)],gold,lw=2,head=10)

    def modelcol(x,first,bullets1,second,bullets2):
        d.node(x+8,140,221,36,first,'white',size=9.9,bold=True,edge='#666666',rounded=11,color=gold)
        d.arrow([(x+63,193),(x+63,178)],gold,lw=.9)
        d.arrow([(x+174,193),(x+174,178)],gold,lw=.9)
        d.text(x+13,205,bullets1,8.9,ha='left',va='top')
        d.node(x+8,297,221,36,second,'white',size=9.8,bold=True,edge='#666666',rounded=11,color=gold)
        d.arrow([(x+63,349),(x+63,335)],gold,lw=.9)
        d.arrow([(x+174,349),(x+174,335)],gold,lw=.9)
        d.text(x+13,358,bullets2,8.9,ha='left',va='top')
    modelcol(17,'Quality Prediction / Assessment',
        '- RouterBench: LightGBM, IRT / MIRT\n- V / VC2 post-response assessment\n- Prompt-group splits; OOF fitting',
        'Protected Adaptive Router',
        '- Quality gain versus calling cost\n- CP / Bernstein risk guard; L fallback\n- Direct routing and cascades compared')
    modelcol(259,'Cost and Hardware Accounting',
        '- Historical API call ledger\n- Self-hosted depreciation model\n- ML.ENERGY; assumed power / PUE',
        'Latency and Service Evaluation',
        '- Azure arrivals and token proxies\n- Queueing and call-path overhead\n- P95 latency; all-input violations')
    modelcol(501,'Spatial MILP + DES',
        '- Frozen S/L model assignments\n- Regional slots and completion limits\n- Causal carbon observations',
        'Scheduling Policy Comparison',
        '- Greedy versus Spatial MILP\n- Full MPC and temporal alternatives\n- Total and incremental carbon ledgers')

    d.box(747,92,235,164,'#FEFBF4','#8E8D7D')
    d.text(864.5,110,'Stress Scenarios',10.1,True,color=gold)
    d.line([(753,127),(976,127)],border,dashed=True,lw=.9)
    for y,s in [(140,'Higher Math / Code task share'),(177,'Quality-prediction perturbations'),(214,'Demand growth and service pressure')]:
        d.node(754,y,221,31,s,'white',size=8.7,bold=False,edge='#666666',rounded=9)
    d.arrow([(864,256),(864,274)],gold,lw=1)
    d.box(747,277,235,169,'#FEFBF4','#8E8D7D')
    d.text(864.5,294,'Fairness and Sustainability',10.1,True,color=gold)
    d.line([(753,311),(976,311)],border,dashed=True,lw=.9)
    for y,h,s in [(324,31,'Task-level quality and service risk'),(363,31,'Rebound and total facility emissions'),(402,33,'Measurement and deployment evidence')]:
        d.node(754,y,221,h,s,'white',size=8.6,bold=False,edge='#666666',rounded=9)

    d.gradient(17,449,724,40,['#FDF7DC','#F6DDA7','#EAD08F'])
    d.text(166,469,'Matched Policies + Common DES',10.1,True,color='#7F734F')
    d.arrow([(316,469),(353,469)],gold,lw=3,head=15)
    d.text(542,469,'Quality-Latency-Carbon Trade-offs',10.1,True,color='#7F734F')
    d.arrow([(717,469),(751,469)],gold,lw=3,head=15)
    d.text(864,462,'Applicability, Overhead Limits\nand Adoption Conditions',9.2,True,color='#7F734F')
    for x in [785,861,938]:d.arrow([(x,435),(x,447)],gold,lw=.9)
    d.text(500,513,'Application and Recommendations for SWITCH',11.5,True)
    d.gradient(12,531,976,33,['#EAF7F1','#ECF0CC','#DFDDFC'])
    d.text(255,548,'When Additional Routing Complexity Is Worthwhile',10.2,True,color='#75785B')
    d.text(757,548,'Evidence-Based Recommendation to SWITCH',10.2,True,color='#77729A')
    d.save('02_based_on_2025_O_award')


def template_2300348():
    d=Diagram(490,font='Times New Roman',width=9)
    green='#A9CD95';pink='#F3BABA';violet='#7425A4';blue='#C2DBED'
    def panel(x,y,w,h,fill,title):
        d.box(x,y,w,h,fill,'#166B19',dashed=True,lw=1.2)
        label=d.text(x,y-9,title,9.8,ha='left')
        label.set_bbox(dict(facecolor='white',edgecolor='none',pad=.2))
    panel(15,34,218,423,'#FFF9E4','Quality-Aware Model Router')
    panel(255,34,353,191,'#FBEDED','Performance Metrics and Resource Models')
    panel(632,34,353,191,'#F3ECF9','Robustness and Sustainability Analysis')
    panel(255,293,353,164,'#EEF5FA','Carbon-Aware Regional Scheduling and DES')
    panel(632,293,353,164,'#EDF7E8','Service Fairness and Deployment Assessment')

    # Same tall left-hand chain, with side inputs around the central predictor.
    d.node(69,55,110,30,'Historical inputs','#CFB452',size=10.5,rounded=0)
    d.arrow([(124,85),(124,112)])
    d.node(60,112,130,72,'Pre-LightGBM\nquality predictor',green,size=11,rounded=0)
    d.text(36,146,'prompt\nfeatures',8.2)
    d.text(211,146,'task\nlabel',8.2)
    d.arrow([(38,161),(38,175),(60,175)],head=6)
    d.arrow([(210,161),(210,175),(190,175)],head=6)
    d.arrow([(124,184),(124,211)])
    d.node(67,211,114,35,'Quality estimates','#CFB452',size=9.6,rounded=0)
    d.arrow([(124,246),(124,277)])
    d.node(47,277,154,50,'Cost threshold\nCP + Bernstein guard','#CFB452',size=9.8,rounded=0)
    d.arrow([(124,327),(124,363)])
    d.node(45,363,158,35,'Task-specific L fallback',green,size=9.4,rounded=0)
    d.arrow([(124,398),(124,422)])
    d.node(53,422,142,27,'Frozen S / L choice','#DD8550',size=10.2,color='white',rounded=0)

    # Four analysis branches reproduce the reference's pink upper middle block.
    d.text(431,50,r'Quality, cost, latency and emissions: $Q, C, T, E$',8.9,italic=True)
    d.line([(267,64),(596,64)],'#ECA7A7',dashed=True,lw=.8)
    d.text(343,78,'Quality / calling cost',9.4)
    d.text(520,78,'Latency / emissions',9.1)
    d.line([(431,69),(431,213)],'#ECA7A7',dashed=True,lw=.7)
    specs=[(268,'Pre / Post',r'$\hat q$','Score / risk'),(352,'API / server',r'$\hat C$','Cost ledger'),
           (446,'Queue model',r'$\hat T$','P95 / violations'),(530,'Power + CI',r'$\hat E$','Facility CO2e')]
    for x,head,var,out in specs:
        d.text(x+33,101,head,8.3)
        d.node(x,113,67,29,var,pink,size=13,bold=False,rounded=0)
        d.arrow([(x+33,142),(x+33,159)])
        d.node(x-1,159,70,35,out,pink,size=8.2,bold=False,rounded=0)
        d.arrow([(x+33,194),(x+33,252)],head=7)
        d.text(x+33,265,'metric',8,color='#4E8452',italic=True)
        d.arrow([(x+33,276),(x+33,310)],head=7)

    # The purple side-by-side relationships retain the reference's two-row pattern.
    d.node(646,78,96,48,'Task mix and\nquality noise','#DCC8ED',size=9.5,bold=False,rounded=0)
    d.node(771,70,106,30,'Service risk',violet,size=9.3,color='white',rounded=0)
    d.node(771,113,106,30,'Task-level harms',violet,size=8.7,color='white',rounded=0)
    d.node(904,78,65,58,'Frozen\nrisk\nguard','#DCC8ED',size=9.3,bold=False,rounded=0)
    d.arrow([(742,101),(771,101)],head=7)
    d.arrow([(904,101),(877,101)],head=7)
    d.node(646,169,116,31,'Demand growth','#DCC8ED',size=9.8,bold=False,rounded=0)
    d.arrow([(762,184),(796,184)],head=7)
    d.node(796,164,172,42,'Rebound / total carbon',violet,size=9.5,color='white',rounded=0)

    # Blue lower-middle map: inputs, optimization model, regional decisions and output.
    d.node(267,310,330,25,'Frozen models + observed resource and carbon state',blue,size=9.2,bold=False,rounded=0)
    d.node(267,347,75,77,'Regional CI\nSlots\nDeadlines',blue,size=8.9,bold=False,rounded=0)
    d.node(399,348,130,29,'DE / FR / ES','#DD8550',size=10.4,color='white',rounded=0)
    d.node(399,377,130,27,'Region assignment',green,size=9.7,bold=False,rounded=0)
    d.node(267,424,139,28,'Spatial MILP','#267CB9',size=11.8,color='white',rounded=0)
    d.node(483,424,112,28,'DES outcomes',blue,size=9.8,bold=False,rounded=0)
    d.arrow([(371,335),(371,423)],lw=2.2,head=12)
    d.line([(342,384),(353,384),(371,406)])
    d.arrow([(406,438),(483,438)],head=7)
    d.arrow([(539,424),(539,394)],head=7)
    d.node(535,355,64,38,'Policy\ncomparison',blue,size=8.6,bold=False,rounded=0)
    d.arrow([(233,435),(255,435)],head=7)

    # The reference's green classification panel becomes the task-level evidence audit.
    d.text(807,312,r'Task / service-group assessment: $\Delta Q_g$, $V_g$, $T_{95,g}$',9.2,italic=True)
    d.node(648,336,94,36,'Group by task\nand service class',green,size=8.7,bold=False,rounded=0)
    d.arrow([(695,372),(695,409)],head=7)
    d.node(647,409,98,34,'Quality + delay\ncomparison',green,size=8.8,bold=False,rounded=0)
    d.node(788,409,83,34,'Evidence\naudit',green,size=9.2,bold=False,rounded=0)
    d.arrow([(745,426),(788,426)],head=7)
    d.arrow([(829,409),(829,386)],head=7)
    for j,col in enumerate(['#72A35D','#82B26E','#95BF81','#AED29D']):
        yy=382-j*10
        d.ax.add_patch(Polygon([(792,yy),(850,yy),(867,yy-12),(809,yy-12)],
            facecolor=col,edgecolor='#526C49',lw=.5,zorder=3+j))
    d.text(829,346,'Q / T / C',7.2)
    d.arrow([(870,354),(900,354)],head=7)
    d.node(900,337,69,36,'SWITCH\nstrategy',green,size=10,bold=False,rounded=0)
    d.arrow([(599,374),(632,374)],head=7)
    d.text(500,478,'Our work',12)
    d.save('03_based_on_2300348')


def previews_and_notes():
    # Source crops are only comparison aids. Standalone deliverables contain our own text.
    src=ROOT.parents[2]/'tmp'/'our_work_sources'
    refs=[('2024-05.png',(130,130,955,1090),'01_based_on_2400996','2400996'),
          ('2025work-05.png',(185,480,1440,1137),'02_based_on_2025_O_award','2025 O award / 2502617'),
          ('2023work-05.png',(172,1020,1165,1500),'03_based_on_2300348','2300348')]
    font=ImageFont.truetype('C:/Windows/Fonts/arialbd.ttf',26)
    for file,box,name,label in refs:
        left=Image.open(src/file).convert('RGB').crop(box)
        right=Image.open(ROOT/f'{name}.png').convert('RGB')
        lh=int(1040*left.height/left.width);rh=int(1040*right.height/right.width)
        out=Image.new('RGB',(2160,max(lh,rh)+95),'#EDF0F3')
        draw=ImageDraw.Draw(out)
        draw.text((25,20),f'Reference: {label}',font=font,fill='#263947')
        draw.text((1105,20),'Your project: matched layout',font=font,fill='#263947')
        out.paste(left.resize((1040,lh),Image.Resampling.LANCZOS),(20,75))
        out.paste(right.resize((1040,rh),Image.Resampling.LANCZOS),(1100,75))
        out.save(ROOT/f'{name}_comparison.png')
    notes='''# 三张样图逐一对应的重绘版

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
'''
    (ROOT/'README_ZH.md').write_text(notes,encoding='utf-8')

if __name__=='__main__':
    template_2400996();template_2025();template_2300348();previews_and_notes()
    print('Three reference-matched diagrams and individual comparisons generated.')
