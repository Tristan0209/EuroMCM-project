from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

OUT = Path(__file__).resolve().parent
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
})
COL = {
    'ink':'#25364a', 'blue':'#416F9B', 'bluebg':'#EFF5FA',
    'green':'#237A68','greenbg':'#EFF8F4',
    'amber':'#BC843A','amberbg':'#FCF6EC',
    'red':'#B75C69','redbg':'#FCF1F2'
}
fig, ax = plt.subplots(figsize=(10.8, 6.7))
fig.patch.set_facecolor('white')
ax.set(xlim=(0,12), ylim=(1.45,8.05))
ax.set_aspect('equal', adjustable='box'); ax.axis('off')

def block(x,y,w,h,label,edge,bg,fs=11):
    ax.add_patch(FancyBboxPatch((x,y),w,h,
        boxstyle='round,pad=0.035,rounding_size=0.12',
        linewidth=1.45,edgecolor=edge,facecolor=bg))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',
            fontsize=fs,fontweight='bold',color=COL['ink'])

def arrow(start,end,color=None,dashed=False):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=14,
        lw=1.65,color=color or COL['green'],
        linestyle=(0,(4,3)) if dashed else '-',
        shrinkA=0,shrinkB=0))

def connector(xy1,xy2,color):
    ax.plot([xy1[0],xy2[0]],[xy1[1],xy2[1]],color=color,lw=1.6)


X=1.70; W=5.75; H=.65; MID=X+W/2
nodes=[
    (7.24,'Input request','blue'),
    (6.19,'Pre-inference: LightGBM','blue'),
    (5.14,'Generate candidate responses','amber'),
    (2.86,'Post-inference: TACVA','green'),
    (1.79,'Quality scores for routing','green'),
]
for y,lab,palette in nodes:
    block(X,y,W,H,lab,COL[palette],COL[palette+'bg'],11.2)
arrow((MID,7.24),(MID,6.88))
arrow((MID,6.19),(MID,5.83))

D_Y=4.36; D_W=1.47; D_H=.52
ax.add_patch(Polygon([(MID,D_Y+D_H),(MID+D_W,D_Y),(MID,D_Y-D_H),(MID-D_W,D_Y)],
                     edgecolor=COL['green'],facecolor='white',linewidth=1.6))
ax.text(MID,D_Y,'Response available?',ha='center',va='center',fontsize=10.6,
        fontweight='bold',color=COL['ink'])
arrow((MID,5.14),(MID,D_Y+D_H+.02))
arrow((MID,D_Y-D_H-.02),(MID,3.54))
ax.text(MID+.26,3.65,'Yes',fontsize=9.8,fontweight='bold',color=COL['green'])
arrow((MID,2.86),(MID,2.44))

bx=.13; by=3.99; bw=1.36; bh=.74
block(bx,by,bw,bh,'Use Qpre',COL['red'],COL['redbg'],10.0)
arrow((MID-D_W,D_Y),(bx+bw+.04,D_Y),COL['red'])
ax.text(1.65,D_Y+.16,'No',fontsize=10,fontweight='bold',color=COL['red'])


RX=8.15; RW=3.40
ax.text(RX+RW/2,6.94,'Model Evaluation',ha='center',fontsize=12,
        fontweight='bold',color=COL['ink'])
block(RX,5.90,RW,.72,'Cross-benchmark validation',COL['amber'],COL['amberbg'],10.1)
block(RX,4.74,RW,.72,'MAE  ·  RMSE  ·  AUC',COL['amber'],COL['amberbg'],10.6)
block(RX,3.58,RW,.72,'TACVA ablation: V / VC₂ / VC₁₁',COL['amber'],COL['amberbg'],9.6)
arrow((RX+RW/2,5.90),(RX+RW/2,5.46),COL['amber'])
arrow((RX+RW/2,4.74),(RX+RW/2,4.30),COL['amber'])

ax.plot([X+W+.07,7.82,7.82],[3.18,3.18,3.94],color=COL['amber'],lw=1.55,ls=(0,(4,3)))
arrow((7.82,3.94),(RX-.06,3.94),COL['amber'],dashed=True)

fig.subplots_adjust(left=.025,right=.975,top=.98,bottom=.03)
for extension in ('png','pdf','svg'):
    target=OUT/f'quality_framework_compact.{extension}'
    fig.savefig(target,dpi=600,bbox_inches='tight',pad_inches=.12,
                facecolor='white')
    print('Saved:',target)
plt.close(fig)
