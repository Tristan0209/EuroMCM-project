import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

P=argparse.ArgumentParser()
ROOT=Path(__file__).resolve().parent
P.add_argument('--cost',default=str(ROOT/'outputs'/'cost_summary.csv'))
P.add_argument('--sweep',default=str(ROOT/'outputs'/'power_sensitivity.csv'))
P.add_argument('--out',default=str(ROOT/'outputs'))
P.add_argument('--demo-watermark',action='store_true')
P.add_argument('--late-summary',default=None,help='CSV with region, arrival_multiplier, replicas, late_fraction (optional if cost_summary already has it)')
A=P.parse_args()
out=Path(A.out);out.mkdir(parents=True,exist_ok=True)
cost=pd.read_csv(A.cost);sweep=pd.read_csv(A.sweep)
for d,fields in [(cost,['region','arrival_multiplier','replicas','n_requests','cost_total_eur','simulation_duration_s','p95_response_s','cost_electricity_eur','cost_hardware_eur']),(sweep,['region','arrival_multiplier','replicas','shared_w','gpu_idle_w','cost_total_eur','cost_electricity_eur','cost_hardware_eur','feasible'])]:
 missing=set(fields)-set(d.columns)
 if missing:raise ValueError(f'Missing CSV fields: {missing}')
shared=np.arange(750,2501,125,dtype=float);idle=np.sort(np.unique(np.r_[np.arange(100,251,10,dtype=float),175.0]));sweep=sweep[sweep.feasible.astype(str).str.lower().isin(['true','1'])].copy()
if sweep[['shared_w','gpu_idle_w']].drop_duplicates().shape[0]!=len(shared)*len(idle):raise ValueError('Incomplete 15 x 17 power grid')
if len(sweep)!=27*len(shared)*len(idle):raise ValueError('Expected 27 scenario x 255 power rows; found '+str(len(sweep)))
if cost.shape[0]!=27:raise ValueError('Cost scenarios must include 3 regions x 3 loads x 3 replicas')
for f in ('region','arrival_multiplier','replicas'):sweep[f]=sweep[f].astype(cost[f].dtype)
keys=['region','arrival_multiplier','replicas']
sweep=sweep.merge(cost[keys+['n_requests','p95_response_s','simulation_duration_s']],on=keys,validate='many_to_one')
sweep['total_eur_per_1000']=sweep.cost_total_eur*1000/sweep.n_requests
sweep['electricity_eur_per_1000']=sweep.cost_electricity_eur*1000/sweep.n_requests
sweep['scenario_rep']=0
sweep=sweep.groupby(keys+['shared_w','gpu_idle_w'],as_index=False).agg(total_eur_per_1000=('total_eur_per_1000','mean'),electricity_eur_per_1000=('electricity_eur_per_1000','mean'),p95_response_s=('p95_response_s','mean'),n_requests=('n_requests','mean'),simulation_duration_s=('simulation_duration_s','mean'))
if A.late_summary:
 late=pd.read_csv(A.late_summary)
 if 'late_fraction' not in late.columns and 'late_fraction_mean' in late.columns:
  late=late.rename(columns={'late_fraction_mean':'late_fraction'})
 missing=set(keys+['late_fraction'])-set(late.columns)
 if missing:raise ValueError(f'Late summary missing fields: {missing}')
 late=late.groupby(keys,as_index=False)['late_fraction'].mean()
elif 'late_fraction' in cost.columns or 'late_fraction_mean' in cost.columns:
 latecol='late_fraction' if 'late_fraction' in cost.columns else 'late_fraction_mean'
 late=cost.groupby(keys,as_index=False)[latecol].mean().rename(columns={latecol:'late_fraction'})
else:late=None
base=sweep[(sweep.shared_w==1500)&(sweep.gpu_idle_w==175)]
if len(base)!=27:raise ValueError('Missing power baseline')
fr=sweep[sweep.region=='FR'];cost_fr=cost[cost.region=='FR']
sel_a=fr[(fr.arrival_multiplier==1)&(fr.replicas==1)].copy()
ref=float(sel_a[(sel_a.shared_w==1500)&(sel_a.gpu_idle_w==175)].total_eur_per_1000.iloc[0]);sel_a['percent_change']=100*(sel_a.total_eur_per_1000/ref-1)
sel_b=fr[fr.arrival_multiplier==2].pivot_table(index=['shared_w','gpu_idle_w'],columns='replicas',values='total_eur_per_1000').reset_index();sel_b['diff_2_minus_1']=sel_b[2]-sel_b[1]
ranges=fr.groupby(['arrival_multiplier','replicas'],as_index=False).agg(cost_min=('total_eur_per_1000','min'),cost_max=('total_eur_per_1000','max'))
ranges=ranges.merge(base[base.region=='FR'][['arrival_multiplier','replicas','total_eur_per_1000','p95_response_s']],on=['arrival_multiplier','replicas'],validate='one_to_one').rename(columns={'total_eur_per_1000':'baseline_cost'})
if late is not None:ranges=ranges.merge(late[late.region=='FR'][['arrival_multiplier','replicas','late_fraction']],on=['arrival_multiplier','replicas'],how='left')
else:ranges['late_fraction']=np.nan


ranges['sla_status']=np.select(
 [(ranges.p95_response_s>10)|(ranges.late_fraction>.05),
  (ranges.p95_response_s<=10)&ranges.late_fraction.notna()&(ranges.late_fraction<=.05)],
 ['FAIL','PASS'],default='UNKNOWN')
ranges['meets_sla']=ranges.sla_status.eq('PASS')

capex=515000*.9; hourly=capex/(5*8760)
d=cost_fr.copy();d['share_total_eur_per_1000']=d.cost_total_eur*1000/d.n_requests
d['whole_box_eur_per_1000']=(d.cost_total_eur-d.cost_hardware_eur+hourly*d.simulation_duration_s/3600)*1000/d.n_requests
d=d.sort_values(['arrival_multiplier','replicas'])
for frame,name in [(sel_a,'panel_a_heatmap_data.csv'),(sel_b,'panel_b_scaling_difference.csv'),(ranges,'panel_c_cost_latency_ranges.csv'),(d,'panel_d_depreciation_comparison.csv'),(sweep,'full_power_cost_plot_data.csv')]:frame.to_csv(out/name,index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig=plt.figure(figsize=(12.4,9.4),layout='constrained')
gs=fig.add_gridspec(2,2)
ax=np.empty((2,2),dtype=object)
ax[0,0]=fig.add_subplot(gs[0,0])
ax[0,1]=fig.add_subplot(gs[0,1])
ax[1,1]=fig.add_subplot(gs[1,1])


cgs=gs[1,0].subgridspec(2,1,height_ratios=[1,2.7],hspace=0.055)
c_top=fig.add_subplot(cgs[0])
c_bottom=fig.add_subplot(cgs[1],sharex=c_top)
ax[1,0]=c_bottom
xedges=np.r_[shared[0]-62.5,(shared[:-1]+shared[1:])/2,shared[-1]+62.5]
yedges=np.r_[idle[0]-5,(idle[:-1]+idle[1:])/2,idle[-1]+5]
def heat(a,data,val,title,cbar):
 z=data.pivot(index='gpu_idle_w',columns='shared_w',values=val).reindex(index=idle,columns=shared).to_numpy(dtype=float)
 vmax=max(abs(np.nanmin(z)),abs(np.nanmax(z))) if val=='percent_change' else None
 mesh=a.pcolormesh(xedges,yedges,z,cmap='coolwarm',shading='flat',vmin=(-vmax if vmax else None),vmax=(vmax if vmax else None),rasterized=False)
 a.plot(1500,175,marker='o',ms=8.5,markeredgecolor='#22364b',markeredgewidth=1.8,markerfacecolor='white',label='Baseline (1500, 175 W)',zorder=9)
 a.set(xlabel='Shared equipment power (W)',ylabel='Idle power per GPU (W)',title=title)
 a.set_xticks([750,1000,1250,1500,1750,2000,2250,2500]);a.set_yticks([100,125,150,175,200,225,250]);a.tick_params(labelsize=8);a.legend(loc='upper left',fontsize=7,framealpha=.9)
 bar=fig.colorbar(mesh,ax=a,shrink=.8,pad=.012);bar.set_label(cbar)
 return a,z
heat(ax[0,0],sel_a,'percent_change','(a) Total cost sensitivity | FR, 1× load, 1 replica','Change from baseline (%)')
_,zb=heat(ax[0,1],sel_b,'diff_2_minus_1','(b) Incremental cost of scaling | FR, 2× load','Additional cost: 2 − 1 replica (EUR / 1000 req.)')
if np.nanmin(zb)>0:ax[0,1].text(.02,.02,'All differences > 0: scaling always costs more',transform=ax[0,1].transAxes,fontsize=8,va='bottom',ha='left',bbox=dict(facecolor='white',alpha=.88,edgecolor='none'))
if np.nanmin(zb)<0<np.nanmax(zb):
 ax[0,1].contour(shared,idle,zb,levels=[0],colors='black',linewidths=1.25)
colors={.5:'#1f77b4',1.0:'#2ca02c',2.0:'#d95f02'};markers={1:'o',2:'s',4:'D'}


a=c_bottom
values=np.sort(ranges.p95_response_s.to_numpy(float))
if not np.isfinite(values).all():
 raise ValueError('Non-finite P95 values in French configurations')
unique=np.unique(values)
gaps=np.diff(unique)

if len(unique)>=3 and len(gaps) and gaps[-1]>max(30.0,3.0*max(unique[-2],1.0)):
 low_max=max(12.0,unique[-2]*1.20)
 high_min=max(low_max+1,unique[-1]*0.93)
 high_max=unique[-1]*1.055
 if high_min<=low_max: raise ValueError('Invalid adaptive broken-axis limits')
 c_bottom.set_ylim(0,low_max)
 c_top.set_ylim(high_min,high_max)
else:


 c_bottom.set_ylim(0,max(12.0,float(values.max())*1.08))
 c_top.set_visible(False)
 c_top.set_ylim(0,1)

def draw_points(axis):
 for _,r in ranges.iterrows():
  col=colors[float(r.arrival_multiplier)]
  mk=markers[int(r.replicas)]
  axis.plot([r.cost_min,r.cost_max],[r.p95_response_s]*2,color=col,lw=2.0,alpha=.78,zorder=2)
  if r.sla_status=='UNKNOWN':
   axis.scatter(r.baseline_cost,r.p95_response_s,s=58,marker='X',color='#777777',edgecolor='black',linewidth=.4,zorder=4)
  else:
   axis.scatter(r.baseline_cost,r.p95_response_s,s=55,marker=mk,
                facecolor=(col if r.sla_status=='PASS' else 'white'),
                edgecolor=col,linewidth=1.5,zorder=4)
 axis.axhline(10,color='gray',ls='--',lw=1,zorder=1)

draw_points(c_bottom)
if c_top.get_visible():
 draw_points(c_top)
 c_top.spines['bottom'].set_visible(False)
 c_bottom.spines['top'].set_visible(False)
 c_top.tick_params(bottom=False,labelbottom=False)
 c_bottom.tick_params(top=False)

 dl=.014
 cut=dict(color='black',clip_on=False,lw=1.1)
 for axis,y in [(c_top,0),(c_bottom,1)]:
  for x in (0,1):
   axis.plot((x-dl,x+dl),(y-dl,y+dl),transform=axis.transAxes,**cut)
else:
 c_bottom.spines['top'].set_visible(False)
c_top.set_title('(c) Cost–latency trade-off | FR',fontsize=11,pad=5)
if not c_top.get_visible():c_bottom.set_title('(c) Cost–latency trade-off | FR',fontsize=11,pad=5)
c_bottom.set_xlabel('Total cost (EUR / 1000 requests)')

c_bottom.set_ylabel('P95 response time (s)')
c_top.tick_params(labelsize=8)
c_bottom.tick_params(labelsize=8)
legend=[Line2D([0],[0],color=colors[l],lw=2,label=f'Load ×{l:g}') for l in (.5,1.,2.)]
legend += [Line2D([0],[0],ls='None',marker=markers[r],color='black',label=f'{r} replicas') for r in (1,2,4)]
legend += [Line2D([0],[0],ls='None',marker='o',markerfacecolor='black',markeredgecolor='black',label='SLA pass'),
           Line2D([0],[0],ls='None',marker='o',markerfacecolor='white',markeredgecolor='black',label='SLA fail'),
           Line2D([0],[0],ls='None',marker='X',color='#777777',label='SLA unknown')]


(c_top if c_top.get_visible() else c_bottom).legend(
 handles=legend,ncol=3,fontsize=6.1,loc='upper right',framealpha=.93,
 borderpad=.35,labelspacing=.35,columnspacing=.75,handlelength=1.4)
a=ax[1,1]
for k,(_,r) in enumerate(d.iterrows()):
 a.plot([r.share_total_eur_per_1000,r.whole_box_eur_per_1000],[k,k],color='#abb4be',lw=2)
 a.scatter(r.share_total_eur_per_1000,k,color='#1f77b4',s=34,zorder=3)
 a.scatter(r.whole_box_eur_per_1000,k,color='#d97935',marker='D',s=37,zorder=3)
a.set_yticks(range(len(d)),[f'{r.arrival_multiplier:g}× load / {int(r.replicas)} GPU' for _,r in d.iterrows()],fontsize=8)
a.invert_yaxis();a.set(title='(d) Depreciation convention | FR, baseline power',xlabel='Total cost (EUR / 1000 requests)')
a.legend(handles=[Line2D([0],[0],marker='o',ls='None',color='#1f77b4',label='Replica-share depreciation'),Line2D([0],[0],marker='D',ls='None',color='#d97935',label='Whole-box depreciation')],fontsize=7,loc='lower right')
if A.demo_watermark:fig.text(.5,.5,'TEST INPUT (15 REQUESTS) — NOT PAPER RESULTS',ha='center',va='center',fontsize=23,color='crimson',alpha=.18,rotation=24,fontweight='bold')
fig.suptitle('Power assumptions and robustness of cost conclusions',fontsize=14,fontweight='bold')
fig.savefig(out/'fig_3_3_power_cost_robustness_en.pdf',bbox_inches='tight')
fig.savefig(out/'fig_3_3_power_cost_robustness_en.png',dpi=300,bbox_inches='tight')
plt.close(fig)
print('Saved PDF, PNG and five plot CSV files to',out)
