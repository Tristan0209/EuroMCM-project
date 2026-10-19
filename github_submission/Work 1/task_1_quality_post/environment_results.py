import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent if HERE.name == 'task_1_quality_post' else HERE
DEFAULT_SUMMARY = PROJECT / 'task1_response time, and environ mental impact' / 'outputs' / 'simulation_summary.csv'
parser = argparse.ArgumentParser()
parser.add_argument('--summary', type=Path, default=DEFAULT_SUMMARY)
parser.add_argument('--out', type=Path, default=HERE / 'outputs')
parser.add_argument('--confirmed-new-power', action='store_true',
                    help='Only set after checking upstream simulation was rerun with 8 GPU, 1500/175/760 W, PUE 1.2.')
args = parser.parse_args()
if not args.summary.is_file():
    raise FileNotFoundError(f'Missing input CSV: {args.summary}\nUse --summary with the actual path.')
args.out.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(args.summary)
required = {'workload','rate_multiplier','replicas','carbon_scenario','carbon_unit',
            'repetitions','p95_latency_s_mean','late_fraction_mean',
            'emissions_g_per_1000_requests_mean'}
missing = required - set(df)
if missing:
    raise ValueError(f'Missing columns: {sorted(missing)}')
df = df[np.isclose(df['rate_multiplier'].astype(float), 1.0)].copy()
if df.empty: raise ValueError('No baseline load rate_multiplier=1.0 rows')
if not (df.repetitions.astype(int) == 5).all():
    raise ValueError('Not all baseline rows represent five repetitions')
scenarios = ['EEA_EU27_2024','Ember_France_2024','Ember_Germany_2024','Ember_Poland_2024']
sub = df[df.carbon_scenario.isin(scenarios)].copy()
if not (sub.carbon_unit == 'gCO2e/kWh').all():
    raise ValueError('Selected data do not all use gCO2e/kWh')
key = ['workload','replicas','carbon_scenario']
if sub.duplicated(key).any():
    raise ValueError(f'Duplicate aggregated rows for {key}')
eea = sub[sub.carbon_scenario == 'EEA_EU27_2024'].copy()
if eea.empty: raise ValueError('No EEA_EU27_2024 records')
if len(eea.workload.unique()) != 2 or sorted(eea.replicas.unique().tolist()) != [1,2,4]:
    raise ValueError('Need two workloads and exactly 1,2,4 replicas for left panel')
eea['sla_pass'] = (eea.p95_latency_s_mean <= 10) & (eea.late_fraction_mean <= .05)
label_map = {'azure_code':'Code generation', 'azure_conversation':'Conversation'}
eea['plot_workload'] = eea.workload.map(lambda x: label_map.get(x,x.replace('_',' ').title()))
if 'Ember_France_2024' not in sub.carbon_scenario.values: raise ValueError('Ember country scenarios missing')
conversation = sub[sub.workload.astype(str).str.contains('conversation',case=False) & (sub.replicas == 2)]
reg_data = []
for s,name in [('Ember_France_2024','France'),('Ember_Germany_2024','Germany'),('Ember_Poland_2024','Poland')]:
    r = conversation[conversation.carbon_scenario == s]
    if len(r) != 1: raise ValueError(f'Need exactly one conversation/2 GPU row for {s}, found {len(r)}')
    reg_data.append({'region':name,'emissions_g_per_1000':float(r.emissions_g_per_1000_requests_mean.iloc[0]),
                     'carbon_scenario':s})
regions = pd.DataFrame(reg_data)
eea[['workload','replicas','p95_latency_s_mean','late_fraction_mean','emissions_g_per_1000_requests_mean','sla_pass','repetitions']].to_csv(args.out/'environment_latency_emissions.csv',index=False)
regions.to_csv(args.out/'environment_region_emissions.csv',index=False)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42})
fig, ax = plt.subplots(1,2,figsize=(11,4.3),layout='constrained')
for idx,(name,g) in enumerate(eea.groupby('plot_workload',sort=True)):
    g = g.sort_values('replicas')
    color = plt.rcParams['axes.prop_cycle'].by_key()['color'][idx]
    ax[0].plot(g.p95_latency_s_mean,g.emissions_g_per_1000_requests_mean, color=color, lw=1.7,label=name)
    for _,r in g.iterrows():
        ax[0].scatter(r.p95_latency_s_mean,r.emissions_g_per_1000_requests_mean,
                      s=57,marker='o',edgecolor=color,facecolor=color if r.sla_pass else 'white',linewidth=1.7,zorder=3)
        ax[0].annotate(str(int(r.replicas)),(r.p95_latency_s_mean,r.emissions_g_per_1000_requests_mean),
                       xytext=(5,5),textcoords='offset points',fontsize=8)
ax[0].axvline(10,color='gray',ls='--',lw=1)
ax[0].set(title='(a) Latency vs. emissions | baseline load',xlabel='P95 response time (s)',
          ylabel=r'Emissions (gCO$_2$e / 1000 requests)')
ax[0].legend(fontsize=8)
bars = ax[1].bar(regions.region,regions.emissions_g_per_1000)
for bar,v in zip(bars,regions.emissions_g_per_1000):
    ax[1].annotate(f'{v:.1f}',(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),
                   textcoords='offset points',ha='center',va='bottom',fontsize=8)
ax[1].margins(y=.16)
ax[1].set(title='(b) Regional emissions | conversation, 2 replicas',
          ylabel=r'Emissions (gCO$_2$e / 1000 requests)')
if not args.confirmed_new_power:
    fig.text(.5,.5,'POWER MODEL NOT VERIFIED FOR INPUT CSV',ha='center',va='center',
             fontsize=18,alpha=.15,color='firebrick',rotation=16,weight='bold')
fig.savefig(args.out/'fig_environment_results_en.pdf',bbox_inches='tight')
fig.savefig(args.out/'fig_environment_results_en.png',dpi=300,bbox_inches='tight')
plt.close(fig)
print('Saved figure to',args.out)
print('WARNING: Upstream DGX power model not verified.' if not args.confirmed_new_power else 'Power model manually confirmed by user.')
