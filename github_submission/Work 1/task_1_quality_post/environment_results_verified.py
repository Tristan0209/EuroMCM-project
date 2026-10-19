import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'task_1_quality_post' else HERE
ENV_OUT = ROOT / 'task1_response time, and environ mental impact' / 'outputs'
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--summary', type=Path, default=ENV_OUT / 'simulation_summary.csv')
parser.add_argument('--manifest', type=Path, default=ENV_OUT / 'run_manifest.json')
parser.add_argument('--out', type=Path, default=HERE / 'outputs')
a = parser.parse_args()
a.out.mkdir(parents=True, exist_ok=True)

checks = {}
def check(label, cond, reason):
    checks[label] = {'pass': bool(cond), 'detail': str(reason)}
    if not cond:
        raise ValueError(f'VERIFICATION FAILED [{label}]: {reason}')

check('files_exist', a.summary.is_file() and a.manifest.is_file(),
      f'{a.summary}; {a.manifest}')
manifest = json.loads(a.manifest.read_text(encoding='utf-8'))

def walk_dict(obj, path=''):
    if isinstance(obj, dict):
        yield path, obj
        for k, v in obj.items():
            yield from walk_dict(v, f'{path}.{k}' if path else k)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_dict(v, f'{path}[{i}]')


profs=[]
for name, node in walk_dict(manifest):
    if isinstance(node, dict) and all(k in node for k in ('shared_w','gpu_idle_w','gpu_busy_w','pue')):
        gpu_count = node.get('gpu_count_per_server',node.get('gpus_per_server',node.get('server_gpu_count',8)))
        profs.append((name,node,int(gpu_count)))
check('manifest_power_profile_found', len(profs)>0, 'no complete shared/idle/busy/PUE block')
for name,node,n_gpu in profs:
    expected={'shared_w':1500.0,'gpu_idle_w':175.0,'gpu_busy_w':760.0,'pue':1.2}
    check('profile_at_'+(name or 'root'),n_gpu == 8 and all(np.isclose(float(node[k]),v,rtol=0,atol=1e-8) for k,v in expected.items()),
          f'actual={{k:node[k] for k in expected}}, GPUs={n_gpu}')


time_gap=abs(a.summary.stat().st_mtime-a.manifest.stat().st_mtime)
check('same_run_write_window',time_gap<=3,f'time difference = {time_gap:.3f} s (max 3 s)')
sha256=hashlib.sha256(a.summary.read_bytes()).hexdigest()

hash_candidates=[]
for path, item in walk_dict(manifest):
    if isinstance(item,dict):
        for k,v in item.items():
            if isinstance(v,str) and len(v)==64 and all(c in '0123456789abcdefABCDEF' for c in v):
                p=(path+'.'+k).lower()
                if 'simulation_summary' in p or ('summary' in p and 'sha' in p):
                    hash_candidates.append(v.lower())
if hash_candidates:
    check('summary_sha256', sha256 in hash_candidates, f'CSV SHA256={sha256}')

raw=pd.read_csv(a.summary)
req={'workload','rate_multiplier','replicas','carbon_scenario','carbon_unit','carbon_boundary',
     'repetitions','p95_latency_s_mean','late_fraction_mean','facility_kwh_mean',
     'emissions_g_mean','emissions_g_per_1000_requests_mean'}
check('summary_columns', req.issubset(raw.columns),f'missing: {sorted(req-set(raw.columns))}')
df=raw.loc[np.isclose(pd.to_numeric(raw.rate_multiplier),1.0)].copy()
check('baseline_load_present',len(df)>0, f'baseline rows={len(df)}')
check('five_repetitions',(pd.to_numeric(df.repetitions,errors='coerce')==5).all(),
      'Every selected row must aggregate 5 repeats; this is not raw 5-row data')
for col in ['p95_latency_s_mean','late_fraction_mean','facility_kwh_mean','emissions_g_mean','emissions_g_per_1000_requests_mean']:
    df[col]=pd.to_numeric(df[col],errors='coerce')
check('metrics_finite',np.isfinite(df[['p95_latency_s_mean','late_fraction_mean','facility_kwh_mean','emissions_g_mean','emissions_g_per_1000_requests_mean']].to_numpy(float)).all(), 'nonfinite metrics')
check('valid_late_and_energy',df.late_fraction_mean.between(0,1).all() and (df.facility_kwh_mean>0).all(), 'late in [0,1], energy > 0')
scenes=['EEA_EU27_2024','Ember_France_2024','Ember_Germany_2024','Ember_Poland_2024']
sub=df[df.carbon_scenario.isin(scenes)].copy()
check('co2e_units',(sub.carbon_unit=='gCO2e/kWh').all(), 'no CO2-only NESO mixed in')
check('no_duplicates',not sub.duplicated(['workload','replicas','carbon_scenario']).any(), 'each group/scenario unique')
eea=sub[sub.carbon_scenario=='EEA_EU27_2024'].copy()
check('EEA_2workloads_3replicas',len(eea)==6 and set(eea.replicas.astype(int))=={1,2,4} and eea.workload.nunique()==2,'six baseline points expected')
ratio=eea.emissions_g_mean/eea.facility_kwh_mean
check('EEA_183_44',np.allclose(ratio.to_numpy(float),183.44,rtol=5e-5,atol=0.01),f'actual factors: {ratio.round(5).tolist()}')

groups=sub.groupby(['workload','replicas'])
check('same_energy_across_carbon_scenarios',all(np.allclose(g.facility_kwh_mean,g.facility_kwh_mean.iloc[0],rtol=5e-5,atol=1e-4) for _,g in groups), 'same physical scenario must have same kWh')
for scenario in scenes:
    check('complete_'+scenario, len(sub[sub.carbon_scenario==scenario])==6,scenario)
    r=sub[sub.carbon_scenario==scenario]
    ratios=r.emissions_g_mean/r.facility_kwh_mean
    check('consistent_factor_'+scenario, (ratios.max()-ratios.min())<0.03, f'inferred intensity approx {float(ratios.mean()):.3f} gCO2e/kWh')

left=eea.copy()
left['sla_pass']=(left.p95_latency_s_mean<=10)&(left.late_fraction_mean<=0.05)
name_map={'azure_code':'Code generation','azure_conversation':'Conversation'}
left['plot_workload']=left.workload.map(lambda s:name_map.get(s,str(s).replace('_',' ').title()))
conv=sub[sub.workload.astype(str).str.contains('conversation',case=False)&(sub.replicas==2)]
rows=[]
for scene,name in [('Ember_France_2024','France'),('Ember_Germany_2024','Germany'),('Ember_Poland_2024','Poland')]:
    z=conv[conv.carbon_scenario==scene]
    check('country_'+name,len(z)==1,f'{scene} for conversation, replicas=2')
    rows.append({'region':name,'carbon_scenario':scene,'gco2e_per_1000_requests':float(z.emissions_g_per_1000_requests_mean.iloc[0])})
regions=pd.DataFrame(rows)
left[['workload','replicas','p95_latency_s_mean','late_fraction_mean','emissions_g_per_1000_requests_mean','sla_pass','repetitions']].to_csv(a.out/'environment_latency_emissions.csv',index=False)
regions.to_csv(a.out/'environment_region_emissions.csv',index=False)

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(10.8,4.15),layout='constrained')
for idx,(label,g) in enumerate(left.groupby('plot_workload',sort=True)):
    g=g.sort_values('replicas'); color=plt.rcParams['axes.prop_cycle'].by_key()['color'][idx]
    ax[0].plot(g.p95_latency_s_mean,g.emissions_g_per_1000_requests_mean,lw=1.6,color=color,label=label)
    for _,r in g.iterrows():
        ax[0].scatter(r.p95_latency_s_mean,r.emissions_g_per_1000_requests_mean,s=62,edgecolor=color,
                      facecolor=color if r.sla_pass else 'white',linewidth=1.7,zorder=3)
        ax[0].annotate(str(int(r.replicas)),(r.p95_latency_s_mean,r.emissions_g_per_1000_requests_mean),
                       xytext=(5,5),textcoords='offset points',fontsize=8)
ax[0].axvline(10,ls='--',lw=1,color='gray')
ax[0].set(title='(a) Latency and emissions | baseline load',xlabel='P95 response time (s)',ylabel=r'Emissions (gCO$_2$e / 1000 requests)')
ax[0].legend(fontsize=8)
bars=ax[1].bar(regions.region,regions.gco2e_per_1000_requests)
for bar,val in zip(bars,regions.gco2e_per_1000_requests):
    ax[1].annotate(f'{val:.1f}',(bar.get_x()+bar.get_width()/2,bar.get_height()),xytext=(0,4),
                   textcoords='offset points',ha='center',fontsize=8)
ax[1].margins(y=.16)
ax[1].set(title='(b) Regional emissions | conversation, 2 replicas',ylabel=r'Emissions (gCO$_2$e / 1000 requests)')
fig.savefig(a.out/'fig_environment_results_en.pdf',bbox_inches='tight')
fig.savefig(a.out/'fig_environment_results_en.png',dpi=300,bbox_inches='tight')
plt.close(fig)
report={'source_csv':str(a.summary),'source_manifest':str(a.manifest),'summary_sha256':sha256,
        'summary_to_manifest_mtime_gap_s':time_gap,'manifest_hash_verified':bool(hash_candidates),
        'checks':checks,'important_limitations':['Manifest and file timestamps corroborate provenance but cannot prove that the simulator executed the formula correctly.',
          'No direct hardware power measurements are verified. These are conditional proxy simulations.',
          'Ember and EEA carbon factors inferred from the provided CSV; upstream source provenance remains subject to the original data audit.']}
(a.out/'environment_figure_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('ALL INPUT CONSISTENCY CHECKS PASSED')
print('PDF:',a.out/'fig_environment_results_en.pdf')
print('PNG:',a.out/'fig_environment_results_en.png')
print('Validation:',a.out/'environment_figure_validation.json')
