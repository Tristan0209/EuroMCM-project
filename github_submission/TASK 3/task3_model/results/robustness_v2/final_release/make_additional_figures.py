from context import *
import numpy as np
import pandas as pd
from analyze import figure
import matplotlib.pyplot as plt
def main():
    f=pd.read_csv(out('core')/'raw_runs.csv')
    regions=[]
    for case in f.itertuples():
        r=pd.read_csv(Path(case.artifact_directory)/'requests.csv.gz')
        for region in ['DE','FR','ES']:
            mask=r.status.eq('completed')&r.region.eq(region)
            regions.append({'strategy':case.strategy,'seed':case.seed,'region':region,'completed_requests':int(mask.sum()),
                'busy_seconds':float(r.loc[mask,'actual_service_s'].sum()),'completed_queue_wait_sum_s':float(r.loc[mask,'queue_wait_s'].sum())})
    pd.DataFrame(regions).to_csv(out('core')/'regional_execution_by_seed.csv',index=False)
    pd.DataFrame(regions).groupby(['strategy','region']).mean(numeric_only=True).drop(columns='seed').reset_index().to_csv(out('core')/'regional_execution_mean.csv',index=False)
    pivot=f.pivot(index='seed',columns='strategy',values='co2e_g')
    effect=pivot['E-Spatial']-pivot['E-Full']
    fig,ax=plt.subplots(figsize=(9,3.5));ax.bar(np.arange(24),effect);ax.axhline(0,color='grey',lw=1)
    ax.set_xlabel('Paired seed index (24 independent service trajectories)');ax.set_ylabel('E-Spatial minus E-Full (g CO2e)')
    ax.set_title('Positive favors E-Full; branch rescue is not independent temporal value')
    figure(fig,out('core'),'temporal_effect_by_seed')
    old=pd.read_csv(out('core')/'before_solver_status_counts.csv').groupby('strategy').status4.sum()
    new=f.groupby('strategy').status4_count.sum().reindex(old.index)
    recovered=f.groupby('strategy').recovery_recovered_count.sum().reindex(old.index)
    fig,ax=plt.subplots(figsize=(8,3.5));x=np.arange(len(old))
    ax.bar(x-.25,old,width=.25,label='Old status4');ax.bar(x,new,width=.25,label='New status4');ax.bar(x+.25,recovered,width=.25,label='Recovered')
    ax.set_xticks(x);ax.set_xticklabels(old.index);ax.set_ylabel('Subproblem invocations, including unused branches');ax.legend(fontsize=8)
    figure(fig,out('core'),'numerical_rejection_and_recovery_counts')
    h=pd.read_csv(out('H')/'paired_summary.csv');h=h[h.metric.eq('co2e_g')]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax in axes:
        for s,g in h.groupby('proposed'):
            g=g.sort_values('delay_scale');ax.errorbar(g.delay_scale,g['mean'],yerr=[g['mean']-g.ci95_lower,g.ci95_upper-g['mean']],marker='o',capsize=3,label=s)
        ax.set_xlabel('Measured optimization wall multiplier');ax.set_ylabel('Net reduction (g CO2e)\n95% paired t interval')
    axes[0].axhline(0,color='grey',lw=1);axes[0].legend(fontsize=7,loc='center left');axes[0].set_title('Absolute net benefit')
    axes[1].set_ylim(h.ci95_lower.min()-.1,h.ci95_upper.max()+.1);axes[1].set_title('Detail of measured sensitivity')
    figure(fig,out('H'),'controller_net_carbon_benefit')
if __name__=='__main__':main()
