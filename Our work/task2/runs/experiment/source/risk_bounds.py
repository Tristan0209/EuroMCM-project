import itertools

import numpy as np
import pandas as pd
from scipy.stats import beta


def binomial_upper(events,n,alpha):
    if not 0 <= events <= n or n<1:raise ValueError('Invalid binomial counts')
    return 1. if events==n else float(beta.ppf(1-alpha,events+1,n-events))


def bernstein_upper(values,alpha):
    values=np.asarray(values,float);n=len(values)
    if n<2:return 1.
    if np.any(values<0) or np.any(values>1):raise ValueError('Bound requires losses in [0,1]')
    log=np.log(2/alpha)
    bound=values.mean()+np.sqrt(2*values.var(ddof=1)*log/n)+7*log/(3*(n-1))
    return float(min(1.,bound))


def task_bounds(frame,values,alpha,duplicate_allowance):
    positive=np.maximum(frame.q_l.to_numpy()[None,:]-values['q'],0).max(axis=0)
    harm=((frame.q_l.to_numpy()[None,:]>=.5)&(values['q']<.5)).any(axis=0)
    rows=[]
    for task in sorted(frame.task_type.unique()):
        mask=frame.task_type.eq(task).to_numpy()
        data=pd.DataFrame({'group':frame.loc[mask,'group'].to_numpy(),'positive':positive[mask],'harm':harm[mask]})
        grouped=data.groupby('group').agg(positive=('positive','max'),harm=('harm','max'))
        n=len(grouped);h=int(grouped.harm.sum());extra=duplicate_allowance[task]
        rows.append({'task_type':task,'independent_groups':n,'harm_groups':h,
            'positive_loss_mean':float(grouped.positive.mean()),'duplicate_mass_allowance':extra,
            'mean_loss_upper':min(1.,bernstein_upper(grouped.positive,alpha)+extra),
            'pass_loss_upper':min(1.,binomial_upper(h,n,alpha)+extra),
            'individual_alpha':alpha})
    return rows


def allowed_fallbacks(bounds,mixtures):
    by_task={b['task_type']:b for b in bounds};tasks=sorted(by_task)
    allowed=[]
    for flags in itertools.product([False,True],repeat=len(tasks)):
        fallback=[t for t,flag in zip(tasks,flags) if flag]
        if any(by_task[t]['mean_loss_upper']>.05 or by_task[t]['pass_loss_upper']>.02 for t in tasks if t not in fallback):continue
        upper_mean=max(sum(w.get(t,0)*by_task[t]['mean_loss_upper'] for t in tasks if t not in fallback) for w in mixtures)
        upper_pass=max(sum(w.get(t,0)*by_task[t]['pass_loss_upper'] for t in tasks if t not in fallback) for w in mixtures)
        if upper_mean<=.02 and upper_pass<=.02:
            allowed.append({'fallback_tasks':fallback,'aggregate_mean_upper':upper_mean,'aggregate_pass_upper':upper_pass})
    assert allowed, 'All-L fallback always has zero relative loss'
    return allowed
