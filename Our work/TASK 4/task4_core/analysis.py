from pathlib import Path
import numpy as np
import pandas as pd
from .sources import read
from .risk import cvar, mixture_weights, paired_mixture_cvar


def paired_rows(records):
    rows = []
    for record in records:
        if record['status'] != 'COMPLETE_VALIDATED':
            raise ValueError('Incomplete world cannot disappear from risk analysis')
        row = read(Path(record['folder'])/'metrics.json')
        world = read(Path(record['folder']).parents[2]/'worlds'/record['world_id']/'world.json')
        rows.append({**row, 'world_id':record['world_id'],'policy':record['policy'],'base_n':world['parameters']['n'],
                     'seed':world['seed'],'growth':world['parameters']['growth'],'l_hosts':world['parameters']['l_hosts']})
    frame = pd.DataFrame(rows).drop_duplicates(['world_id','policy'])
    pairs = []
    for wid, part in frame.groupby('world_id'):
        indexed = part.set_index('policy')
        if not {'Greedy','Spatial_MILP'} <= set(indexed.index):
            continue
        a,b = indexed.loc['Greedy'],indexed.loc['Spatial_MILP']
        pairs.append({'world_id':wid,'seed':a.seed,'requests':int(a.requests),'growth':a.growth,'l_hosts':int(a.l_hosts),
                      'scheduler_carbon_saving_g':a.co2e_g-b.co2e_g,'scheduler_energy_saving_kwh':a.energy_kwh-b.energy_kwh,
                      'service_violation_difference_P1_minus_P0':b.service_violation_rate-a.service_violation_rate,
                      'P0_total_co2e_g':a.co2e_g,'P1_total_co2e_g':b.co2e_g,
                      'P1_not_completed':int(b.not_completed_count),
                      'P1_effective_quality':b.effective_delivered_quality_per_input})
    return frame,pd.DataFrame(pairs)


def formal_mixture_risk(frame, regimes_by_world, bootstrap=2000, seed=20501018):
    ids = list(regimes_by_world)
    regimes = np.array([regimes_by_world[w] for w in ids])
    if len(ids) != 2000 or sum(regimes==0) != 1000 or sum(regimes==1) != 1000 or bootstrap != 2000:
        raise ValueError('Core v2 requires 1000 independent worlds per regime and 2000 bootstrap draws')
    if frame[['world_id','policy']].duplicated().any():
        raise ValueError('Repeated references are not independent samples')
    a = frame[frame.policy.eq('Greedy')].set_index('world_id').loc[ids]
    b = frame[frame.policy.eq('Spatial_MILP')].set_index('world_id').loc[ids]
    losses = {k:(b[k].to_numpy(),a[k].to_numpy()) for k in
              ['service_violation_rate','model_selection_harm','delivery_incremental_harm','end_to_end_harm','co2e_g']}
    carbon_regret = (b.co2e_g.to_numpy()-a.co2e_g.to_numpy())/100
    losses['carbon_regret_100g'] = (carbon_regret, np.zeros(len(ids)))
    losses['positive_carbon_regret_100g'] = (np.maximum(0,carbon_regret),np.zeros(len(ids)))
    losses['auxiliary_composite'] = (.4*b.service_violation_rate.to_numpy()/.001 +
        .3*b.end_to_end_harm.to_numpy()/.02 + .3*np.maximum(0,carbon_regret),
        .4*a.service_violation_rate.to_numpy()/.001 + .3*a.end_to_end_harm.to_numpy()/.02)
    results = {}
    for p in [0.,.1,.2,.4]:
        result = {}
        for metric,(proposed,baseline) in losses.items():
            result[metric] = paired_mixture_cvar(proposed,baseline,regimes,p,bootstrap=bootstrap,seed=seed)
            tail = cvar(proposed,weights=mixture_weights(regimes,p))
            result[metric]['proposed_tail_weights'] = dict(zip(ids,tail['tail_probability_weights'].tolist()))
            result[metric]['positive_loss_worlds'] = int((proposed > 0).sum())
            result[metric]['cvar_of_paired_difference'] = cvar(proposed-baseline,weights=mixture_weights(regimes,p))['cvar']
        results[str(p)] = result
    return {'status':'CONDITIONAL_SIMULATION_ONLY','pointwise_intervals_not_joint':True,'results':results}


def rebound_rows(frame):

    base = frame[frame.policy.eq('Greedy') & frame.growth.eq(0) & frame.l_hosts.eq(1)]
    if base.seed.duplicated().any():
        raise ValueError('Specify one matched original normal baseline per seed')
    baseline = base.set_index('seed').co2e_g.to_dict()
    base_n = base.set_index('seed').base_n.to_dict()
    rows = []
    for row in frame[frame.policy.eq('Spatial_MILP')].itertuples():
        if row.seed in baseline:
            if row.base_n != base_n[row.seed]:
                raise ValueError('Growth comparison requires the SAME original demand cohort')
            rows.append({'world_id':row.world_id,'growth':row.growth,'l_hosts':row.l_hosts,'seed':row.seed,
                         'fixed_demand_total_carbon_change_g':row.co2e_g-baseline[row.seed],
                         'not_completed_count':row.not_completed_count,'effective_delivered_quality':row.effective_delivered_quality_per_input,
                         'interpretation':'exogenous growth only; efficiency-induced elasticity not identified'})
    return pd.DataFrame(rows)


def lhs_screen(frame):

    from .protocol import LHS_NAMES
    required = set(LHS_NAMES+['scramble','parameter_point','seed','scheduler_carbon_saving_g'])
    if not required <= set(frame):
        raise ValueError('Missing LHS parameters/pair metadata')
    if frame[['scramble','parameter_point','seed']].duplicated().any():
        raise ValueError('Repeated references do not add LHS evidence')
    tables, tops, signs = [],[],[]
    for scramble in [0,1]:
        part = frame[frame.scramble.eq(scramble)]
        groups = part.groupby('parameter_point')
        if len(groups) != 16 or not groups.seed.nunique().eq(3).all():
            raise ValueError('Core screen requires 16 points x3 distinct seeds per scramble')
        means = groups[LHS_NAMES+['scheduler_carbon_saving_g']].mean()
        score = means.corr(method='spearman')['scheduler_carbon_saving_g'].drop('scheduler_carbon_saving_g')
        table = score.rename('spearman_descriptive').reset_index().rename(columns={'index':'parameter'})
        table['scramble'] = scramble
        tables.append(table)
        top = score.abs().sort_values(ascending=False).head(5).index.tolist()
        tops.append(top)
        signs.append(score.to_dict())
    overlap = set(tops[0]) & set(tops[1])
    sign_changes = [p for p in overlap if np.sign(signs[0][p])!=np.sign(signs[1][p])]
    return pd.concat(tables,ignore_index=True),{'top5_overlap':len(overlap),'important_sign_changes':sign_changes,
             'trigger_extension_by_ranking':len(overlap)<3 or bool(sign_changes),
             'boundary_and_unexplained_tail_review_still_required':True,'low_power_interactions_not_excluded':True,
             'PRCC_significance_claim':False}
