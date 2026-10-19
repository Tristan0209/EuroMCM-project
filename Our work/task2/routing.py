import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

NEURAL = Path(__file__).resolve().parent / 'support'
sys.path.insert(0, str(NEURAL))
from comparison_models import DiscreteGainLogistic, LinearArms

SEED = 20261016
INTERNAL_MEAN_LOSS = .01
INTERNAL_PASS_LOSS = .01
FINAL_MEAN_LOSS = .02
FINAL_PASS_LOSS = .02
FINAL_TASK_MEAN_LOSS = .05
FINAL_TASK_PASS_LOSS = .02


def fit_cost(frame):
    design = np.column_stack([np.ones(len(frame)), frame.prompt_chars.to_numpy()/1000])
    train = frame.phase.eq('train').to_numpy()
    coefficients = {}
    for action in ['s', 'l']:
        coef, _ = nnls(design[train], frame['cost_' + action].to_numpy()[train])
        coefficients[action] = coef.tolist()
        frame['cost_hat_' + action] = design @ coef
    c0 = float(frame.loc[train, 'cost_hat_l'].mean())
    frame['cost_delta'] = (frame.cost_hat_l - frame.cost_hat_s) / c0
    return frame, {'coefficients': coefficients, 'normalization_c0': c0,
                   'features': ['intercept', 'prompt_chars/1000'], 'fit_on': 'train_only'}


def execute(frame, score, params):

    score = np.atleast_2d(score)
    initial_l = score - params['lambda'] * frame.cost_delta.to_numpy() >= params['tau']
    gamma = params.get('gamma')
    if gamma is None:
        up = np.zeros_like(initial_l)
    else:
        mc = frame.task_kind.eq('multiple_choice').to_numpy()
        v = frame.v_s.to_numpy()
        required=(~initial_l)&mc[None,:]
        if np.any(required & ~np.isfinite(v)[None,:]):
            raise ValueError('Executed small-model MC check requires V; missing is not zero')
        trigger = (mc & (v < gamma)) | (~mc & frame.hard_s.to_numpy())
        up = (~initial_l) & trigger
    final_l = initial_l | up
    if params.get('chooser') == 'VC2':
        mc = frame.task_kind.eq('multiple_choice').to_numpy()
        available=np.isfinite(frame.vc2_s.to_numpy())&np.isfinite(frame.vc2_l.to_numpy())
        if np.any(up & mc[None,:] & ~available[None,:]):
            raise ValueError('VC2 requires two paid generated responses')

        prefer_l = frame.vc2_l.to_numpy() >= frame.vc2_s.to_numpy()
        final_l = np.where(up & mc, prefer_l, final_l)
    calls_s = (~initial_l).astype(int)
    calls_l = (initial_l | up).astype(int)
    q = np.where(final_l, frame.q_l.to_numpy(), frame.q_s.to_numpy())
    cost = calls_s * frame.cost_s.to_numpy() + calls_l * frame.cost_l.to_numpy()
    return {'initial_l': initial_l, 'upgraded': up, 'final_l': final_l,
            'calls_s': calls_s, 'calls_l': calls_l, 'q': q, 'cost': cost}


def metrics(frame, values):
    q, cost = values['q'], values['cost']
    mean_loss = frame.q_l.mean() - q.mean(axis=1)
    pass_loss = frame.q_l.ge(.5).mean() - (q >= .5).mean(axis=1)
    means, passes = [], []
    for task in sorted(frame.task_type.unique()):
        mask = frame.task_type.eq(task).to_numpy()
        means.append(frame.loc[mask, 'q_l'].mean() - q[:, mask].mean(axis=1))
        passes.append(frame.loc[mask, 'q_l'].ge(.5).mean() - (q[:, mask] >= .5).mean(axis=1))
    max_mean, max_pass = np.max(means, axis=0), np.max(passes, axis=0)
    ok = (mean_loss <= FINAL_MEAN_LOSS+1e-12) & (pass_loss <= FINAL_PASS_LOSS+1e-12)
    ok &= (max_mean <= FINAL_TASK_MEAN_LOSS+1e-12) & (max_pass <= FINAL_TASK_PASS_LOSS+1e-12)
    return {'requests': len(frame), 'repetitions': len(q), 'quality_mean': float(q.mean()),
        'quality_sd': float(q.mean(axis=1).std(ddof=0)), 'pass_rate': float((q >= .5).mean()),
        'pass_rate_sd': float((q >= .5).mean(axis=1).std(ddof=0)),
        'quality_loss': float(mean_loss.mean()), 'quality_loss_worst': float(mean_loss.max()),
        'pass_loss': float(pass_loss.mean()), 'pass_loss_worst': float(pass_loss.max()),
        'max_task_mean_loss_worst': float(max_mean.max()), 'max_task_pass_loss_worst': float(max_pass.max()),
        'model_cost_usd_mean': float(cost.mean()), 'model_cost_usd_total': float(cost.mean(axis=0).sum()),
        'cost_saving': float(1-cost.mean()/frame.cost_l.mean()),
        'cost_saving_sd': float((1-cost.mean(axis=1)/frame.cost_l.mean()).std(ddof=0)),
        'large_call_rate': float(values['calls_l'].mean()), 'upgrade_rate': float(values['upgraded'].mean()),
        'calls_mean': float((values['calls_s']+values['calls_l']).mean()),
        'final_large_rate': float(values['final_l'].mean()), 'final_constraints_all_reps_ok': bool(ok.all()),
        'failed_reps': int((~ok).sum())}


def slice_prediction(prediction, mask):
    return {**prediction, 'mu': prediction['mu'][mask], 'std': prediction['std'][mask],
            'scores': prediction['scores'][:, mask]}


def tune_task(frame, prediction, no_post=False, fixed_cascade=False):
    records = []
    for k in prediction['k_grid']:
        base = prediction['scores'] + k * prediction['std'][None, :]
        for penalty in ([0.] if fixed_cascade else [0., .5, 1.]):
            h = base - penalty * frame.cost_delta.to_numpy()
            thresholds = [float(np.nextafter(h.max(), np.inf))] if fixed_cascade else np.unique(np.r_[
                np.nextafter(h.min(), -np.inf), np.quantile(h, np.linspace(0,1,41)), np.nextafter(h.max(), np.inf)])
            for tau in thresholds:
                for gamma in ([None] if no_post else [None, .25, .5, .75, 1.01]):
                    for chooser in (['L'] if gamma is None else ['L', 'VC2']):
                        params = {'k': float(k), 'lambda': float(penalty), 'tau': float(tau),
                                  'gamma': gamma, 'chooser': chooser}
                        values = execute(frame, base, params)
                        measured = metrics(frame, values)
                        feasible = measured['quality_loss_worst'] <= INTERNAL_MEAN_LOSS+1e-12 and measured['pass_loss_worst'] <= INTERNAL_PASS_LOSS+1e-12
                        records.append({'candidate': len(records), **params, **measured, 'feasible': bool(feasible)})
    grid = pd.DataFrame(records)
    feasible = grid[grid.feasible]
    if len(feasible):
        win = feasible.sort_values(['model_cost_usd_mean','large_call_rate','quality_mean','candidate'],
                                   ascending=[True,True,False,True]).iloc[0]
    else:
        if not fixed_cascade: raise AssertionError('All-L endpoint must be feasible')
        win = grid.sort_values(['quality_mean', 'model_cost_usd_mean'], ascending=[False,True]).iloc[0]
    params = {key: (None if key == 'gamma' and pd.isna(win[key]) else win[key]) for key in ['k','lambda','tau','gamma','chooser']}
    return {'parameters': params, 'candidate': int(win.candidate), 'feasible': bool(win.feasible),
            'select_metrics': win.to_dict()}, grid


def tune(frame, prediction, no_post=False, fixed_cascade=False):
    selected, grids = {}, []
    for task, sub in frame.groupby('task_type', sort=True):
        mask = frame.task_type.eq(task).to_numpy()
        chosen, grid = tune_task(sub.reset_index(drop=True), slice_prediction(prediction, mask), no_post, fixed_cascade)
        selected[task] = chosen
        grids.append(grid.assign(task_type=task))
    return selected, pd.concat(grids, ignore_index=True)


def apply(frame, prediction, selected, fallback_tasks=()):
    count = prediction['scores'].shape[0]
    combined = {name: np.zeros((count, len(frame)), dtype=dtype) for name, dtype in
                [('initial_l',bool),('upgraded',bool),('final_l',bool),('calls_s',int),('calls_l',int),('q',float),('cost',float)]}
    for task, sub in frame.groupby('task_type', sort=True):
        mask = frame.task_type.eq(task).to_numpy()
        pred = slice_prediction(prediction, mask)
        if task in fallback_tasks:
            params = {'k':0., 'lambda':0., 'tau':-np.inf, 'gamma':None, 'chooser':'L'}
        else:
            params = selected[task]['parameters']
        scores = pred['scores'] + params['k'] * pred['std'][None, :]
        values = execute(sub, scores, params)
        for name in combined: combined[name][:, mask] = values[name]
    return combined


def ledger(frame, values, strategy, phase):
    records = []
    for rep in range(len(values['q'])):
        records.append(pd.DataFrame({'strategy': strategy, 'phase': phase, 'rep': rep,
            'request_id': frame.request_id.to_numpy(), 'group': frame.group.to_numpy(),
            'task_type': frame.task_type.to_numpy(),
            **{name: value[rep] for name, value in values.items()}}))
    return pd.concat(records, ignore_index=True)


def baseline(frame, use_large):
    score = np.zeros((1, len(frame)))
    return execute(frame, score, {'lambda':0., 'tau':-1. if use_large else 1., 'gamma':None, 'chooser':'L'})
