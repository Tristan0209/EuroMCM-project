import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits

from q1_adapter import LGBMRegressor, digest
from routing import DiscreteGainLogistic, LinearArms, NEURAL, SEED
from prepare import save

sys.path.insert(0, str(NEURAL))
from neural_adapter import GainEnsemble
from lightgbm import early_stopping

HERE = Path(__file__).resolve().parent
FAMILIES = ['Pre_direct', 'Ridge', 'Bayesian_Ridge', 'Logistic_Regression', 'CART',
            'LightGBM', 'MLP', 'Random_Forest', 'LinUCB', 'Thompson_Sampling']


def fit_one(name, x, frame):
    train, val = frame.phase.eq('train').to_numpy(), frame.phase.eq('select').to_numpy()
    y = frame.delta_q.to_numpy()
    log, start = [], time.perf_counter()
    model, std, k_grid = None, np.zeros(len(frame)), [0.]
    with threadpool_limits(limits=2):
        if name == 'Ridge':
            best = np.inf
            for alpha in [10., 100.]:
                candidate = Ridge(alpha=alpha).fit(x[train], y[train])
                error = float(((np.clip(candidate.predict(x[val]),-1,1)-y[val])**2).mean())
                log.append({'alpha':alpha, 'select_mse':error})
                if error < best: model, best = candidate, error
        elif name == 'Bayesian_Ridge':
            model = BayesianRidge(max_iter=500, tol=1e-4).fit(x[train], y[train])
        elif name == 'Logistic_Regression':
            model = DiscreteGainLogistic(.01).fit(x[train], y[train])
        elif name == 'CART':
            model = DecisionTreeRegressor(max_depth=4, min_samples_leaf=20, random_state=SEED).fit(x[train], y[train])
        elif name == 'Random_Forest':
            model = RandomForestRegressor(n_estimators=200, max_depth=12, min_samples_leaf=20,
                max_features=.5, n_jobs=4, random_state=SEED).fit(x[train], y[train])
        elif name == 'LightGBM':
            model = LGBMRegressor(n_estimators=400, learning_rate=.03, num_leaves=31,
                min_child_samples=50, reg_lambda=1., random_state=SEED, n_jobs=4, verbosity=-1)
            model.fit(x[train], y[train], eval_set=[(x[val],y[val])], callbacks=[early_stopping(30,verbose=False)])
        elif name == 'MLP':
            config = {'hidden_layers':[64,16], 'ensemble_seeds':[SEED,SEED+1,SEED+2],
                      'l2_alpha':.01, 'batch_size':256, 'learning_rate':.0003,
                      'blas_threads':2, 'max_epochs':80, 'min_validation_improvement':1e-5,
                      'patience':10, 'online_update_epochs':2}
            model = GainEnsemble(config).fit(x[train],y[train],x[val],y[val])
            mu, std = model.predict(x)
            log.extend(model.checkpoints); k_grid = [0., .5, 1.]
        else:
            raise ValueError(name)
        if name != 'MLP': mu = np.clip(model.predict(x),-1,1)
        if name == 'Random_Forest':

            tree_values = np.stack([tree.predict(x) for tree in model.estimators_])
            std = tree_values.std(axis=0); k_grid = [0.,.5,1.]
    return model, {'mu':mu, 'std':std, 'scores':mu[None], 'k_grid':k_grid}, {
        'family':name, 'offline_fit_wall_s':time.perf_counter()-start,
        'search':log, 'select_gain_mse':float(((mu[val]-y[val])**2).mean()),
        'standard_deviation_status':'uncalibrated_ensemble_disagreement' if len(k_grid)>1 else 'not_used'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE / 'runs/experiment')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    assert (out/'prepared.json').exists(), 'Run prepare.py first'
    frame = joblib.load(out/'frame.joblib'); x = np.load(out/'features.npy',allow_pickle=False)
    assert x.shape == (len(frame),395)
    train, val = frame.phase.eq('train').to_numpy(), frame.phase.eq('select').to_numpy()
    signature = {'features_sha256':digest(out/'features.npy'), 'pre_selection_sha256':digest(out/'pre_selection.json'),
                 'fitting_source_sha256':digest(Path(__file__)), 'training_rows':int(train.sum()),
                 'neural_class_source_sha256':digest(NEURAL/'network.py'),
                 'selection_rows':int(val.sum()), 'seed':SEED}
    if (out/'router_fit_signature.json').exists():
        assert json.loads((out/'router_fit_signature.json').read_text(encoding='utf-8')) == signature
    save(out/'router_fit_signature.json',signature)
    models_dir = out/'routers'; models_dir.mkdir(exist_ok=True)
    all_predictions, fit_logs = {}, []
    mu = frame.pre_delta.to_numpy()
    all_predictions['Pre_direct'] = {'mu':mu,'std':np.zeros(len(frame)),'scores':mu[None],'k_grid':[0.]}
    for name in FAMILIES[1:8]:
        path = models_dir / (name+'.joblib')
        if path.exists() and args.resume:
            model, prediction, log = joblib.load(path)
        else:
            model, prediction, log = fit_one(name,x,frame)
            joblib.dump((model,prediction,log),path,compress=3)
        all_predictions[name] = prediction; fit_logs.append(log)
        print('ROUTER FIT',name,'MSE=',log['select_gain_mse'],'seconds=',log['offline_fit_wall_s'],flush=True)
    linear_path=models_dir/'linear_arms.joblib'
    if linear_path.exists() and args.resume:
        linear, log = joblib.load(linear_path)
    else:
        best, linear, trials = np.inf, None, []
        with threadpool_limits(limits=2):
            for ridge in [1.,10.,100.]:
                arm = LinearArms(ridge).fit(x[train],frame[['q_s','q_l']].to_numpy()[train])
                mean,_ = arm.moments(x[val])
                error=float(((np.clip(mean[:,1]-mean[:,0],-1,1)-frame.delta_q.to_numpy()[val])**2).mean())
                trials.append({'ridge':ridge,'select_mse':error})
                if error < best: best,linear=error,arm
        log={'family':'linear_arms','select_gain_mse':best,'search':trials}
        joblib.dump((linear,log),linear_path,compress=3)
    with threadpool_limits(limits=2):
        mean,_=linear.moments(x); mu=np.clip(mean[:,1]-mean[:,0],-1,1)
        all_predictions['LinUCB']={'mu':mu,'std':np.zeros(len(frame)),'scores':linear.score(x,'LinUCB')[None],'k_grid':[0.]}
        all_predictions['Thompson_Sampling']={'mu':mu,'std':np.zeros(len(frame)), 'k_grid':[0.],
            'scores':np.stack([linear.score(x,'Thompson Sampling',np.random.default_rng(SEED+100+r)) for r in range(5)])}
    fit_logs.append(log)
    save(out/'router_fit_log.json',fit_logs)
    joblib.dump(all_predictions,out/'router_predictions.joblib',compress=3)

    records=[]
    for name,pred in all_predictions.items():
        delta=pred['mu'][val]-frame.delta_q.to_numpy()[val]
        records.append({'family':name,'select_gain_rmse':float(np.sqrt((delta**2).mean())),
                        'select_gain_mae':float(np.abs(delta).mean()),
                        'select_gain_spearman':float(spearmanr(pred['mu'][val],frame.delta_q.to_numpy()[val]).correlation)})
    pd.DataFrame(records).to_csv(out/'router_selection_metrics.csv',index=False)
    print('ROUTERS READY',out,flush=True)


if __name__=='__main__':main()
