import copy
import sys

import numpy as np
import pandas as pd



_dask_was_loaded = 'dask' in sys.modules
_dask_before = sys.modules.get('dask')
sys.modules['dask'] = None
try:
    from lightgbm import LGBMRegressor, early_stopping
finally:
    if _dask_was_loaded:
        sys.modules['dask'] = _dask_before
    else:
        sys.modules.pop('dask',None)
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.neural_network import MLPRegressor
from threadpoolctl import threadpool_limits

from data_protocol import HERE
from quality import diagnostics


class GainEnsemble:
    def __init__(self, config):
        self.config = config
        self.models, self.history, self.checkpoints = [], [], []

    def fit(self, x_train, y_train, x_val, y_val):
        c = self.config
        with threadpool_limits(limits=c['blas_threads']):
            for seed in c['ensemble_seeds']:
                net = MLPRegressor(hidden_layer_sizes=tuple(c['hidden_layers']), activation='relu', solver='adam',
                    alpha=c['l2_alpha'], batch_size=c['batch_size'], learning_rate_init=c['learning_rate'],
                    max_iter=1, early_stopping=False, shuffle=False, random_state=seed)
                rng = np.random.default_rng(seed)
                best_loss, stale, best, best_epoch = float('inf'), 0, None, 0
                for epoch in range(1, c['max_epochs']+1):
                    idx = rng.permutation(len(y_train))
                    net.partial_fit(x_train[idx], y_train[idx])
                    predicted = np.clip(net.predict(x_val), -1, 1)
                    loss = float(np.mean((predicted-y_val)**2))
                    self.history.append({'seed': seed, 'epoch': epoch, 'validation_mse': loss, 'training_loss': float(net.loss_)})
                    if loss < best_loss - c['min_validation_improvement']:
                        best_loss, best, best_epoch, stale = loss, copy.deepcopy(net), epoch, 0
                    else:
                        stale += 1
                    if epoch == 1 or epoch % 10 == 0:
                        print('MLP seed=%d epoch=%d val_MSE=%.6f best=%.6f' % (seed,epoch,loss,best_loss), flush=True)
                    if stale >= c['patience']:
                        break
                self.models.append(best)
                self.checkpoints.append({'seed': seed, 'best_epoch': best_epoch, 'epochs_run': epoch, 'best_validation_mse': best_loss})
        return self

    def predict_members(self, x):
        with threadpool_limits(limits=self.config['blas_threads']):
            return np.vstack([np.clip(net.predict(x), -1, 1) for net in self.models])

    def predict(self, x):
        values = self.predict_members(x)
        return values.mean(axis=0), values.std(axis=0, ddof=0)

    def update(self, x, y, seed):

        with threadpool_limits(limits=self.config['blas_threads']):
            rng = np.random.default_rng(seed)
            for net in self.models:
                for _ in range(self.config['online_update_epochs']):
                    idx = rng.permutation(len(y))
                    net.partial_fit(x[idx], y[idx])


def fit_baselines(x, frame, config):
    train, val = frame.split.eq('train').to_numpy(), frame.split.eq('validation').to_numpy()
    y = frame.delta_q.to_numpy()
    scores, best, best_loss = [], None, float('inf')
    with threadpool_limits(limits=config['blas_threads']):
        for alpha in [1.0, 10.0, 100.0]:
            model = Ridge(alpha=alpha).fit(x[train], y[train])
            loss = float(np.mean((np.clip(model.predict(x[val]),-1,1)-y[val])**2))
            scores.append({'model':'Ridge','alpha':alpha,'validation_mse':loss})
            if loss < best_loss:
                best, best_loss = model, loss
    lgbm = LGBMRegressor(n_estimators=400, learning_rate=.03, num_leaves=31, min_child_samples=50,
                        reg_lambda=1., random_state=config['seed'], n_jobs=4, verbosity=-1)
    lgbm.fit(x[train], y[train], eval_set=[(x[val],y[val])], callbacks=[early_stopping(30,verbose=False)])
    scores.append({'model':'LightGBM','alpha':None,'validation_mse':float(np.mean((np.clip(lgbm.predict(x[val]),-1,1)-y[val])**2))})
    return best, lgbm, pd.DataFrame(scores)


V_FEATURES = ['model_is_L','format_error','empty','refusal','hard_flag','repetition']


def v_features(request, response, action):
    d = diagnostics(request, response)
    return [float(action=='L'),float(d['format_error']),float(d['empty']),float(d['refusal']),float(d['hard_flag']),d['repetition']]


class RetrainedV:





    def __init__(self, config):
        self.config = config

    def make_model(self, seed):
        c = self.config['post_v']
        return LGBMRegressor(n_estimators=c['n_estimators'], num_leaves=c['num_leaves'],
            min_child_samples=c['min_child_samples'], reg_lambda=c['reg_lambda'],
            random_state=seed, n_jobs=4, verbosity=-1)

    def fit_predict_oof(self, frame):
        rows = []
        for i, req in enumerate(frame.to_dict('records')):
            if req['task_kind'] != 'multiple_choice':
                continue
            for a, suffix in [('S','s'),('L','l')]:
                rows.append({'position':i,'action_id':a,'prompt_group':req['prompt_group'], 'split':req['split'],
                             'quality':req['q_'+suffix], 'features':v_features(req,req['response_'+suffix],a)})
        x = np.asarray([r['features'] for r in rows],dtype=float)
        y = np.array([r['quality'] for r in rows])
        train = np.array([r['split']=='train' for r in rows])
        groups = np.array([r['prompt_group'] for r in rows])
        positions = np.flatnonzero(train)
        prediction = np.full(len(rows), np.nan)
        fit_ids = np.empty(len(rows),dtype=object)
        self.provenance = {}
        for fold,(fit,hold) in enumerate(GroupKFold(self.config['post_v']['folds']).split(x[train],y[train],groups[train])):
            fit_idx,hold_idx = positions[fit],positions[hold]
            net = self.make_model(self.config['seed']+fold)
            net.fit(x[fit_idx],y[fit_idx])
            prediction[hold_idx] = np.clip(net.predict(x[hold_idx]),0,1)
            name = 'V_fold_%d'%fold
            fit_ids[hold_idx] = name
            self.provenance[name] = {'training_prompt_hashes':sorted(set(groups[fit_idx]))}
        self.model = self.make_model(self.config['seed'])
        self.model.fit(x[train],y[train])
        prediction[~train] = np.clip(self.model.predict(x[~train]),0,1)
        fit_ids[~train] = 'V_full_train'
        self.provenance['V_full_train'] = {'training_prompt_hashes':sorted(set(groups[train]))}
        output = pd.DataFrame([{'request_id':frame.iloc[r['position']].request_id, 'action_id':r['action_id'],
                               'split':r['split'],'v_score':float(prediction[i]),'quality_true':r['quality'],
                               'fit_id':fit_ids[i]} for i,r in enumerate(rows)])
        scores = {a:np.full(len(frame),np.nan) for a in ['S','L']}
        for i,r in enumerate(rows):
            scores[r['action_id']][r['position']] = prediction[i]
        return scores, output
