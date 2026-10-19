import time
import warnings

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import BayesianRidge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from threadpoolctl import threadpool_limits


class DiscreteGainLogistic:

    def __init__(self, c):
        self.model = LogisticRegression(C=c, solver='lbfgs', max_iter=1500, tol=1e-4)

    def fit(self, x, y):
        self.values, labels = np.unique(y, return_inverse=True)
        if len(self.values) > 17:
            raise ValueError('This adapter requires discrete gains, not arbitrary continuous targets')
        self.model.fit(x, labels)
        return self

    def predict(self, x):
        return self.model.predict_proba(x) @ self.values[self.model.classes_]


class LinearArms:





    def __init__(self, ridge):
        self.ridge = ridge

    @staticmethod
    def augment(x):
        return np.column_stack([np.ones(len(x)), x]).astype(np.float64)

    def fit(self, x, rewards):
        z = self.augment(x)
        gram = z.T @ z + self.ridge * np.eye(z.shape[1])
        self.a = np.stack([gram.copy(), gram.copy()])
        self.b = np.stack([z.T @ rewards[:, a] for a in range(2)])
        self.refresh()
        prediction = z @ self.theta.T
        self.noise = np.maximum(((rewards - prediction) ** 2).mean(axis=0), 1e-6)
        return self

    def refresh(self):
        self.inverse = np.stack([cho_solve(cho_factor(a, lower=True), np.eye(len(a))) for a in self.a])
        self.theta = np.einsum('aij,aj->ai', self.inverse, self.b)

    def moments(self, x):
        z = self.augment(x)
        mean = z @ self.theta.T
        variance = np.column_stack([np.einsum('ij,ij->i', z @ inv, z) for inv in self.inverse])
        return mean, np.maximum(variance, 0)

    def score(self, x, kind, rng=None):
        mean, variance = self.moments(x)
        if kind == 'LinUCB':

            utility = mean + np.sqrt(variance)
        elif kind == 'Thompson Sampling':


            utility = mean + rng.normal(size=mean.shape) * np.sqrt(variance * self.noise)
        else:
            raise ValueError(kind)
        return utility[:, 1] - utility[:, 0]

    def update(self, x, selected_actions, observed_rewards):
        z = self.augment(x)
        assert len(z) == len(selected_actions) == len(observed_rewards)
        for action in range(2):
            mask = selected_actions == action
            self.a[action] += z[mask].T @ z[mask]
            self.b[action] += z[mask].T @ observed_rewards[mask]
        self.refresh()


def fit_additional(x, frame, seed):
    train = frame.split.eq('train').to_numpy()
    val = frame.split.eq('validation').to_numpy()
    target = frame.delta_true.to_numpy()
    candidates = {
        'Bayesian Ridge': [BayesianRidge(max_iter=500, tol=1e-4)],
        'Logistic Regression': [DiscreteGainLogistic(c) for c in [.01, .1, 1.]],
        'Random Forest': [RandomForestRegressor(n_estimators=200, max_depth=depth,
                           min_samples_leaf=leaf, max_features=.5, n_jobs=4, random_state=seed)
                          for depth, leaf in [(12, 20), (None, 20), (None, 50)]],
        'CART': [DecisionTreeRegressor(max_depth=depth, min_samples_leaf=leaf, random_state=seed)
                 for depth in [4, 8, 12] for leaf in [20, 100]]}
    fitted, predictions, logs = {}, {}, []
    with threadpool_limits(limits=2):
        for name, models in candidates.items():
            best, loss = None, np.inf
            for number, model in enumerate(models):
                start = time.perf_counter()
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always', ConvergenceWarning)
                    model.fit(x[train], target[train])
                convergence = [str(w.message) for w in caught if issubclass(w.category, ConvergenceWarning)]
                prediction = np.clip(model.predict(x[val]), -1, 1)
                mse = float(((prediction - target[val]) ** 2).mean())
                logs.append({'family': name, 'candidate': number, 'validation_gain_mse': mse,
                             'fit_wall_s': time.perf_counter() - start, 'convergence_warnings': convergence})
                print(f'FIT {name} candidate={number} val_MSE={mse:.6f} seconds={logs[-1]["fit_wall_s"]:.1f}', flush=True)
                if not convergence and mse < loss:
                    best, loss = model, mse
            if best is None:
                raise RuntimeError(f'No converged candidate for {name}')
            fitted[name] = best
            mean = np.clip(best.predict(x), -1, 1)
            predictions[name] = {'mu': mean, 'std': np.zeros(len(x)), 'scores': mean[None, :], 'k_grid': [0.]}

        best, loss = None, np.inf
        rewards = frame[['q_s', 'q_l']].to_numpy()
        for ridge in [1., 10., 100.]:
            start = time.perf_counter()
            model = LinearArms(ridge).fit(x[train], rewards[train])
            mean, _ = model.moments(x[val])
            mse = float(((np.clip(mean[:, 1] - mean[:, 0], -1, 1) - target[val]) ** 2).mean())
            logs.append({'family': 'Linear arms warm start', 'candidate': ridge,
                         'validation_gain_mse': mse, 'fit_wall_s': time.perf_counter() - start,
                         'convergence_warnings': []})
            if mse < loss:
                best, loss = model, mse
        fitted['linear_arms'] = best
        mean, _ = best.moments(x)
        mu = np.clip(mean[:, 1] - mean[:, 0], -1, 1)
        predictions['LinUCB'] = {'mu': mu, 'std': np.zeros(len(x)),
                                 'scores': best.score(x, 'LinUCB')[None, :], 'k_grid': [0.]}
        predictions['Thompson Sampling'] = {'mu': mu, 'std': np.zeros(len(x)), 'k_grid': [0.],
            'scores': np.stack([best.score(x, 'Thompson Sampling', np.random.default_rng(seed + 100 + r))
                                for r in range(5)])}
    return fitted, predictions, logs
