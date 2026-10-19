import numpy as np


def cvar(losses, alpha=.95, weights=None):
    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or not 0 <= alpha < 1:
        raise ValueError('Finite nonempty 1-D losses and 0<=alpha<1 required')
    w = np.full(len(values), 1 / len(values)) if weights is None else np.asarray(weights, float)
    if w.shape != values.shape or not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1, rtol=0, atol=1e-12):
        raise ValueError('Invalid normalized probabilities')
    order = np.argsort(values, kind='stable')
    positive = order[w[order] > 0]
    cumulative = np.cumsum(w[positive])
    cutoff = min(int(np.searchsorted(cumulative, alpha, side='left')), len(positive) - 1)
    var = float(values[positive[cutoff]])
    mass = np.where(values > var, w, 0.)
    remainder = max(0., (1 - alpha) - mass.sum())
    boundary = values == var
    if remainder:
        mass[boundary] = w[boundary] * remainder / w[boundary].sum()
    tail = mass / (1 - alpha)
    return {'var': var, 'cvar': float(tail @ values), 'alpha': alpha,
            'tail_mass': float(mass.sum()), 'tail_ess': float(1 / (tail @ tail)),
            'world_ess': float(1 / (w @ w)), 'positive_tail_worlds': int((mass > 0).sum()),
            'distinct_tail_losses': int(np.unique(values[mass > 0]).size),
            'tail_probability_weights': tail}


def mixture_weights(regimes, probability):
    regimes = np.asarray(regimes)
    if not 0 <= probability <= 1 or not np.isin(regimes, [0, 1]).all():
        raise ValueError('Binary regimes and probability in [0,1] required')
    w = np.zeros(len(regimes))
    for regime, target in [(0, 1 - probability), (1, probability)]:
        mask = regimes == regime
        if target and not mask.any():
            raise ValueError('Positive target mass without sampled regime')
        if mask.any():
            w[mask] = target / mask.sum()
    return w


def paired_mixture_cvar(proposed, baseline, regimes, probability=.2, alpha=.95, bootstrap=2000, seed=20261018):
    proposed, baseline, regimes = map(np.asarray, [proposed, baseline, regimes])
    if proposed.shape != baseline.shape or proposed.shape != regimes.shape or bootstrap < 2:
        raise ValueError('Paired vectors and at least two bootstrap replicates required')
    weights = mixture_weights(regimes, probability)
    p, b = cvar(proposed, alpha, weights), cvar(baseline, alpha, weights)
    rng = np.random.default_rng(seed)
    strata = [np.flatnonzero(regimes == s) for s in [0, 1]]
    draws = []
    for _ in range(bootstrap):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata if len(s)])
        w = mixture_weights(regimes[idx], probability)
        pp, bb = cvar(proposed[idx], alpha, w)['cvar'], cvar(baseline[idx], alpha, w)['cvar']
        draws.append((pp, bb, pp - bb))
    intervals = np.quantile(np.asarray(draws), [.025, .975], axis=0)
    return {'proposed': {k: v for k, v in p.items() if k != 'tail_probability_weights'},
            'baseline': {k: v for k, v in b.items() if k != 'tail_probability_weights'},
            'proposed_minus_baseline': p['cvar'] - b['cvar'],
            'ci95_proposed': intervals[:, 0].tolist(), 'ci95_baseline': intervals[:, 1].tolist(),
            'ci95_difference': intervals[:, 2].tolist(), 'bootstrap_replicates': bootstrap,
            'probability': probability, 'worlds': len(proposed),
            'method': 'paired world resampling within fixed regime strata; percentile intervals; not new worlds'}
