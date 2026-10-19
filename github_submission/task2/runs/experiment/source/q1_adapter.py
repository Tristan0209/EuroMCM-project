import ast
import hashlib
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from input_paths import paired_data

_old_dask = sys.modules.get('dask')
_had_dask = 'dask' in sys.modules
sys.modules['dask'] = None
try:
    from lightgbm import LGBMRegressor
finally:
    if _had_dask:
        sys.modules['dask'] = _old_dask
    else:
        sys.modules.pop('dask', None)

SEED = 42
BASE = ['token_length', 'eval_name', 'model_id']
ABILITY = ['ability_math', 'ability_code', 'ability_general', 'ability_current']
VARIANTS = {'A_Baseline': BASE, 'B_Ability': BASE + ABILITY,
            'C_Difficulty': BASE + ['predicted_difficulty'],
            'D_Full_MIRT': BASE + ABILITY + ['predicted_difficulty', 'ability_gap']}
VALID = ['model_id', 'format_error', 'empty', 'refusal', 'hard_flag', 'repetition']
TWO = VALID + ['peer_agree', 'peer_valid']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def lgbm(**kwargs):
    kwargs.setdefault('n_jobs', 4)
    return LGBMRegressor(**kwargs)


def definitions(path, names, namespace):
    tree = ast.parse(Path(path).read_text(encoding='utf-8-sig'))
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    assert {node.name for node in selected} == set(names)
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace


def load_pre_source(work, model_names):
    source = work / 'task1_quality_pre/Quality(sub-model lightGBM+mirt).py'
    namespace = {'__name__': 'work1_pre_definitions', 'np': np, 'pd': pd,
                 'torch': torch, 'nn': torch.nn, 'LGBMRegressor': lgbm, 'SEED': SEED,
                 'MODEL_NAMES': model_names, 'MODEL_INDEX': {m: i for i, m in enumerate(model_names)},
                 'DIMS': ['Math', 'Code', 'General']}
    return definitions(source, ['get_dimension', 'score_scheme', 'GradedMIRT', 'fit_mirt'], namespace)


def load_post_source(work):
    import re
    namespace = {'__name__': 'work1_post_definitions', 'np': np, 'pd': pd, 'ast': ast, 're': re,
        'MC_BENCHMARKS': {'arc-challenge', 'hellaswag', 'winogrande'},
        'REFUSAL': re.compile(r"\b(i (?:cannot|can't|don't know)|unable to|as an ai)\b|无法回答|不知道|不能回答", re.I)}
    return definitions(work / 'task_1_quality_post/quality(sub-model_assessment_aware_ablation).py',
                       ['kind_of', 'norm_answer', 'choice', 'numeric_final', 'diagnostic'], namespace)


def load_cohort(project, old_run):
    requests = pd.read_json(paired_data(project) / 'requests.jsonl', lines=True)
    requests['group'] = requests.prompt.map(lambda s: hashlib.sha256(' '.join(s.split()).encode()).hexdigest())
    assert requests.groupby('group').split.nunique().max() == 1
    outcomes = pd.read_json(paired_data(project) / 'outcomes.jsonl', lines=True)
    for action, suffix in [('S', 's'), ('L', 'l')]:
        part = outcomes[outcomes.action_id.eq(action)].set_index('request_id')
        assert set(part.index) == set(requests.request_id)
        for dest, source in [('q_', 'quality_true'), ('cost_', 'cost_usd'),
                             ('response_', 'response'), ('output_', 'output_tokens_proxy')]:
            requests[dest + suffix] = requests.request_id.map(part[source])
    requests['delta_q'] = requests.q_l - requests.q_s
    raw = pd.read_csv(project / 'AI_datasets/RouterBench/routerbench_0shot.csv', low_memory=False)
    assert not raw.sample_id.duplicated().any()
    raw = raw.set_index('sample_id').loc[requests.request_id].reset_index()
    models = [c[:-len('|model_response')] for c in raw.columns if c.endswith('|model_response')]
    assert len(models) == 11
    pieces = []
    for model in models:
        p = pd.DataFrame({'request_id': requests.request_id, 'group': requests.group,
                         'split': requests.split, 'eval_name': requests.eval_name,
                         'token_length': requests.prompt_chars, 'model_id': model,
                         'score': pd.to_numeric(raw[model], errors='raise').to_numpy()})
        assert np.isfinite(p.score).all() and p.score.between(0, 1).all()
        pieces.append(p)
    long = pd.concat(pieces, ignore_index=True)
    x = np.load(old_run / 'features.npy', allow_pickle=False)
    gains = pd.read_csv(old_run / 'gain_predictions.csv')
    assert requests.request_id.tolist() == gains.request_id.tolist()
    manifest = __import__('json').loads((old_run / 'manifest.json').read_text(encoding='utf-8'))
    assert digest(old_run / 'features.npy') == manifest['feature_sha256']
    assert x.shape == (len(requests), 392)
    return requests, long, raw, models, x


def validation_partition(frame):

    select = set()
    rng = np.random.default_rng(20261016 + 850)
    groups = frame[frame.split.eq('validation')].drop_duplicates('group')
    for _, block in groups.groupby('eval_name', sort=True):
        ids = block.group.to_numpy().copy(); rng.shuffle(ids)
        select.update(ids[:max(1, len(ids) // 2)])
    phase = frame.split.copy()
    phase.loc[frame.split.eq('validation')] = np.where(frame.loc[frame.split.eq('validation'), 'group'].isin(select), 'select', 'confirm')
    return phase


class PreModel:
    def __init__(self, model_names, benchmark_names):
        self.model_names, self.benchmark_names = list(model_names), list(benchmark_names)

    def basic(self, frame):
        out = frame.copy()
        for col, categories in [('eval_name', self.benchmark_names), ('model_id', self.model_names)]:
            out[col] = pd.Categorical(out[col], categories=categories)
        return out

    def augmented(self, frame):
        out = self.basic(frame)
        out['predicted_difficulty'] = self.difficulty_model.predict(out[['token_length', 'eval_name']])
        model = pd.Categorical(out.model_id, categories=self.model_names).codes
        dimension = np.array([self.dimension(s) for s in out.eval_name.astype(object)])
        for j, name in enumerate(['math', 'code', 'general']):
            out['ability_' + name] = self.theta[model, j]
        di = pd.Categorical(dimension, categories=['Math', 'Code', 'General']).codes
        out['ability_current'] = self.theta[model, di]
        out['ability_gap'] = out.ability_current - out.predicted_difficulty
        return out

    @staticmethod
    def dimension(name):
        name = str(name).lower()
        if any(x in name for x in ['math', 'algebra', 'remainder']): return 'Math'
        if any(x in name for x in ['mbpp', 'computer', 'machine-learning']): return 'Code'
        return 'General'

    def predict(self, frame, variant):
        augmented = self.augmented(frame)
        return np.clip(self.models[variant].predict(augmented[VARIANTS[variant]]), 0, 1)


def mirt_fit(fit, model_names, benchmark_names, ns):
    pipeline = PreModel(model_names, benchmark_names)
    data = pipeline.basic(fit)
    data['sample_id'] = data.request_id
    data['dimension'] = data.eval_name.astype(object).map(ns['get_dimension'])
    data['scheme'] = data.eval_name.astype(object).map(ns['score_scheme'])
    levels = [np.array([0., 1.]), np.linspace(0, 1, 5), np.linspace(0, 1, 11)]
    data['grade'] = [int(np.argmin(np.abs(levels[int(s)] - y))) for s, y in zip(data.scheme, data.score)]
    torch.manual_seed(SEED)
    pipeline.theta, difficulties = ns['fit_mirt'](data, epochs=20)
    items = data.drop_duplicates('request_id').copy()
    items['difficulty'] = items.request_id.map(difficulties)
    pipeline.difficulty_model = lgbm(n_estimators=200, learning_rate=.05, max_depth=5, random_state=SEED, verbosity=-1)
    pipeline.difficulty_model.fit(items[['token_length', 'eval_name']], items.difficulty)
    return pipeline


def fit_pre_pipeline(train, model_names, ns, label):

    benchmarks = sorted(train.eval_name.unique())
    base = PreModel(model_names, benchmarks)
    augmented = base.basic(train).copy()
    extras = ABILITY + ['predicted_difficulty', 'ability_gap']
    for c in extras: augmented[c] = np.nan
    for fold, (fit, hold) in enumerate(GroupKFold(3).split(train, groups=train.group)):
        pipeline = mirt_fit(train.iloc[fit], model_names, benchmarks, ns)
        transformed = pipeline.augmented(train.iloc[hold])
        augmented.loc[train.index[hold], extras] = transformed[extras].to_numpy()
        print(f'PRE {label} inner={fold} fit_groups={train.iloc[fit].group.nunique()}', flush=True)
    assert np.isfinite(augmented[extras]).all().all()
    final = mirt_fit(train, model_names, benchmarks, ns)
    final.models = {}
    for variant, cols in VARIANTS.items():
        model = lgbm(n_estimators=200, learning_rate=.05, max_depth=5, random_state=SEED, verbosity=-1)
        model.fit(augmented[cols], train.score)
        final.models[variant] = model
    return final


def pre_crossfit(frame, long, model_names, work, out):
    ns = load_pre_source(work, model_names)
    train = long[long.split.eq('train')].copy()
    result = long[['request_id', 'group', 'split', 'model_id']].copy()
    for variant in VARIANTS: result[variant] = np.nan
    fit_ids = np.empty(len(long), dtype=object)
    provenance = {}
    for fold, (fit, hold) in enumerate(GroupKFold(3).split(train, groups=train.group)):
        fit_frame, hold_frame = train.iloc[fit], train.iloc[hold]
        assert set(fit_frame.group).isdisjoint(hold_frame.group)
        model = fit_pre_pipeline(fit_frame, model_names, ns, f'outer_{fold}')
        for variant in VARIANTS:
            result.loc[hold_frame.index, variant] = model.predict(hold_frame, variant)
        name = f'Pre_outer_{fold}'
        fit_ids[hold_frame.index] = name
        provenance[name] = {'training_groups': sorted(fit_frame.group.unique()), 'oof': True}
        joblib.dump(model, out / f'pre_outer_{fold}.joblib', compress=3)
    final = fit_pre_pipeline(train, model_names, ns, 'full_train')
    hold = long[~long.split.eq('train')]
    for variant in VARIANTS: result.loc[hold.index, variant] = final.predict(hold, variant)
    fit_ids[hold.index] = 'Pre_full_train'
    provenance['Pre_full_train'] = {'training_groups': sorted(train.group.unique()), 'oof': False}
    result['fit_id'] = fit_ids
    assert np.isfinite(result[list(VARIANTS)]).all().all()
    result.to_csv(out / 'pre_predictions.csv.gz', index=False, compression='gzip')
    joblib.dump(final, out / 'pre_full_train.joblib', compress=3)
    return result, provenance


def post_crossfit(frame, raw, model_names, work, out):
    ns = load_post_source(work)
    mc = frame.task_kind.eq('multiple_choice').to_numpy()
    mids = {'S': 'mistralai/mixtral-8x7b-chat', 'L': 'gpt-4-1106-preview'}
    rows = []
    for model in model_names:
        for i in np.flatnonzero(mc):
            response = ns['norm_answer'](raw.iloc[i][model + '|model_response'])
            diagnostic = ns['diagnostic'](frame.iloc[i].prompt, response, 'multiple_choice')
            rows.append({'position': i, 'request_id': frame.iloc[i].request_id, 'group': frame.iloc[i].group,
                         'split': frame.iloc[i].split, 'model_id': model, 'score': float(raw.iloc[i][model]), **diagnostic})
    data = pd.DataFrame(rows)
    data['model_id'] = pd.Categorical(data.model_id, categories=model_names)
    pair = data[data.model_id.isin(mids.values())].copy()
    choices = pair.set_index(['request_id', 'model_id']).answer_choice.to_dict()
    peers = [choices[(r.request_id, mids['L'] if r.model_id == mids['S'] else mids['S'])] for r in pair.itertuples()]
    pair['peer_valid'] = [int(pd.notna(p)) for p in peers]
    pair['peer_agree'] = [float(a == b) if pd.notna(a) and pd.notna(b) else np.nan for a, b in zip(pair.answer_choice, peers)]
    predictions, provenance, final_models = [], {}, {}
    for name, source, cols in [('V', data, VALID), ('VC2', pair, TWO)]:
        source = source.reset_index(drop=True)
        train = source[source.split.eq('train')]
        values = np.full(len(source), np.nan); fit_ids = np.empty(len(source), dtype=object)
        for fold, (fit, hold) in enumerate(GroupKFold(3).split(train, groups=train.group)):
            tr, te = train.iloc[fit], train.iloc[hold]
            net = lgbm(n_estimators=250, learning_rate=.05, max_depth=5, random_state=SEED, verbosity=-1)
            net.fit(tr[cols], tr.score)
            values[te.index] = np.clip(net.predict(te[cols]), 0, 1)
            fit_id = f'{name}_fold_{fold}'; fit_ids[te.index] = fit_id
            provenance[fit_id] = {'training_groups': sorted(tr.group.unique()), 'oof': True}
        net = lgbm(n_estimators=250, learning_rate=.05, max_depth=5, random_state=SEED, verbosity=-1)
        net.fit(train[cols], train.score)
        te = source[~source.split.eq('train')]
        values[te.index] = np.clip(net.predict(te[cols]), 0, 1)
        fit_id = f'{name}_full_train'; fit_ids[te.index] = fit_id
        provenance[fit_id] = {'training_groups': sorted(train.group.unique()), 'oof': False}
        source['prediction'] = values; source['fit_id'] = fit_ids; source['assessor'] = name
        predictions.append(source[['request_id', 'group', 'split', 'model_id', 'score', 'prediction', 'fit_id', 'assessor']])
        final_models[name] = net
        for action, model_id in mids.items():
            by_id = source[source.model_id.eq(model_id)].set_index('request_id')
            frame[name.lower() + '_' + action.lower()] = frame.request_id.map(by_id.prediction)
        print('POST trained', name, 'rows=', len(train), flush=True)
    for action in ['s', 'l']:
        frame['hard_' + action] = [bool(ns['diagnostic'](r.prompt, getattr(r, 'response_' + action), ns['kind_of'](r.eval_name))['hard_flag'])
                                  for r in frame.itertuples()]
    pd.concat(predictions).to_csv(out / 'post_predictions.csv.gz', index=False, compression='gzip')
    joblib.dump({'models': final_models, 'model_names': model_names, 'features': {'V': VALID, 'VC2': TWO},
                 'peer_protocol': 'actual S/L pair; retrained, not original fixed-next-11 checkpoint'}, out / 'post_models.joblib', compress=3)
    return frame, provenance
