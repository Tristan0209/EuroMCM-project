import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
OUT = HERE / 'output'
MIDS = {'S': 'mistralai/mixtral-8x7b-chat', 'L': 'gpt-4-1106-preview'}
REQUEST_FIELDS = ['request_id', 'source_row', 'eval_name', 'task_type', 'prompt_chars', 'input_tokens_proxy', 'split']


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def paths():
    return {k: PROJECT / v for k, v in read_json(HERE / 'source_paths.json').items()}


def requests_source():

    with paths()['paired_requests'].open(encoding='utf-8') as f:
        return pd.DataFrame([{k: r[k] for k in REQUEST_FIELDS} for r in map(json.loads, f)])


def decisions_source():
    cfg = read_json(HERE / 'config.json')
    d = pd.read_csv(paths()['run'] / 'decisions.csv.gz', usecols=[
        'strategy', 'phase', 'rep', 'request_id', 'task_type', 'initial_l', 'upgraded', 'final_l', 'calls_s', 'calls_l'])
    return d[d.strategy.eq(cfg['source_strategy']) & d.phase.eq(cfg['source_phase']) & d.rep.eq(cfg['source_rep'])].copy()


def training_lengths(requests):

    train = requests[requests.split.eq('train')].set_index('request_id')
    rows = []
    with paths()['paired_outcomes'].open(encoding='utf-8') as f:
        for line in f:
            r = json.loads(line)
            if r['request_id'] not in train.index:
                continue
            rows.append((r['request_id'], train.at[r['request_id'], 'task_type'], r['action_id'], r['output_tokens_proxy']))
    history = pd.DataFrame(rows, columns=['request_id', 'task_type', 'model_type', 'output_tokens_proxy'])
    if history.duplicated(['request_id', 'model_type']).any():
        raise ValueError('Duplicate train S/L outcome')
    if len(history) != 2 * len(train) or not set(history.model_type) == {'S', 'L'}:
        raise ValueError('Incomplete training outcomes')
    if not np.isfinite(history.output_tokens_proxy).all() or not history.output_tokens_proxy.ge(1).all():
        raise ValueError('Invalid training output proxy')
    stats = history.groupby(['task_type', 'model_type'], sort=True).output_tokens_proxy.agg(['count', 'mean']).reset_index()
    stats['estimated_output_tokens_proxy'] = np.maximum(1, np.floor(stats['mean'] + 0.5)).astype(np.int64)
    stats['source_partition'] = 'train_only'
    return stats


def service_estimates(requests, stats, cfg, profile):
    x = requests.input_tokens_proxy.to_numpy(dtype=float) / 1000
    result = {}
    for a in ['S', 'L']:
        lookup = stats[stats.model_type.eq(a)].set_index('task_type').estimated_output_tokens_proxy
        output = requests.task_type.map(lookup)
        if output.isna().any():
            raise ValueError('No train-only output estimate for task; explicitly configure a fallback before adding new tasks')
        y = (output.to_numpy(dtype=float) - 1) / 1000
        features = np.column_stack([np.ones(len(x)), x, y, x*x, x*y, y*y])
        multiplier = 1 if a == 'S' else cfg['large_service_multiplier']
        first = features @ np.array(profile['prefill_coefficients']) * multiplier
        total = first + features @ np.array(profile['decode_coefficients']) * multiplier
        result[a] = {'output': output.to_numpy(), 'first': first, 'total': total}
    return result


def frozen_route(task, benchmark, q_s, q_l, prompt_chars, registry, cost, params, fallback):

    if benchmark not in registry or task in fallback or task != 'General':
        return 'L'
    costs = {a: np.array([1, prompt_chars / 1000]) @ cost['coefficients'][a.lower()] for a in ['S', 'L']}
    h = q_l - q_s - params['lambda'] * (costs['L'] - costs['S']) / cost['normalization_c0']
    return 'L' if h >= params['tau'] else 'S'


def utc_strings(values):
    return pd.Series(values).dt.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
