import argparse
import json
import shutil
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from input_paths import frozen_features, paired_data

from q1_adapter import (VARIANTS, digest, load_cohort, post_crossfit,
                        pre_crossfit, validation_partition)

HERE = Path(__file__).resolve().parent
MIDS = {'s': 'mistralai/mixtral-8x7b-chat', 'l': 'gpt-4-1106-preview'}


def save(path, value):
    def clean(item):
        if isinstance(item,dict):return {str(k):clean(v) for k,v in item.items()}
        if isinstance(item,(list,tuple,np.ndarray)):return [clean(v) for v in item]
        if isinstance(item,(float,np.floating)):return float(item) if np.isfinite(item) else None
        if isinstance(item,(np.integer,)):return int(item)
        if isinstance(item,(np.bool_,)):return bool(item)
        if isinstance(item,Path):return str(item)
        return item
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False, default=str), encoding='utf-8')



save_json = save


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-project', type=Path, required=True)
    parser.add_argument('--out', type=Path, default=HERE / 'runs/work1_integrated_20261016')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    project, out = args.source_project.resolve(), args.out.resolve()
    if out.exists() and not args.resume: raise FileExistsError(out)
    out.mkdir(parents=True, exist_ok=True)
    work = project / 'Work 1'
    old_run = frozen_features(project)
    files = [p for p in work.rglob('*') if p.is_file() and p.suffix in ['.py', '.json', '.joblib', '.csv']
             and '__pycache__' not in p.parts and ('outputs' not in p.parts or p.name in ['cost_summary.csv', 'cost_summary.json'])]
    records = {'work1_source': {str(p): digest(p) for p in files},
               'raw_source': str(project / 'AI_datasets/RouterBench/routerbench_0shot.csv'),
               'raw_sha256': digest(project / 'AI_datasets/RouterBench/routerbench_0shot.csv'),
               'old_feature_sha256': digest(old_run / 'features.npy'),
               'adapter_sha256': digest(HERE / 'q1_adapter.py'),
               'old_split_sha256': digest(paired_data(project) / 'request_splits.csv'),
               'existing_post_weights_used_for_research': False,
               'reason': 'Work 1 export used all historical rows and has no held-out-group manifest',
               'pre_original_v2_file_available': (project / 'dataset/routerbench_0shot-v2.csv').exists(),
               'task_metadata_assumption': 'registered task/benchmark label available before generation',
               'test_status': 'previously_inspected_research_holdout_not_globally_unseen'}
    if (out / 'q1_source_audit.json').exists():
        assert json.loads((out / 'q1_source_audit.json').read_text(encoding='utf-8')) == records
    save(out / 'q1_source_audit.json', records)
    snapshots = out / 'work1_source'; snapshots.mkdir(exist_ok=True)
    for p in files:
        if p.suffix in ['.py', '.json']:
            dest = snapshots / p.relative_to(work); dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, dest)
    frame, long, raw, model_names, base_x = load_cohort(project, old_run)
    frame['phase'] = validation_partition(frame)
    print('COHORT', len(frame), 'phases=', frame.phase.value_counts().to_dict(), flush=True)
    frame[['request_id', 'group', 'split', 'phase', 'task_type', 'eval_name']].to_csv(out / 'splits.csv', index=False)
    torch.set_num_threads(4)
    with threadpool_limits(limits=2):
        if args.resume and (out / 'pre_predictions.csv.gz').exists():
            pre = pd.read_csv(out / 'pre_predictions.csv.gz')
            pre_provenance = json.loads((out / 'pre_provenance.json').read_text(encoding='utf-8'))
        else:
            pre, pre_provenance = pre_crossfit(frame, long, model_names, work, out)
            save(out / 'pre_provenance.json', pre_provenance)
        indices = frame.set_index('request_id').phase
        selection = pre[pre.request_id.map(indices).eq('select') & pre.model_id.isin(MIDS.values())].copy()
        truth = long.set_index(['request_id', 'model_id']).score
        selection['truth'] = [truth.loc[(r.request_id, r.model_id)] for r in selection.itertuples()]
        metrics = [{'variant': variant, 'select_rmse': float(np.sqrt(((selection[variant]-selection.truth)**2).mean())),
                    'select_mae': float((selection[variant]-selection.truth).abs().mean())} for variant in VARIANTS]
        winner = min(metrics, key=lambda r: (r['select_rmse'], r['variant']))['variant']
        save(out / 'pre_selection.json', {'selected_on': 'selection_partition_only_S_L', 'winner': winner, 'candidates': metrics})
        print('PRE SELECTED', winner, metrics, flush=True)
        for action, mid in MIDS.items():
            values = pre[pre.model_id.eq(mid)].set_index('request_id')[winner]
            frame['pre_' + action] = frame.request_id.map(values)
        frame['pre_delta'] = frame.pre_l - frame.pre_s
        scaler = StandardScaler().fit(frame.loc[frame.phase.eq('train'), ['pre_s', 'pre_l', 'pre_delta']])
        x = np.column_stack([base_x, scaler.transform(frame[['pre_s', 'pre_l', 'pre_delta']])]).astype(np.float32)
        assert np.isfinite(x).all() and x.shape == (len(frame), 395)
        np.save(out / 'features.npy', x, allow_pickle=False)
        joblib.dump(scaler, out / 'pre_feature_scaler.joblib')
        if args.resume and (out / 'post_frame.joblib').exists():
            post_columns = joblib.load(out / 'post_frame.joblib')
            for col in post_columns: frame[col] = post_columns[col]
        else:
            frame, post_provenance = post_crossfit(frame, raw, model_names, work, out)
            save(out / 'post_provenance.json', post_provenance)
            joblib.dump(frame[['v_s', 'v_l', 'vc2_s', 'vc2_l', 'hard_s', 'hard_l']], out / 'post_frame.joblib', compress=3)
    joblib.dump(frame, out / 'frame.joblib', compress=3)
    save(out / 'prepared.json', {'completed': True, 'requests': len(frame), 'models_for_pre_training': model_names,
         'phases': frame.phase.value_counts().to_dict(), 'feature_dimension': 395,
         'pre_winner': winner, 'features_sha256': digest(out / 'features.npy'),
         'pre_train_oof': True, 'post_train_oof': True, 'original_work1_modified': False})
    print('PREPARED', out, flush=True)


if __name__ == '__main__': main()
