import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
CANONICAL = Path('C:/Users/吴连杰/.codex/worktrees/47e1/EuroMCM-project')
RELEASE = CANONICAL / 'TASK 3/task3_model/results/robustness_v2/final_release'


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(name, value):
    (HERE / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding='utf-8')


def protected_snapshot():

    files = [p for directory in ['task2', 'TASK 3', 'Work 1', 'latency_environment']
             for p in (PROJECT / directory).rglob('*') if p.is_file()]
    files += [PROJECT / 'AI_datasets/RouterBench/routerbench_0shot.csv',
              Path('C:/Users/吴连杰/Desktop/2026_Problem_Lambda.pdf')]
    return {str(p): sha(p) for p in sorted(set(files)) if p.is_file()}


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    if args.verify:
        before = json.loads((HERE / 'protected_before.json').read_text(encoding='utf-8'))
        after = protected_snapshot()
        changed = [p for p, h in before.items() if after.get(p) != h]
        added = sorted(set(after) - set(before))
        save('protection_after.json', {'status': 'PASS' if not changed and not added else 'FAIL',
                                     'protected_files': len(before), 'changed_or_missing': changed, 'added_upstream': added})
        assert not changed and not added, (changed, added)
        print('PASS D: protected content hashes:', len(before))
        return
    assert not (HERE / 'protected_before.json').exists(), 'Preserve initial evidence'
    save('protected_before.json', protected_snapshot())
    sources = list((PROJECT / 'task2').glob('*.py'))
    sources += list((PROJECT / 'task2/support').glob('*.py'))
    sources += list(RELEASE.glob('*.py')) + list((RELEASE / 'v2').glob('*.py'))
    sources += list((RELEASE / 'tests').glob('*.py'))
    sources += list((CANONICAL / 'TASK 3/task3_model/src').glob('*.py'))
    sources += list((PROJECT / 'TASK 3/data_preparation').glob('*.py'))
    sources += [PROJECT / 'TASK 3/task3_model/task3_final_paper' / name
                for name in ['Task3_Paper_EN.md', 'Task3_Equations.md', 'Task3_Source_Audit.md', 'Task3_Unresolved_Issues.md']]
    rows = []
    for p in sorted(set(sources)):
        text = p.read_text(encoding='utf-8-sig')
        symbols = []
        if p.suffix == '.py':
            tree = ast.parse(text)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    symbols.append({'name': node.name, 'line': node.lineno,
                                    'signature': ast.get_source_segment(text, node).splitlines()[0]})
        rows.append({'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size,
                     'scope': 'full UTF-8/AST read; semantic audit focuses on routing, simulation, accounting and analysis interfaces',
                     'symbols': symbols})
    save('source_inventory.json', rows)
    relative = [str(p.relative_to(PROJECT)) for p in (PROJECT / 'task2').glob('*.py')]
    relative += ['task2/runs/experiment/' + name for name in [
        'final_model_choice.json', 'final_safety_calibration.json', 'finite_sample_calibration.json',
        'pre_full_train.joblib', 'pre_predictions.csv.gz', 'selected_parameters.json',
        'cost_predictor.json', 'decisions.csv.gz', 'splits.csv']]
    relative += ['TASK 3/task3_model/src/' + p.name for p in (PROJECT / 'TASK 3/task3_model/src').glob('*.py')]
    relative += ['TASK 3/task3_model/results/data/' + p.name for p in (PROJECT / 'TASK 3/task3_model/results/data').glob('*') if p.is_file()]
    comparisons = []
    for rel in relative:
        a, b = PROJECT / rel, CANONICAL / rel
        ah, bh = sha(a) if a.is_file() else None, sha(b) if b.is_file() else None
        comparisons.append({'relative_path': rel, 'D_sha256': ah, 'C_sha256': bh,
                            'identical': ah is not None and ah == bh})
    save('location_comparison.json', comparisons)
    import pandas as pd
    comparison = pd.read_csv(RELEASE / 'FINAL_COMPARISON.csv')
    pilot = pd.read_csv(RELEASE / 'results/pilot/raw_runs.csv')
    runtime = comparison.groupby('strategy')[['runtime_seconds', 'solver_wall_seconds']].agg(['mean', 'max'])
    runtime.columns = ['_'.join(c) for c in runtime.columns]
    runtime.reset_index().to_csv(HERE / 'historical_runtime.csv', index=False)
    save('interface_summary.json', {
        'paper_root': str(PROJECT / 'TASK 3/task3_model/task3_final_paper'), 'canonical_release': str(RELEASE),
        'D_release_present': (PROJECT / 'TASK 3/task3_model/results/robustness_v2/final_release').exists(),
        'full_source_files_parsed': len(rows), 'location_pairs': len(comparisons),
        'location_mismatches': [r['relative_path'] for r in comparisons if not r['identical']],
        'final_comparison_rows': len(comparison), 'core_independent_seeds': int(comparison.seed.nunique()),
        'core_policies': sorted(comparison.strategy.unique()),
        'pilot_runtime': pilot[['strategy', 'n', 'runtime_seconds']].to_dict('records'),
        'new_performance_experiments_executed': 0, 'new_model_training_executed': 0,
    })
    print('Source files:', len(rows), 'location mismatches:', sum(not r['identical'] for r in comparisons))


if __name__ == '__main__':
    main()
