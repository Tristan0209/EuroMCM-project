from pathlib import Path
import importlib.util
import json
import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent


PROJECT_ROOT = ROOT.parent.parent



BASE_SCRIPT = ROOT / 'quality(sub-model_assessment_aware_ablation).py'

DATA = PROJECT_ROOT / 'AI_datasets/RouterBench/routerbench_0shot.csv'
SAVE = ROOT / 'saved_models'
if not DATA.exists():
    raise FileNotFoundError(f"找不到数据集：{DATA}")
if not BASE_SCRIPT.exists():
    raise FileNotFoundError(f"找不到基础脚本：{BASE_SCRIPT}")

SAMPLE_PER_BENCHMARK = None
SEED = 42
VALID = ['model_id','format_error','empty','refusal','hard_flag','repetition']
TWO = VALID + ['peer_agree','peer_valid']

spec = importlib.util.spec_from_file_location('quality_base', BASE_SCRIPT)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def main():
    SAVE.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA, low_memory=False)
    mc = df.eval_name.astype(str).str.startswith('mmlu-') | df.eval_name.isin(['arc-challenge','hellaswag','winogrande'])
    df = df.loc[mc].copy()
    if SAMPLE_PER_BENCHMARK is not None:
        df = pd.concat([g.sample(min(len(g), SAMPLE_PER_BENCHMARK), random_state=SEED)
                        for _, g in df.groupby('eval_name', sort=True)], ignore_index=True)
    columns = [c for c in df.columns if c.endswith('|model_response')]
    ids = [c[:-len('|model_response')] for c in columns]
    if len(ids) < 2: raise ValueError('Need at least two models')
    parts = []
    for j, model_id in enumerate(ids):
        p = df[['sample_id','eval_name',model_id,model_id+'|model_response']].copy()
        p.columns = ['sample_id','eval_name','score','response']
        p['model_id'] = model_id; p['model_order'] = j
        parts.append(p)
    long = pd.concat(parts, ignore_index=True)
    long.score = pd.to_numeric(long.score, errors='coerce')
    long = long.dropna(subset=['score']).reset_index(drop=True)
    long.response = long.response.map(base.norm_answer)
    feats = pd.DataFrame([base.diagnostic('',r,'multiple_choice') for r in long.response])
    long = pd.concat([long,feats],axis=1)
    lookup = long.set_index(['sample_id','model_order']).answer_choice.to_dict()
    M = len(ids)
    long['peer_choice'] = [lookup.get((sid,(int(j)+1)%M)) for sid,j in zip(long.sample_id,long.model_order)]
    long['peer_valid'] = long.peer_choice.notna().astype(int)
    long['peer_agree'] = [float(a == b) if a is not None and b is not None else np.nan
                          for a,b in zip(long.answer_choice,long.peer_choice)]
    long['model_id'] = pd.Categorical(long.model_id, categories=ids)
    models = {}
    for name, features in [('V',VALID),('VC_2',TWO)]:
        from lightgbm import LGBMRegressor
        reg = LGBMRegressor(n_estimators=250,learning_rate=.05,max_depth=5,
                            random_state=SEED,verbosity=-1,n_jobs=-1)
        reg.fit(long[features],long.score)
        models[name] = reg
        joblib.dump(reg,SAVE/f'{name}.joblib')
        print(name, 'fit rows:',len(long))
    meta = {'models': ids,'valid_features':VALID,'two_features':TWO,
            'score_target':'RouterBench score (not independently verified correctness)',
            'peer_protocol':'fixed next model in model list for offline training; free peer choice is a distribution shift',
            'sample_per_benchmark':SAMPLE_PER_BENCHMARK,'seed':SEED,
            'task_scope':'multiple_choice only'}
    (SAVE/'metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Saved models to',SAVE)

if __name__ == '__main__': main()
