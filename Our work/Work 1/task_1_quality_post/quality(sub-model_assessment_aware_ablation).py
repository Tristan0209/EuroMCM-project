from pathlib import Path
import ast
import re
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from lightgbm import LGBMRegressor

SEED = 42

ROOT = Path(__file__).resolve().parent


PROJECT_ROOT = ROOT.parent.parent
DATA = PROJECT_ROOT / 'AI_datasets/RouterBench/routerbench_0shot.csv'
if not DATA.exists():
    DATA = Path(__file__).with_name('routerbench_0shot-v2.csv')
OUT = Path('../../final_quality_outputs')
SAMPLE_PER_BENCHMARK = 30
MAX_MC_HOLDOUTS = 12
MIN_MC_REQUESTS = 15
LOW_SCORE = .25
RNG = np.random.default_rng(SEED)

MC_BENCHMARKS = {'arc-challenge', 'hellaswag', 'winogrande'}
REFUSAL = re.compile(r"\b(i (?:cannot|can't|don't know)|unable to|as an ai)\b|无法回答|不知道|不能回答",re.I)
NUM = re.compile(r'(?<![\w.])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?![\w.])')

def kind_of(name):
    n = str(name).lower()
    if n.startswith('mmlu-') or n in MC_BENCHMARKS: return 'multiple_choice'
    if 'riddle' in n: return 'bracket_riddle'
    if 'mbpp' in n: return 'code'
    if any(x in n for x in ['math', 'algebra', 'remainder']): return 'math'
    return 'open_or_other'

def norm_answer(x):
    if pd.isna(x): return ''
    s = str(x)
    if s.startswith('[') and s.endswith(']'):
        try:
            v = ast.literal_eval(s)
            if isinstance(v,(list,tuple)): return '\n'.join(map(str,v))
        except (SyntaxError,ValueError,RecursionError,MemoryError,TypeError): pass
    return s

def choice(text):
    text=text.strip().strip('[](){}"\'` ').upper()
    m=re.fullmatch(r'(?:OPTION\s*|ANSWER\s*[:：]?\s*)?([A-D])(?:[.。])?',text)
    return m.group(1) if m else None

def numeric_final(text):

    matches=re.findall(r'(?:final answer|the answer is|答案(?:是|为|：|:)|\\boxed\{)\s*[:：]?\s*([-+]?[\d,]+(?:\.\d+)?)',text,re.I)
    if not matches: return None
    try: return round(float(matches[-1].replace(',','')),8)
    except ValueError: return None

def diagnostic(prompt,response,kind):
    t=re.findall(r'[\u3400-\u9fff]|[a-z0-9_]+',response.lower())
    grams=list(zip(t,t[1:],t[2:]))
    repetition=1-len(set(grams))/len(grams) if len(grams)>=10 else 0.0
    a=response.strip()
    empty=int(not a)
    refusal=int(bool(REFUSAL.search(a)))
    opt=choice(a) if kind=='multiple_choice' else None
    is_mc=int(kind=='multiple_choice')
    bracket_ok=int(bool(re.search(r'\[[^\[\]\s]{1,6}\]',a))) if kind=='bracket_riddle' else np.nan
    fences=re.findall(r'```(?:python|py)\s*\n(.*?)```',a,re.I|re.S)
    syntax_ok=np.nan
    if kind=='code' and fences:
        try:
            for code in fences: ast.parse(code)
            syntax_ok=1.0
        except SyntaxError: syntax_ok=0.0
    format_error=(kind=='multiple_choice' and opt is None) or (kind=='bracket_riddle' and bracket_ok==0) or (kind=='code' and syntax_ok==0)
    hard=int(bool(empty or refusal or format_error or repetition>.65))
    marked_number=numeric_final(a) if kind=='math' else None
    return {'response_tokens':len(t),'log_response_tokens':np.log1p(len(t)),
            'log_prompt_chars':np.log1p(len(prompt)),'response_chars':len(a),
            'line_count':a.count('\n')+1,'repetition':repetition,
            'empty':empty,'refusal':refusal,'format_error':int(format_error),
            'hard_flag':hard,'choice_valid':float(opt is not None) if is_mc else np.nan,
            'bracket_valid':bracket_ok,'python_syntax_valid':syntax_ok,
            'contains_code_fence':int('```' in a),
            'contains_list':int(bool(re.search(r'(?m)^\s*(?:[-*]|\d+[.)])\s+',a))),
            'answer_choice':opt,'math_final_number':marked_number,
            'numeric_answer_extracted':int(marked_number is not None) if kind=='math' else np.nan,
            'has_executable_test_result':0, 'reference_correctness_known':0}

def stats(y,p):
    y=np.asarray(y,float);p=np.clip(np.asarray(p,float),0,1)
    v=len(y)>4 and np.unique(y).size>1 and np.unique(p).size>1
    low=(y<=LOW_SCORE)
    return dict(n=len(y),MAE=float(mean_absolute_error(y,p)),RMSE=float(np.sqrt(mean_squared_error(y,p))),
                Spearman=float(spearmanr(y,p).statistic) if v else np.nan,
                low_AUC=float(roc_auc_score(low,-p)) if np.unique(low).size==2 else np.nan)

def rank_metric(df,pred):
    vals=[]
    for _,g in df.groupby('sample_id',sort=False):
        if g.score.nunique()>1 and g[pred].nunique()>1:
            r=spearmanr(g.score,g[pred]).statistic
            if np.isfinite(r): vals.append(float(r))
    return len(vals), float(np.mean(vals)) if vals else np.nan

def fit_predict(tr,te,features):
    model=LGBMRegressor(n_estimators=250,learning_rate=.05,max_depth=5,random_state=SEED,verbosity=-1,n_jobs=-1)
    model.fit(tr[features],tr.score)
    return np.clip(model.predict(te[features]),0,1)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    wide=pd.read_csv(DATA,low_memory=False)
    response_columns=[c for c in wide.columns if c.endswith('|model_response')]
    if not response_columns: raise ValueError('RouterBench |model_response columns not found')
    if SAMPLE_PER_BENCHMARK is not None:
        wide=pd.concat([g.sample(min(SAMPLE_PER_BENCHMARK,len(g)),random_state=SEED)
                        for _,g in wide.groupby('eval_name',sort=True)],ignore_index=True)
    pieces=[]
    for j,col in enumerate(response_columns):
        mid=col[:-len('|model_response')]
        required=['sample_id','prompt','eval_name',mid,col]
        part=wide[required].copy()
        part.columns=['sample_id','prompt','eval_name','score','response']
        part['model_id']=mid;part['model_order']=j
        cost=mid+'|total_cost'
        part['cost']=pd.to_numeric(wide[cost],errors='coerce').values if cost in wide else np.nan
        pieces.append(part)
    df=pd.concat(pieces,ignore_index=True)
    df.score=pd.to_numeric(df.score,errors='coerce')
    df=df.dropna(subset=['score']).reset_index(drop=True)
    df.prompt=df.prompt.map(norm_answer);df.response=df.response.map(norm_answer)
    df['task_kind']=df.eval_name.map(kind_of)
    print('Requests:',df.sample_id.nunique(),'responses:',len(df),'benchmarks:',df.eval_name.nunique(),flush=True)
    feats=pd.DataFrame([diagnostic(p,a,k) for p,a,k in zip(df.prompt,df.response,df.task_kind)])
    df=pd.concat([df,feats],axis=1)

    mc=df[df.task_kind=='multiple_choice'][['sample_id','model_order','answer_choice']]
    cnt=mc.dropna(subset=['answer_choice']).groupby(['sample_id','answer_choice']).size().to_dict()
    n_models=len(response_columns)
    df['agreement_other_models']=np.nan
    m=df.task_kind.eq('multiple_choice') & df.answer_choice.notna()
    df.loc[m,'agreement_other_models']=[(cnt.get((i,a),0)-1)/(n_models-1)
                                      for i,a in zip(df.loc[m,'sample_id'],df.loc[m,'answer_choice'])]

    choices=mc.set_index(['sample_id','model_order']).answer_choice.to_dict()
    df['peer_agree']=np.nan;df['peer_valid']=np.nan
    mc_mask=df.task_kind.eq('multiple_choice')
    aa=df.loc[mc_mask,'answer_choice'].tolist()
    pp=[choices.get((sid,(int(ord_)+1)%n_models)) for sid,ord_ in
        zip(df.loc[mc_mask,'sample_id'],df.loc[mc_mask,'model_order'])]
    df.loc[mc_mask,'peer_valid']=[int(x is not None) for x in pp]
    df.loc[mc_mask,'peer_agree']=[float(x==y) if x is not None and y is not None else np.nan
                                   for x,y in zip(aa,pp)]
    df.model_id=df.model_id.astype('category')
    df.task_kind=df.task_kind.astype('category')


    single=['log_response_tokens','log_prompt_chars','response_chars','line_count','repetition',
            'empty','refusal','format_error','hard_flag','choice_valid','bracket_valid',
            'python_syntax_valid','contains_code_fence','contains_list','numeric_answer_extracted']
    configs={
        'P_model_task_baseline':['model_id','task_kind','log_prompt_chars'],
        'S_single_response':['model_id','task_kind']+single,
    }
    ids=df.sample_id.drop_duplicates().to_numpy()
    train_ids,test_ids=train_test_split(ids,test_size=.20,random_state=SEED)
    tr=df[df.sample_id.isin(train_ids)].copy()
    te=df[df.sample_id.isin(test_ids)].copy()
    print('Random held-out: train',len(tr),'test',len(te),flush=True)
    pred=te[['sample_id','eval_name','task_kind','model_id','score','hard_flag','choice_valid']].copy()
    rows=[]
    for key,columns in configs.items():
        p=fit_predict(tr,te,columns);pred[key]=p
        n,r=rank_metric(pred,key)
        rows.append(dict(setting='random_80_20',model=key,**stats(te.score,p),
                         rank_requests=n,within_request_spearman=r))

    per_task=[]
    for kind,g in pred.groupby('task_kind',observed=True):
        for key in configs:
            per_task.append(dict(task_kind=str(kind),model=key,**stats(g.score,g[key])))

    trmc=tr[tr.task_kind=='multiple_choice'].copy()
    temc=te[te.task_kind=='multiple_choice'].copy()
    validity=['model_id','format_error','empty','refusal','hard_flag','repetition']
    models={'V_validity':validity,'VC_two_peers':validity+['peer_agree','peer_valid'],
            'VC_all_peers_offline':validity+['agreement_other_models']}
    choice_rows=[]
    mc_pred=temc[['sample_id','eval_name','model_id','score','cost']].copy()
    for key,cols in models.items():
        p=fit_predict(trmc,temc,cols);mc_pred[key]=p
        n,r=rank_metric(mc_pred,key)
        choice_rows.append(dict(setting='MC_random_80_20',model=key,**stats(temc.score,p),
                                rank_requests=n,within_request_spearman=r,
                                extra_calls=0 if key.startswith('V_') else (1 if 'two' in key else n_models-1)))

    mc_all=df[df.task_kind=='multiple_choice'].copy()
    counts=mc_all.groupby('eval_name').sample_id.nunique()
    eligible=counts[counts>=MIN_MC_REQUESTS].index.tolist()
    holdouts=sorted([x for x in eligible if not x.startswith('mmlu-')])+sorted([x for x in eligible if x.startswith('mmlu-')])
    if MAX_MC_HOLDOUTS is not None:holdouts=holdouts[:MAX_MC_HOLDOUTS]
    cross=[]
    for idx,b in enumerate(holdouts,1):
        train_b=mc_all[mc_all.eval_name!=b]
        test_b=mc_all[mc_all.eval_name==b].copy()
        pframe=test_b[['sample_id','score']].copy()
        for key,cols in models.items():
            p=fit_predict(train_b,test_b,cols)
            pframe[key]=p
            n,r=rank_metric(pframe,key)
            cross.append(dict(heldout=b,model=key,**stats(test_b.score,p),
                              rank_requests=n,within_request_spearman=r))
        print(f'MC holdout {idx}/{len(holdouts)}: {b}',flush=True)
    crossdf=pd.DataFrame(cross)
    crosssum=(crossdf.groupby('model',as_index=False)[['MAE','RMSE','Spearman','low_AUC','within_request_spearman']]
              .mean().rename(columns=lambda x: 'macro_'+x if x not in ['model'] else x)) if len(crossdf) else pd.DataFrame()
    evidence=df.groupby('task_kind',observed=True).agg(
        responses=('score','size'),requests=('sample_id','nunique'),
        hard_flag_rate=('hard_flag','mean'),
        format_error_rate=('format_error','mean'),
        number_extracted_rate=('numeric_answer_extracted','mean'),
        python_syntax_check_rate=('python_syntax_valid','count')).reset_index()

    evidence['meaning']='Diagnosis only; correctness unverified without reference, executable tests or judge labels'
    for name,table in {
        'all_response_features.csv':df[['sample_id','eval_name','task_kind','model_id','score','cost']+single+
                                           ['answer_choice','math_final_number','peer_agree','peer_valid','agreement_other_models',
                                            'reference_correctness_known','has_executable_test_result']],
        'random_ablation_summary.csv':pd.DataFrame(rows),
        'random_by_task.csv':pd.DataFrame(per_task),
        'random_test_predictions.csv':pred,
        'mc_random_comparison.csv':pd.DataFrame(choice_rows),
        'mc_random_predictions.csv':mc_pred,
        'mc_heldout_benchmark.csv':crossdf,
        'mc_heldout_macro_summary.csv':crosssum,
        'task_evidence_coverage.csv':evidence,
    }.items():
        table.to_csv(OUT/name,index=False,encoding='utf-8-sig')
    print('\nRANDOM ALL-TASK BASELINE / POST ESTIMATOR')
    print(pd.DataFrame(rows).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nMC SAME-SPLIT COMPARISON')
    print(pd.DataFrame(choice_rows).to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nMC CROSS-BENCHMARK MACRO')
    print(crosssum.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nTASK COVERAGE')
    print(evidence[['task_kind','requests','responses','hard_flag_rate','number_extracted_rate','python_syntax_check_rate']].to_string(index=False))
    print('\nSaved:',OUT.resolve())
    print('LIMITS: math extraction is NOT math verification; Python AST is NOT a unit test; open responses lack semantic truth assessment.')
    print('LIMITS: all-peer consensus has 10 extra calls; two-peer has 1. Offline score labels are only for training and validation.')

if __name__=='__main__':
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',category=RuntimeWarning)
        main()
