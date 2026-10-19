from pathlib import Path
import ast
import re
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from lightgbm import LGBMRegressor

SEED = 42


ROOT = Path(__file__).resolve().parent


PROJECT_ROOT = ROOT.parent.parent


DATA = PROJECT_ROOT / 'AI_datasets/RouterBench/routerbench_0shot.csv'


OUT = PROJECT_ROOT / 'tacva_two_experiments_outputs'
OUT.mkdir(parents=True, exist_ok=True)


print("数据文件路径：", DATA)
print("数据文件是否存在：", DATA.exists())

if not DATA.exists():
    raise FileNotFoundError(f"找不到数据集：{DATA}")


SAMPLE_PER_BENCHMARK = 30
MAX_MC_BENCHMARKS = None
MIN_REQUESTS = 15
LOW_SCORE = 0.25
TRIGGER_SCORE = 0.25



def parse_response(z):
    if pd.isna(z): return ''
    s = str(z)
    if s.startswith('[') and s.endswith(']'):
        try:
            x = ast.literal_eval(s)
            if isinstance(x, (list,tuple)): return '\n'.join(map(str,x))
        except (SyntaxError, ValueError, TypeError, RecursionError, MemoryError): pass
    return s

def option(z):
    z = z.strip().strip('[](){}"\'` ').upper()
    v = re.fullmatch(r'(?:OPTION\s*|ANSWER\s*[:：]?\s*)?([A-D])(?:[.。])?', z)
    return v.group(1) if v else None

def meta(z):
    t = re.findall(r'[\u3400-\u9fff]|[a-z0-9_]+',z.lower())
    g = list(zip(t,t[1:],t[2:]))
    rep = 1-len(set(g))/len(g) if len(g)>=10 else 0
    refusal = int(bool(re.search(r"\b(i (?:cannot|can't|don't know)|unable to|as an ai)\b|无法回答|不知道|不能回答",z,re.I)))
    empty = int(not z.strip())
    fmt = int(option(z) is None)
    high_rep = int(rep>.65)
    return dict(format_invalid=fmt,empty=empty,refusal=refusal,high_repetition=high_rep,
                hard_flag=int(bool(fmt or empty or refusal or high_rep)))

def metrics(y,p):
    y=np.asarray(y,dtype=float);p=np.clip(np.asarray(p,dtype=float),0,1)
    low=(y<=LOW_SCORE).astype(int)
    return dict(n=len(y), MAE=mean_absolute_error(y,p),RMSE=np.sqrt(mean_squared_error(y,p)),
        Spearman=float(spearmanr(y,p).statistic) if len(np.unique(y))>1 and len(np.unique(p))>1 else np.nan,
        low_AUC=roc_auc_score(low,-p) if len(np.unique(low))==2 else np.nan)

def within_request(df,col):
    r=[]
    for _,g in df.groupby('sample_id',sort=False):
        if g.score.nunique()>1 and g[col].nunique()>1:
            v=spearmanr(g.score,g[col]).statistic
            if np.isfinite(v):r.append(v)
    return len(r),float(np.mean(r)) if r else np.nan

def train_reg(tr,fields):
    model=LGBMRegressor(n_estimators=250,learning_rate=.05,max_depth=5,random_state=SEED,verbosity=-1,n_jobs=-1)
    model.fit(tr[fields],tr.score)
    return model

VALID=['model_id','format_invalid','empty','refusal','high_repetition','hard_flag']
TWO=VALID+['peer_agree','peer_valid']
FULL=VALID+['agreement_other_10','full_valid_peer_frac']


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    wide=pd.read_csv(DATA,low_memory=False)

    names=wide.eval_name.astype(str)
    mask=names.str.startswith('mmlu-') | names.isin(['arc-challenge','hellaswag','winogrande'])
    wide=wide.loc[mask].copy()
    if SAMPLE_PER_BENCHMARK is not None:
        wide=pd.concat([x.sample(min(len(x),SAMPLE_PER_BENCHMARK),random_state=SEED)
                        for _,x in wide.groupby('eval_name',sort=True)],ignore_index=True)
    ids=[c[:-len('|model_response')] for c in wide.columns if c.endswith('|model_response')]
    parts=[]
    for j,mid in enumerate(ids):
        cost_col=mid+'|total_cost'
        t=wide[['sample_id','eval_name',mid,mid+'|model_response',cost_col]].copy()
        t.columns=['sample_id','eval_name','score','response','cost']
        t['model_id']=mid;t['model_order']=j
        parts.append(t)
    df=pd.concat(parts,ignore_index=True)
    df.score=pd.to_numeric(df.score,errors='coerce')
    df.cost=pd.to_numeric(df.cost,errors='coerce')
    df.response=df.response.map(parse_response)
    df=df.dropna(subset=['score']).reset_index(drop=True)
    assert df.groupby('sample_id')['model_id'].nunique().eq(len(ids)).all(), 'Missing response score: cannot form full candidate set'
    df['answer_choice']=df.response.map(option)
    df=pd.concat([df,pd.DataFrame([meta(x) for x in df.response])],axis=1)
    df.model_id=df.model_id.astype('category')
    M=len(ids)
    look=df.set_index(['sample_id','model_order'])['answer_choice'].to_dict()
    df['peer_order']=(df.model_order+1)%M
    df['peer_choice']=[look.get((i,j)) for i,j in zip(df.sample_id,df.peer_order)]
    df['peer_agree']=[float(a==b) if a is not None and b is not None else np.nan
                      for a,b in zip(df.answer_choice,df.peer_choice)]
    df['peer_valid']=df.peer_choice.notna().astype(int)
    counts=df.dropna(subset=['answer_choice']).groupby(['sample_id','answer_choice']).size().to_dict()
    df['agreement_other_10']=[(counts.get((i,a),0)-1)/(M-1) if a is not None else np.nan
                              for i,a in zip(df.sample_id,df.answer_choice)]
    df['full_valid_peer_frac']=(df.groupby('sample_id')['answer_choice'].transform(lambda s:s.notna().sum())
                                -df.answer_choice.notna().astype(int))/(M-1)
    counts=df.groupby('eval_name').sample_id.nunique()
    benchmarks=counts[counts>=MIN_REQUESTS].index.tolist()

    non_mmlu=[b for b in benchmarks if not b.startswith('mmlu-')]
    mmlu=sorted([b for b in benchmarks if b.startswith('mmlu-')])
    ordered=non_mmlu + mmlu
    if MAX_MC_BENCHMARKS is not None: ordered=ordered[:MAX_MC_BENCHMARKS]
    benchmarks=ordered
    if not benchmarks: raise RuntimeError('No benchmark eligible: lower MIN_REQUESTS')
    print(f'MC requests={df.sample_id.nunique()} answers={len(df)}; folds={len(benchmarks)}',flush=True)
    print('Held-out benchmarks:',benchmarks,flush=True)
    score_rows=[];all_pred=[];policy_rows=[];policy_actions=[]
    for fi,b in enumerate(benchmarks,1):
        tr=df[df.eval_name!=b].copy();te=df[df.eval_name==b].copy()

        print(f'[{fi}/{len(benchmarks)}] {b}: train {len(tr)}; test {len(te)}',flush=True)
        fitted={k:train_reg(tr,v) for k,v in [('V',VALID),('VC_2',TWO),('VC_11',FULL)]}
        pp=te[['sample_id','eval_name','model_id','model_order','score','cost','answer_choice','hard_flag']].copy()
        for k,cols in [('V',VALID),('VC_2',TWO),('VC_11',FULL)]:
            pp[k]=np.clip(fitted[k].predict(te[cols]),0,1)
            n_rank,rho=within_request(pp,k)
            score_rows.append(dict(heldout=b,model=k,**metrics(pp.score,pp[k]),
                                   n_rank_requests=n_rank,within_request_spearman=rho))
        all_pred.append(pp)


        cost_av=tr.groupby('model_id',observed=True).cost.mean().sort_values()
        if len(cost_av)<2:continue
        first_id,second_id=cost_av.index[:2].tolist()
        a=te[te.model_id.eq(first_id)].copy()
        z=te[te.model_id.eq(second_id)].copy()
        assert a.sample_id.is_unique and z.sample_id.is_unique
        a=a.set_index('sample_id',drop=False).sort_index()
        z=z.set_index('sample_id',drop=False).reindex(a.index)


        aa=a.copy(); zz=z.copy()
        for cur,peer in [(aa,zz),(zz,aa)]:
            cur['peer_valid']=peer.answer_choice.notna().astype(int).values
            cur['peer_agree']=np.where(cur.answer_choice.notna() & peer.answer_choice.notna(),
                                       (cur.answer_choice==peer.answer_choice).astype(float),np.nan)
        s0=np.clip(fitted['V'].predict(aa[VALID]),0,1)
        score_a=np.clip(fitted['VC_2'].predict(aa[TWO]),0,1)
        score_z=np.clip(fitted['VC_2'].predict(zz[TWO]),0,1)

        trigger=(aa.hard_flag.to_numpy().astype(bool)) | (s0 < TRIGGER_SCORE)
        best_second=(score_z>score_a)
        always_choose=best_second
        cond_choose=trigger & best_second
        for policy,use_second,chosen_second in [
            ('first_only',np.zeros(len(aa),bool),np.zeros(len(aa),bool)),
            ('always_two',np.ones(len(aa),bool),always_choose),
            ('conditional_two',trigger,cond_choose)
        ]:
            y=np.where(chosen_second,zz.score,aa.score).astype(float)
            initial_cost=aa.cost.to_numpy(float)
            add_cost=np.where(use_second,zz.cost.to_numpy(float),0.)
            policy_rows.append(dict(heldout=b,policy=policy,first_model=first_id,peer_model=second_id,
                n_requests=len(y),mean_actual_score=np.mean(y),low_rate=np.mean(y<=LOW_SCORE),
                mean_total_cost=np.nanmean(initial_cost+add_cost),mean_extra_cost=np.nanmean(add_cost),
                second_call_rate=np.mean(use_second),changed_answer_rate=np.mean(chosen_second),
                mean_score_gain_over_first=np.mean(y-aa.score.to_numpy(float))))
            policy_actions.extend([dict(heldout=b,sample_id=i,policy=policy,score=float(s),
                   cost=float(c),extra_called=int(ec),switched=int(sw))
                   for i,s,c,ec,sw in zip(aa.index,y,initial_cost+add_cost,use_second,chosen_second)])
    comparisons=pd.DataFrame(score_rows)
    preds=pd.concat(all_pred,ignore_index=True)
    policies=pd.DataFrame(policy_rows)
    actions=pd.DataFrame(policy_actions)
    macro=(comparisons.groupby('model')[['MAE','RMSE','Spearman','low_AUC','within_request_spearman']].mean()
           .add_prefix('macro_').reset_index())
    pooled=[]
    for k in ['V','VC_2','VC_11']:
        res=metrics(preds.score,preds[k])
        pooled.append(dict(model=k,**{f'pooled_{x}':v for x,v in res.items()}))
    summ=macro.merge(pd.DataFrame(pooled),on='model')
    pol_summ=policies.groupby('policy').apply(lambda g:pd.Series({
        'benchmarks':len(g),'n_requests':g.n_requests.sum(),
        'macro_mean_score':g.mean_actual_score.mean(),
        'pooled_mean_score':np.average(g.mean_actual_score,weights=g.n_requests),
        'pooled_low_rate':np.average(g.low_rate,weights=g.n_requests),
        'pooled_total_cost':np.average(g.mean_total_cost,weights=g.n_requests),
        'pooled_extra_cost':np.average(g.mean_extra_cost,weights=g.n_requests),
        'second_call_rate':np.average(g.second_call_rate,weights=g.n_requests),
        'switch_rate':np.average(g.changed_answer_rate,weights=g.n_requests)
    }),include_groups=False).reset_index()
    for fname,obj in [('generalization_by_benchmark.csv',comparisons),('generalization_summary.csv',summ),
                      ('generalization_predictions.csv',preds),('policy_by_benchmark.csv',policies),
                      ('policy_summary.csv',pol_summ),('policy_decisions.csv',actions)]:
        obj.to_csv(OUT/fname,index=False,encoding='utf-8-sig')
    print('\nGENERALIZATION (MC):\n',summ.to_string(index=False,float_format=lambda x:f'{x:.4f}'))
    print('\nCOST/QUALITY ROUTING SIMULATION:\n',pol_summ.to_string(index=False,float_format=lambda x:f'{x:.6f}'))
    print('\nNOTE: fixed cheapest/second-cheapest peers by train mean cost; proxy offline policy, not optimized router.')
    print('NOTE: no real latency or CO2 verification; only MC benchmarks; score is benchmark-specific.')
    print('Saved to',OUT.resolve())

if __name__=='__main__': main()
