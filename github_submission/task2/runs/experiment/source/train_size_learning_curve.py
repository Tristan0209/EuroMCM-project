import argparse, io, json, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SMALL='S'; LARGE='L'; LAMBDA=.5; TAU=-.6170992537; C0=.00325439052

def load(path):
    with zipfile.ZipFile(path) as z:
        req=pd.read_csv(z.open('inputs/paired/request_splits.csv'))
        outcomes=[]
        for raw in z.open('inputs/paired/outcomes.jsonl'):
            r=json.loads(raw)
            if r['action_id'] in ('S','L'):
                outcomes.append((r['request_id'],r['action_id'],r['quality_true'],r['cost_usd']))
    o=pd.DataFrame(outcomes,columns=['request_id','arm','score','cost'])
    if o.duplicated(['request_id','arm']).any():raise ValueError('duplicate outcomes')
    wide=o.pivot(index='request_id',columns='arm',values=['score','cost'])
    wide.columns=[f'{k.lower()}_{a.lower()}' for k,a in wide.columns]
    d=req.join(wide,on='request_id')
    assert len(d)==36417 and d[['score_s','score_l','cost_s','cost_l']].notna().all().all()
    assert d.groupby('prompt_group').split.nunique().max()==1
    u=d.prompt_chars.to_numpy()/1000
    d['dc_norm']=((.00007346462+.0041631118*u)-(9.4887300e-6+.00016285148*u))/C0
    return d

def fit_model(train,seed):


    long=pd.concat([train.assign(arm='S',label=train.score_s),train.assign(arm='L',label=train.score_l)],ignore_index=True)
    x=long[['prompt_chars','eval_name','arm']].copy()
    for c in ['eval_name','arm']:x[c]=x[c].astype('category')
    y=long.label.to_numpy()
    model=LGBMRegressor(n_estimators=200,learning_rate=.05,max_depth=5,random_state=42,
                       n_jobs=4,verbosity=-1)
    model.fit(x,y,categorical_feature=['eval_name','arm'])
    return model

def predict(model,data):
    out=data.copy()
    for arm in 'SL':
        x=out[['prompt_chars','eval_name']].copy();x['arm']=arm
        for c in ['eval_name','arm']:x[c]=x[c].astype('category')
        out['pred_'+arm.lower()]=np.clip(model.predict(x),0,1)
    return out

def feat(row):
    return np.array([1.,np.clip(row['prompt_chars']/2000,0,2),row['pred_s'],row['pred_l']])

class Calibrator:
    def __init__(self,ridge,alpha):
        self.A={a:np.eye(4)*ridge for a in 'SL'}
        self.b={a:np.zeros(4) for a in 'SL'}
        self.alpha=alpha
    def adjust(self,row,a):
        x=feat(row);v=np.linalg.solve(self.A[a],x)
        mu=x@np.linalg.solve(self.A[a],self.b[a])
        return mu+self.alpha*np.sqrt(max(0.,x@v))
    def update(self,row,a,score):
        x=feat(row)
        self.A[a]+=np.outer(x,x)
        self.b[a]+=x*(score-row['pred_'+a.lower()])

def sequence_by_group(test,seed):
    rng=np.random.default_rng(seed)
    groups=list(test.groupby('prompt_group',sort=False).groups.values())
    idx=np.concatenate([np.asarray(list(groups[i]),dtype=int) for i in rng.permutation(len(groups))])
    return test.loc[idx].reset_index(drop=True)

def evaluate(stream,method,seed,audit_prob,audit_usd,alpha,ridge,shift_drop,shift_at,checkpoints,ratio):
    rng=np.random.default_rng(seed+99117)
    cal=Calibrator(ridge,alpha) if method=='online' else None
    n=len(stream);cum_score=cum_l_score=cum_pass=cum_l_pass=cum_cost=cum_l_cost=cum_aud=small=general=small_general=0.
    result=[];cp=set(checkpoints)
    for t,row in enumerate(stream.itertuples(index=False),1):


        x=row._asdict(); is_shift=(shift_drop>0 and t>int(n*shift_at) and x['task_type']=='General')
        q_s=max(0.,x['score_s']-shift_drop) if is_shift else x['score_s']
        q_l=x['score_l']
        if x['task_type']!='General':a='L'
        else:
            h=x['pred_l']-x['pred_s']-LAMBDA*x['dc_norm']
            if cal is not None:h+=cal.adjust(x,'L')-cal.adjust(x,'S')
            a='L' if h>=TAU else 'S'
        observed=q_l if a=='L' else q_s
        audited=method=='online' and rng.random()<audit_prob
        if audited:cal.update(x,a,observed)
        cost=x['cost_'+a.lower()]+(audit_usd if audited else 0.)
        cum_score+=observed;cum_l_score+=q_l
        cum_pass+=float(observed>=.5);cum_l_pass+=float(q_l>=.5)
        cum_cost+=cost;cum_l_cost+=x['cost_l'];cum_aud+=int(audited)
        small+=a=='S';general+=x['task_type']=='General';small_general+=int(a=='S' and x['task_type']=='General')
        if t in cp:
            result.append(dict(train_ratio=ratio,method=method,seed=seed,requests=t,
              cumulative_score=cum_score/t,score_loss_vs_L=(cum_l_score-cum_score)/t,
              pass_rate=cum_pass/t,pass_loss_vs_L=(cum_l_pass-cum_pass)/t,
              cumulative_net_saving=1-cum_cost/cum_l_cost,cumulative_cost_usd=cum_cost,
              feedback_count=int(cum_aud),small_fraction=small/t,
              small_fraction_general=small_general/max(1,general),
              shifted=int(shift_drop>0),shift_drop=shift_drop))
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--zip',type=Path,required=True)
    p.add_argument('--out',type=Path,default=Path('train_size_results'))
    p.add_argument('--fractions',type=float,nargs='+',default=[.2,.5,1.0])
    p.add_argument('--seeds',type=int,default=3)
    p.add_argument('--audit-prob',type=float,default=.10)
    p.add_argument('--audit-usd',type=float,default=.0001)
    p.add_argument('--alpha',type=float,default=.10)
    p.add_argument('--ridge',type=float,default=20.)
    p.add_argument('--shift-drop',type=float,default=0.,help='Synthetic S score drop AFTER shift-at in General, 0=no shift')
    p.add_argument('--shift-at',type=float,default=.5)
    args=p.parse_args()
    if any(not 0<f<=1 for f in args.fractions) or args.seeds<1 or not 0<=args.audit_prob<=1 or args.audit_usd<0 or args.alpha<0 or args.ridge<=0 or not 0<=args.shift_drop<=1 or not 0<args.shift_at<1:
        p.error('Invalid parameters')
    args.out.mkdir(parents=True,exist_ok=True)
    data=load(args.zip);train=data[data.split=='train'].copy();test=data[data.split=='test'].copy()
    checkpoints=sorted(set(np.linspace(250,len(test),25,dtype=int).tolist()+[len(test)]))
    results=[];train_counts=[]

    groups=np.sort(train.prompt_group.unique())
    rng=np.random.default_rng(20261017)
    groups=rng.permutation(groups)
    for frac in sorted(set(args.fractions)):
        selected=set(groups[:int(np.ceil(len(groups)*frac))])
        subset=train[train.prompt_group.isin(selected)]
        print(f'Train fraction={frac:.2f}: {len(subset)} requests, {len(selected)} prompt groups',flush=True)
        predictor=fit_model(subset,42)
        prepared=predict(predictor,test).reset_index(drop=True)
        train_counts.append(dict(train_ratio=frac,train_requests=len(subset),train_prompt_groups=len(selected),test_requests=len(test)))
        for k in range(args.seeds):
            seed=20261017+k;stream=sequence_by_group(prepared,seed)
            for method in ('static','online'):
                results.extend(evaluate(stream,method,seed,args.audit_prob,args.audit_usd,args.alpha,args.ridge,
                                        args.shift_drop,args.shift_at,checkpoints,frac))
    frame=pd.DataFrame(results)
    frame.to_csv(args.out/'learning_curves_all_seeds.csv',index=False)
    pd.DataFrame(train_counts).to_csv(args.out/'training_sizes.csv',index=False)
    metrics=['cumulative_score','score_loss_vs_L','pass_rate','pass_loss_vs_L','cumulative_net_saving','feedback_count','small_fraction_general']
    avg=frame.groupby(['train_ratio','method','requests'],as_index=False)[metrics].mean()
    avg.to_csv(args.out/'learning_curves_mean.csv',index=False)
    final=frame[frame.requests.eq(len(test))].groupby(['train_ratio','method'],as_index=False)[metrics].agg(['mean','std'])
    final.to_csv(args.out/'final_summary.csv')
    for metric,label in [('score_loss_vs_L','Cumulative score loss vs all L'),('cumulative_net_saving','Net cost saving'),('feedback_count','Cumulative paid feedback'),('small_fraction_general','General S selection fraction')]:
        fig,ax=plt.subplots(figsize=(9,5),dpi=150)
        for (ratio,method),g in avg.groupby(['train_ratio','method']):
            g=g.sort_values('requests');ax.plot(g.requests,g[metric],label=f'{int(ratio*100)}% train / {method}')
        ax.set(xlabel='Processed requests (simulated order)',ylabel=label,title=label)
        ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.25);fig.tight_layout()
        fig.savefig(args.out/f'{metric}.png');plt.close(fig)
    report={'experimental_design':'Training fractions sampled by entire prompt_group ONLY from existing train split; select/confirm untouched; test fixed.',
      'model':'New TWO-action LightGBM trained on S/L only. Results are NOT original 11-model trained Pre-LightGBM numbers.',
      'baseline':'Fixed team cost penalty, General threshold, Code/Math -> L; no threshold recalibration.',
      'online':'Per-action ridge residual update + optional UCB, chosen-arm audited feedback only.',
      'feedback_assumption':{'probability':args.audit_prob,'usd_per_audit':args.audit_usd},
      'shift':{'synthetic':args.shift_drop>0,'drop_S_General':args.shift_drop,'after_fraction':args.shift_at},
      'important_limitations':['Repeated permutations of the SAME historical test data, not 100k independent observations.',
        'Training-fraction tests and cumulative sample-count tests answer different questions.',
        'Old test has been inspected repeatedly, no new untouched test set.',
        'Feedback quality assumed observable for audited chosen action only; no counterfactual training.',
        'Online model has no automatically calibrated safety guarantee.'], 'seed_count':args.seeds}
    (args.out/'experiment_design.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\nFinal summary:');print(frame[frame.requests.eq(len(test))].groupby(['train_ratio','method'])[metrics].mean().round(5).to_string())
    print('Saved:',args.out.resolve())
if __name__=='__main__':main()
