import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from q1_adapter import BASE, VALID, TWO, load_post_source
from routing import NEURAL, SEED
import neural_adapter

HERE=Path(__file__).resolve().parent
MIDS={'S':'mistralai/mixtral-8x7b-chat','L':'gpt-4-1106-preview'}


def task_of(name):
    name=name.lower()
    if name in {'mbpp','humaneval'}: return 'Code'
    if any(s in name for s in ['math','algebra','statistics','remainder-theorem']):return 'Math'
    return 'General'


class IntegratedRouter:
    def __init__(self,out):
        self.out=Path(out)
        choice='final_model_choice.json' if (self.out/'final_model_choice.json').exists() else 'model_choice.json'
        guard='final_safety_calibration.json' if (self.out/'final_safety_calibration.json').exists() else 'safety_calibration.json'
        self.family=json.loads((self.out/choice).read_text(encoding='utf-8'))['family']
        self.parameters=json.loads((self.out/'selected_parameters.json').read_text(encoding='utf-8'))[self.family]
        self.fallback=json.loads((self.out/guard).read_text(encoding='utf-8'))['fallback_tasks']
        self.pre=joblib.load(self.out/'pre_full_train.joblib')
        self.pre_variant=json.loads((self.out/'pre_selection.json').read_text(encoding='utf-8'))['winner']
        self._pre_scaler=None
        self.cost=json.loads((self.out/'cost_predictor.json').read_text(encoding='utf-8'))
        self._post=None
        self.post_functions=load_post_source(self.out/'work1_source')
        self.feature_encoder=None;self.model=None

    @property
    def pre_scaler(self):
        if self._pre_scaler is None:
            self._pre_scaler=joblib.load(self.out/'pre_feature_scaler.joblib')
        return self._pre_scaler

    @property
    def post(self):
        if self._post is None:
            self._post=joblib.load(self.out/'post_models.joblib')
        return self._post

    def route(self,prompt,benchmark,seed=SEED):
        if not isinstance(prompt,str) or not prompt.strip():raise ValueError('A nonempty prompt is required')
        task=task_of(benchmark)
        if benchmark not in self.pre.benchmark_names or task in self.fallback:
            return {'initial_action':'L','task_type':task,'reason':'unregistered_task_or_safety_calibration_fallback',
                    'LLM_calls_executed':0,'absolute_quality_guarantee':False}
        rows=pd.DataFrame([{'token_length':len(prompt),'eval_name':benchmark,'model_id':MIDS[a]} for a in ['S','L']])
        with threadpool_limits(limits=2):
            if self.pre_variant=='A_Baseline':

                quality=np.clip(self.pre.models[self.pre_variant].predict(self.pre.basic(rows)[BASE]),0,1)
            else:quality=self.pre.predict(rows,self.pre_variant)
            std=0.
            if self.family=='Pre_direct':mu=score=float(quality[1]-quality[0])
            else:
                if self.feature_encoder is None:
                    import features
                    audit=json.loads((self.out/'q1_source_audit.json').read_text(encoding='utf-8'))
                    project=Path(audit['raw_source']).parents[2]
                    self.feature_encoder=joblib.load(HERE/'support/feature_encoder.joblib')
                text=pd.DataFrame([{'prompt':prompt,'prompt_chars':len(prompt)}])
                base=self.feature_encoder.transform(text)
                extra=self.pre_scaler.transform(pd.DataFrame([{'pre_s':quality[0],'pre_l':quality[1],
                                                               'pre_delta':quality[1]-quality[0]}]))
                x=np.column_stack([base,extra]).astype(np.float32)
                if self.family in ['LinUCB','Thompson_Sampling']:
                    if self.model is None:self.model=joblib.load(self.out/'routers/linear_arms.joblib')[0]
                    means,_=self.model.moments(x);mu=float(np.clip(means[0,1]-means[0,0],-1,1))
                    kind='LinUCB' if self.family=='LinUCB' else 'Thompson Sampling'
                    score=float(self.model.score(x,kind,np.random.default_rng(seed))[0])
                else:
                    if self.model is None:self.model=joblib.load(self.out/'routers'/f'{self.family}.joblib')[0]
                    if self.family=='MLP':
                        mean,spread=self.model.predict(x);mu=float(mean[0]);std=float(spread[0])
                    else:
                        mu=float(np.clip(self.model.predict(x)[0],-1,1))
                        if self.family=='Random_Forest':std=float(np.std([t.predict(x)[0] for t in self.model.estimators_]))
                    score=mu
        p=self.parameters[task]['parameters']
        costs={a:float(np.array([1,len(prompt)/1000])@np.array(self.cost['coefficients'][a.lower()])) for a in ['S','L']}
        h=score+p['k']*std-p['lambda']*(costs['L']-costs['S'])/self.cost['normalization_c0']
        return {'initial_action':'L' if h>=p['tau'] else 'S','task_type':task,'family':self.family,
            'pre_q_s':float(quality[0]),'pre_q_l':float(quality[1]),'predicted_gain':mu,
            'uncalibrated_disagreement':std,'decision_score':h,'threshold':p['tau'],
            'post_threshold':p['gamma'],'peer_chooser':p['chooser'],'cost_hat_usd':costs,
            'LLM_calls_executed':0,'quality_is_benchmark_score_proxy':True,'absolute_quality_guarantee':False}

    def diagnostics(self,prompt,response,benchmark):
        fn=self.post_functions
        return fn['diagnostic'](prompt,fn['norm_answer'](response),fn['kind_of'](benchmark))

    def check_small(self,prompt,response,benchmark):
        task=task_of(benchmark);d=self.diagnostics(prompt,response,benchmark)
        if task in self.fallback or benchmark not in self.pre.benchmark_names:
            return {'upgrade':True,'reason':'safety_fallback','correctness_verified':False}
        p=self.parameters[task]['parameters'];mc=self.post_functions['kind_of'](benchmark)=='multiple_choice'
        if p['gamma'] is None:return {'upgrade':False,'post_enabled':False,'hard_flag':bool(d['hard_flag']),
                                    'correctness_verified':False}
        proxy=None
        if mc:
            row=pd.DataFrame([{**d,'model_id':MIDS['S']}]);row['model_id']=pd.Categorical(row.model_id,categories=self.post['model_names'])
            proxy=float(np.clip(self.post['models']['V'].predict(row[VALID])[0],0,1))
        return {'upgrade':bool(proxy<p['gamma']) if mc else bool(d['hard_flag']),
                'post_enabled':True,'quality_proxy':proxy,'hard_flag':bool(d['hard_flag']),
                'correctness_verified':False,'quality_is_benchmark_score_proxy':True}

    def choose_after_upgrade(self,prompt,small_response,large_response,benchmark):
        if small_response is None or large_response is None:raise ValueError('Both generated responses are required; a peer is not free')
        if benchmark not in self.pre.benchmark_names or task_of(benchmark) in self.fallback:
            return {'final_action':'L','model_calls_required':2,'method':'safety_fallback','correctness_verified':False}
        task=task_of(benchmark);p=self.parameters[task]['parameters']
        if p['chooser']!='VC2' or self.post_functions['kind_of'](benchmark)!='multiple_choice':
            return {'final_action':'L','model_calls_required':2,'method':'L_after_upgrade','correctness_verified':False}
        diagnostics={a:self.diagnostics(prompt,r,benchmark) for a,r in [('S',small_response),('L',large_response)]}
        rows=[]
        for a in ['S','L']:
            peer=diagnostics['L' if a=='S' else 'S']['answer_choice'];own=diagnostics[a]['answer_choice']
            rows.append({**diagnostics[a],'model_id':MIDS[a],'peer_valid':int(peer is not None),
                         'peer_agree':float(own==peer) if own is not None and peer is not None else np.nan})
        data=pd.DataFrame(rows);data['model_id']=pd.Categorical(data.model_id,categories=self.post['model_names'])
        values=np.clip(self.post['models']['VC2'].predict(data[TWO]),0,1)
        return {'final_action':'L' if values[1]>=values[0] else 'S','model_calls_required':2,'method':'VC2',
                'quality_proxy_s':float(values[0]),'quality_proxy_l':float(values[1]),'correctness_verified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'runs/experiment')
    parser.add_argument('--prompt',required=True);parser.add_argument('--benchmark',required=True)
    args=parser.parse_args();print(json.dumps(IntegratedRouter(args.out).route(args.prompt,args.benchmark),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
