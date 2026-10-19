import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from threadpoolctl import threadpool_limits
from .sources import TASK2, read, sha

MIDS = {'S': 'mistralai/mixtral-8x7b-chat', 'L': 'gpt-4-1106-preview'}


class PreDirectRouter:
    def __init__(self, run=None):
        self.run = Path(run) if run is not None else TASK2 / 'runs/experiment'
        sys.dont_write_bytecode = True
        sys.path.insert(0, str(TASK2))
        from integrated_api import task_of
        self.task_of = task_of
        choice = read(self.run / 'final_model_choice.json')
        self.variant = read(self.run / 'pre_selection.json')['winner']
        if choice['family'] != 'Pre_direct' or self.variant != 'A_Baseline':
            raise ValueError('Only the frozen final Pre_direct/A_Baseline is supported')
        self.fallback = set(read(self.run / 'final_safety_calibration.json')['fallback_tasks'])
        self.parameters = read(self.run / 'selected_parameters.json')['Pre_direct']
        if any(p['parameters']['gamma'] is not None or p['parameters']['k'] != 0
               or p['parameters']['chooser'] != 'L' for p in self.parameters.values()):
            raise ValueError('A changed Post/uncertainty policy cannot use this adapter')
        self.cost = read(self.run / 'cost_predictor.json')
        if not self.cost['normalization_c0'] > 0:
            raise ValueError('Invalid frozen cost normalization')
        self.pre = joblib.load(self.run / 'pre_full_train.joblib')
        self.hashes = {name: sha(self.run / name) for name in [
            'pre_full_train.joblib', 'final_model_choice.json', 'final_safety_calibration.json',
            'pre_selection.json', 'selected_parameters.json', 'cost_predictor.json']}

    def route_frame(self, frame, error_s=0., error_l=0.):

        rows = frame[['request_id', 'prompt', 'eval_name']].copy().reset_index(drop=True)
        if rows.empty or rows.request_id.duplicated().any():
            raise ValueError('Nonempty unique request IDs required')
        if not rows.prompt.map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError('Nonempty prompts required')
        if not rows.eval_name.map(lambda x: isinstance(x, str)).all():
            raise ValueError('String benchmark registry names required')
        result = rows[['request_id', 'eval_name']].copy()
        result['task_type'] = rows.eval_name.map(self.task_of)
        result['prompt_chars'] = rows.prompt.map(len)
        registered = rows.eval_name.isin(self.pre.benchmark_names).to_numpy()
        predicted = {m: np.full(len(rows), np.nan) for m in MIDS}
        with threadpool_limits(limits=2):
            for m, model_id in MIDS.items():
                if registered.any():
                    data = pd.DataFrame({'token_length': result.loc[registered, 'prompt_chars'],
                                         'eval_name': rows.loc[registered, 'eval_name'], 'model_id': model_id})
                    predicted[m][registered] = np.clip(self.pre.models[self.variant].predict(
                        self.pre.basic(data)[['token_length', 'eval_name', 'model_id']]), 0, 1)
        for m, error in [('S', error_s), ('L', error_l)]:
            error = np.broadcast_to(np.asarray(error, float), (len(rows),))
            if not np.isfinite(error).all():
                raise ValueError('Nonfinite prediction error')
            result['pre_' + m] = predicted[m]
            result['perturbed_' + m] = np.clip(predicted[m] + error, 0, 1)
        design = np.column_stack([np.ones(len(rows)), result.prompt_chars / 1000])
        costs = {m: design @ self.cost['coefficients'][m.lower()] for m in MIDS}
        result['cost_delta'] = (costs['L'] - costs['S']) / self.cost['normalization_c0']
        result['score'] = np.nan
        result['threshold'] = np.nan
        result['selected_model'] = 'L'
        result['fallback_reason'] = ''
        for index in range(len(rows)):
            task = result.at[index, 'task_type']
            if not registered[index] or task in self.fallback:
                result.at[index, 'fallback_reason'] = 'unregistered_task_or_safety_calibration_fallback'
                continue
            params = self.parameters[task]['parameters']
            score = (result.at[index, 'perturbed_L'] - result.at[index, 'perturbed_S']
                     - params['lambda'] * result.at[index, 'cost_delta'])
            result.at[index, 'score'] = score
            result.at[index, 'threshold'] = params['tau']
            result.at[index, 'selected_model'] = 'L' if score >= params['tau'] else 'S'
        result['calls_s'] = result.selected_model.eq('S').astype(int)
        result['calls_l'] = result.selected_model.eq('L').astype(int)
        result['unchanged_prediction_rule'] = bool(np.all(np.asarray(error_s) == 0) and np.all(np.asarray(error_l) == 0))
        result['risk_guard_coverage'] = 'not_reassessed_for_scenario_mixture_or_duplicates'
        result['LLM_calls_executed'] = 0
        return result
