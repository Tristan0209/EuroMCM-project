import numpy as np
import pandas as pd


def quality_harms(records, paired_quality):
    frame = records.merge(paired_quality[['request_id', 'q_s', 'q_l']], on='request_id', validate='one_to_one')
    if len(frame) != len(records) or frame[['q_s', 'q_l']].isna().any().any():
        raise ValueError('Complete evaluation-only S/L quality required')
    if not np.isfinite(frame[['q_s', 'q_l']].to_numpy()).all() or not frame[['q_s', 'q_l']].apply(lambda c: c.between(0, 1)).all().all():
        raise ValueError('Quality proxy outside [0,1]')
    selected = np.where(frame.selected_model.eq('L'), frame.q_l, frame.q_s)
    valid = frame.status.eq('completed') & ~frame.service_constraint_violation.astype(bool)
    frame['model_selection_signed_loss'] = frame.q_l - selected
    frame['model_selection_harm'] = np.maximum(frame.q_l - selected, 0.)
    frame['delivery_quality_lost'] = (1 - valid.astype(int)) * selected
    frame['end_to_end_signed_loss'] = frame.q_l - valid.astype(int) * selected
    frame['end_to_end_harm'] = np.maximum(frame.end_to_end_signed_loss, 0.)
    frame['delivery_incremental_harm'] = frame.end_to_end_harm - frame.model_selection_harm
    frame['model_pass_harm'] = (frame.q_l >= .5) & (selected < .5)
    frame['end_to_end_pass_harm'] = (frame.q_l >= .5) & (~valid | (selected < .5))
    frame['delivery_pass_increment'] = frame.end_to_end_pass_harm.astype(int) - frame.model_pass_harm.astype(int)
    return frame


def fairness(frame, group_columns, class_weights=None):
    if not group_columns or frame.empty:
        raise ValueError('Nonempty explicit groups required')
    rows = []
    for key, part in frame.groupby(group_columns, dropna=False, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        completed = part.status.eq('completed')
        response = part.loc[completed, 'response_s'].dropna() if 'response_s' in part else pd.Series(dtype=float)
        row = {**dict(zip(group_columns, key)), 'n': len(part),
               'unique_prompt_groups': part['group'].nunique(),
               'completed': int(completed.sum()), 'failed': int(part.status.eq('failed').sum()),
               'censored': int(part.status.eq('censored_at_H').sum()),
               'violation_rate': float(part.service_constraint_violation.mean()),
               'completion_rate': float(completed.mean()),
               'completed_p95_s': float(response.quantile(.95)) if len(response) else None,
               'model_selection_harm': float(part.model_selection_harm.mean()),
               'delivery_incremental_harm': float(part.delivery_incremental_harm.mean()),
               'end_to_end_harm': float(part.end_to_end_harm.mean())}
        waits = part.get('intentional_delay_s', pd.Series(np.nan, index=part.index)) + part.get('queue_wait_s', pd.Series(np.nan, index=part.index))
        denominator = part.max_wait_s
        ratio = waits.where(denominator > 0) / denominator.where(denominator > 0)
        row['completed_mean_wait_budget_ratio'] = float(ratio[completed].mean()) if ratio[completed].notna().any() else None
        row['zero_wait_budget_count'] = int(denominator.eq(0).sum())
        if class_weights is not None:
            if any(w < 0 for w in class_weights.values()) or not np.isclose(sum(class_weights.values()), 1):
                raise ValueError('Invalid standardization weights')
            rates = part.groupby('service_class').service_constraint_violation.mean()
            missing = [c for c, w in class_weights.items() if w > 0 and c not in rates.index]
            row['standardized_violation_rate'] = None if missing else float(sum(w * rates[c] for c, w in class_weights.items() if w))
            row['missing_standardization_classes'] = ','.join(missing)
        rows.append(row)
    return pd.DataFrame(rows)


def failure_decomposition(sim, records, quality_frame=None):





    result = records[['request_id', 'status', 'failure_reason', 'fallback_reason']].copy()
    result['upstream_model_selection_harm'] = False
    result['upstream_quality_assessed'] = quality_frame is not None
    if quality_frame is not None:
        lookup = quality_frame.set_index('request_id').model_selection_harm
        result['upstream_model_selection_harm'] = result.request_id.map(lookup).gt(0)
    result['server_capacity_exposure'] = False
    result['queue_expiration'] = result.failure_reason.str.contains('wait|queue', regex=True)
    result['milp_numerical_rejection'] = False
    result['milp_solver_timeout'] = False
    result['fallback_used'] = result.fallback_reason.ne('')
    started = records.get('service_start_s', pd.Series(np.nan, index=records.index)).notna()
    completion = records.get('completion_s', pd.Series(np.nan, index=records.index))
    result['execution_timeout'] = (started & ((completion > records.deadline_s + 1e-6)
                                  | (records.status.eq('censored_at_H') & (sim.end > records.deadline_s + 1e-6)))).to_numpy()
    result['window_censoring'] = result.status.eq('censored_at_H')
    for log in sim.solver_log:
        numerical = log.get('status') == 4 and log.get('recovery', {}).get('unrecoverable', False)
        mask = result.request_id.isin(log.get('request_ids', []))
        if numerical:
            result.loc[mask, 'milp_numerical_rejection'] = True
        if log.get('status') == 1:
            result.loc[mask, 'milp_solver_timeout'] = True
    for idx, rec in records.iterrows():
        if pd.isna(rec.get('region')):
            continue
        start = rec.get('ready_s', rec.arrival_s)
        stop = rec.get('service_start_s', np.nan)
        stop = rec.terminal_s if pd.isna(stop) else stop
        events = sorted((t, c) for t, r, m, c in sim.availability.events
                        if r == rec.region and m == rec.selected_model and t <= sim.end)
        for j, (time, cap) in enumerate(events):
            end = events[j+1][0] if j+1 < len(events) else sim.end
            if cap == 0 and max(start, time) < min(stop, end):
                result.loc[result.request_id.eq(rec.request_id), 'server_capacity_exposure'] = True
    primary = []
    violation_by_id = records.set_index('request_id').service_constraint_violation
    for rec in result.itertuples():
        if rec.window_censoring:
            cause = 'window_censoring'
        elif rec.execution_timeout:
            cause = 'execution_timeout'
        elif rec.status == 'completed' and not bool(violation_by_id.loc[rec.request_id]):
            cause = 'valid_completion'
        elif rec.queue_expiration:
            cause = 'queue_or_predicted_limit'
        elif rec.server_capacity_exposure:
            cause = 'observed_capacity_exposure'
        else:
            cause = 'other_service_failure'
        primary.append(cause)
    result['primary_service_outcome'] = primary
    return result
