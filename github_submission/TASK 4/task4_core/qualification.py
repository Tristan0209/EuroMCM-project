import math

LIMITS = {'service_violation_upper': .001, 'cvar95_service_upper': .005,
          'model_selection_harm_upper': .02, 'model_pass_harm_upper': .02,
          'end_to_end_harm_upper': .02, 'worst_group_model_harm_upper': .05,
          'worst_group_violation_increment_upper': .005}
REAL_EVIDENCE = ['independent_answer_quality', 'physical_power', 'asof_forecasts', 'production_trace', 'representative_groups']


def qualify_offline(evidence):
    required = list(LIMITS) + ['total_carbon_saving_lower', 'minimum_tail_ess', 'contracts_passed', 'worlds_complete']
    if any(k not in evidence for k in required):
        return {'status': 'INSUFFICIENT_EVIDENCE', 'reason': 'missing preregistered evidence fields'}
    numbers = list(LIMITS) + ['total_carbon_saving_lower', 'minimum_tail_ess']
    if any(not isinstance(evidence[k], (int, float)) or not math.isfinite(evidence[k]) for k in numbers):
        return {'status': 'INSUFFICIENT_EVIDENCE', 'reason': 'nonfinite/unknown estimates'}
    if not evidence['contracts_passed'] or not evidence['worlds_complete']:
        return {'status': 'INSUFFICIENT_EVIDENCE', 'reason': 'software or all-world evidence gate failed'}
    if evidence['minimum_tail_ess'] < 50:
        return {'status': 'INSUFFICIENT_EVIDENCE', 'reason': 'tail ESS below 50'}
    failed = [name for name, limit in LIMITS.items() if evidence[name] > limit]
    if evidence['total_carbon_saving_lower'] <= 0:
        failed.append('total_carbon_saving_lower')
    if failed:
        return {'status': 'NOT_QUALIFIED_IN_TESTED_SCOPE', 'failed': failed}
    missing_real = [k for k in REAL_EVIDENCE if evidence.get(k) is not True]
    return {'status': 'CONDITIONAL_SIMULATION_CANDIDATE_ONLY' if missing_real else 'ELIGIBLE_FOR_CONTROLLED_DEPLOYMENT_REVIEW',
            'missing_real_evidence': missing_real, 'scope': 'offline assessment, not a per-request policy or production guarantee'}
