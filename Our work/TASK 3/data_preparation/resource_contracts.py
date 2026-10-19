import numpy as np


def service_at_pool(reference_seconds, reference_multiplier, pool_multiplier):

    if not np.isfinite([reference_seconds, reference_multiplier, pool_multiplier]).all():
        raise ValueError('Service inputs must be finite')
    if reference_seconds < 0 or reference_multiplier <= 0 or pool_multiplier <= 0:
        raise ValueError('Invalid service/multiplier')
    return reference_seconds * pool_multiplier / reference_multiplier


def incremental_energy_kwh(service_seconds, idle_w, busy_w, pue):
    if not np.isfinite([service_seconds,idle_w,busy_w,pue]).all():
        raise ValueError('Nonfinite energy input')
    if service_seconds < 0 or idle_w < 0 or busy_w < idle_w or pue < 1:
        raise ValueError('Invalid energy inputs')
    return service_seconds*(busy_w-idle_w)*pue/3.6e6


def facility_energy_components(horizon_seconds, capacity, busy_server_seconds, idle_w, busy_w, pue):
    if (not np.isfinite([horizon_seconds,capacity,busy_server_seconds]).all()
            or horizon_seconds < 0 or capacity < 0 or capacity % 1
            or not 0 <= busy_server_seconds <= capacity*horizon_seconds):
        raise ValueError('Invalid aggregate observation window or occupancy')
    increment = incremental_energy_kwh(busy_server_seconds,idle_w,busy_w,pue)
    base = capacity*idle_w*horizon_seconds*pue/3.6e6
    return {'idle_baseline_kwh':base, 'busy_increment_kwh':increment, 'facility_kwh':base+increment}
