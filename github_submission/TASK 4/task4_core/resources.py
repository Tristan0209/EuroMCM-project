import math
import numpy as np
import pandas as pd


def pools_from_hosts(hosts):
    hosts = pd.DataFrame(hosts).copy()
    required = ['host_id', 'region', 'model', 'gpus', 'slots', 'shared_w', 'gpu_idle_w', 'gpu_busy_w', 'pue', 'server_max_w']
    if hosts.empty or not set(required) <= set(hosts) or hosts.host_id.duplicated().any():
        raise ValueError('Unique, complete physical host table required')
    for row in hosts.itertuples():
        vals = [getattr(row, k) for k in ['gpus', 'slots', 'shared_w', 'gpu_idle_w', 'gpu_busy_w', 'pue', 'server_max_w']]
        if not all(math.isfinite(float(x)) for x in vals):
            raise ValueError('Nonfinite host parameters')
        if not (int(row.gpus) == row.gpus and int(row.slots) == row.slots and 1 <= row.slots <= row.gpus
                and 0 <= row.gpu_idle_w <= row.gpu_busy_w and row.shared_w >= 0 and row.pue >= 1
                and row.shared_w + row.gpus * row.gpu_busy_w <= row.server_max_w):
            raise ValueError('Invalid physical host capacity or power')
    pools = {}
    for (region, model), part in hosts.groupby(['region', 'model'], sort=False):
        delta = part.gpu_busy_w - part.gpu_idle_w
        if not np.allclose(delta, delta.iloc[0], rtol=0, atol=0) or part.pue.nunique() != 1:
            raise ValueError('Heterogeneous marginal power/PUE within a pool needs a host-aware optimizer')
        idle = float((part.shared_w + part.gpus * part.gpu_idle_w).sum())
        pools[(region, model)] = {'capacity': int(part.slots.sum()), 'idle': idle,
                                 'busy': idle + float(delta.iloc[0]), 'idle_base_w': idle,
                                 'pue': float(part.pue.iloc[0]), 'installed_gpus': int(part.gpus.sum()),
                                 'host_id': '|'.join(part.host_id), 'physical_hosts': part.to_dict('records')}
    return pools


def host_for_replica(pool, replica):
    hosts = pool.get('physical_hosts')
    if not hosts:
        return pool.get('host_id')
    offset = 0
    for host in hosts:
        if offset <= replica < offset + host['slots']:
            return host['host_id']
        offset += host['slots']
    raise ValueError('Replica outside physical host mapping')
