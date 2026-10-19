import hashlib
import json
from pathlib import Path
from .sources import sha

REQUIRED = {'source_hashes', 'routing_hash', 'request_hash', 'world_arrays_hash',
            'host_hash', 'network_hash', 'carbon_hash', 'forecast_hash', 'capacity_hash',
            'horizon_s', 'anchor_utc', 'controller_config', 'solver_protocol', 'accounting_version', 'policy'}


def execution_signature(config):
    if not REQUIRED <= set(config):
        raise ValueError('Incomplete execution signature: ' + ','.join(sorted(REQUIRED - set(config))))
    serialized = json.dumps(config, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def reusable(record, config):
    if record.get('status') != 'COMPLETE_VALIDATED' or record.get('signature') != execution_signature(config):
        return False
    artifacts = record.get('artifact_hashes', {})
    return bool(artifacts) and all(Path(p).is_file() and sha(p) == expected for p, expected in artifacts.items())
