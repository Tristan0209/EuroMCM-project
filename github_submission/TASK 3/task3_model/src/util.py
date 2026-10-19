import hashlib
import json
from functools import lru_cache
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
PROJECT=ROOT.parents[1]
PREP=ROOT.parent/'data_preparation'
DATA=ROOT/'results/data'


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def save(path,value):
    def clean(v):
        if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
        if isinstance(v,(list,tuple)):return [clean(x) for x in v]
        if isinstance(v,np.ndarray):return clean(v.tolist())
        if isinstance(v,np.generic):return clean(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        if isinstance(v,Path):return str(v)
        return v
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(clean(value),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def snapshot():
    task2=read(ROOT/'configs/upstream.json')['task2_directory']
    roots=[PROJECT/task2,PROJECT/'Work 1',PREP]
    return {str(p.relative_to(PROJECT)):sha(p) for base in roots for p in base.rglob('*') if p.is_file()}


INPUT_NAMES=['requests.csv','server_config.csv','network_config.csv','service_random_conditions.csv','alignment.json',
    'carbon_lifecycle.csv','carbon_lifecycle.metadata.json','carbon_direct.csv','carbon_direct.metadata.json',
    'server_availability_normal.csv','server_availability_local_failure.csv','server_availability_high_load_capacity_reduction.csv',
    'evaluation_only_historical_quality.csv','physical_host_config.csv',
    'resource_accounting_contract.json','resource_parameters.json','upstream_source_manifest.json']


def signature_for_files(paths):
    hashes={str(path):sha(path) for path in paths}
    return {'files':hashes,'sha256':hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()}


@lru_cache(maxsize=1)
def input_signature():
    from .power import upstream_paths
    paths=upstream_paths()
    manifest=read(DATA/'submission_upstream_source_manifest.json')
    for name,expected in manifest['files'].items():
        path=PROJECT/name
        if not path.is_file() or sha(path)!=expected:
            raise RuntimeError('Upstream source changed; rerun --phase audit before experiments: '+name)
    source_files=[paths['resource_config'],paths['resource_core'],paths['run']/'resource_assumptions.json',
        paths['run']/'decisions.csv.gz',paths['run']/'pre_predictions.csv.gz',paths['run']/'pre_full_train.joblib',
        paths['run']/'selected_parameters.json',paths['run']/'final_model_choice.json']
    return signature_for_files([DATA/name for name in INPUT_NAMES]+[PREP/'config.json',ROOT/'configs/upstream.json']+source_files)
