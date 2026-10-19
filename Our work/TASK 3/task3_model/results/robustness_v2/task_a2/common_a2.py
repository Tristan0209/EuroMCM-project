import os
from pathlib import Path
import sys
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
PROJECT=BASE.parents[1]
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONHASHSEED']='20261017'
sys.path.insert(0,str(BASE))
import hashlib
import json
from src.util import read,save,sha

def protected_snapshot():
    paths=set()
    for root in [BASE,PROJECT/'task2',PROJECT/'Work 1',BASE.parent/'data_preparation']:
        paths.update(p for p in root.rglob('*') if p.is_file() and HERE not in p.parents)
    paths.update(PROJECT/p for p in read(BASE/'results/data/upstream_source_manifest.json')['files'])
    return {str(p.relative_to(PROJECT)):sha(p) for p in sorted(paths)}

def code_signature():
    paths=[BASE/'src'/n for n in ['domain.py','accounting.py','carbon.py','power.py']]
    paths+=list((HERE/'v2').glob('*.py'))+[HERE/'configs/protocol.json']
    return {str(p.relative_to(BASE)):sha(p) for p in paths}
