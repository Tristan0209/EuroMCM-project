import os,sys,json,hashlib
from pathlib import Path
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parents[2];PROJECT=BASE.parents[1]
A2=ROOT.parent/'task_a2';EXT=A2/'extensions';PREVIOUS=ROOT.parent/'task_final_validation'
sys.path[:0]=[str(ROOT),str(BASE),str(A2),str(EXT),str(PREVIOUS)]
from src.util import read,save,sha
SEEDS=list(range(2027101800,2027101824))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,default=str).encode()).hexdigest()
def out(name):
    p=ROOT/'results'/name;p.mkdir(parents=True,exist_ok=True);return p
def signature():
    paths=list(ROOT.glob('*.py'))+list((ROOT/'v2').glob('*.py'))+list((BASE/'src').glob('*.py'))
    paths+=list((BASE/'results/data').glob('*.csv'))+list((BASE/'results/data').glob('*.json'))
    paths+=[BASE/'configs/core.json',A2/'configs/solver_protocol_frozen.json',ROOT/'configs/protocol.json']
    return {str(p.relative_to(PROJECT)):sha(p) for p in sorted(set(paths)) if p.exists()}
