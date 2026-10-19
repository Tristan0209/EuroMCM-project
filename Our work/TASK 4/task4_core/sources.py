import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
CANONICAL = PROJECT
RELEASE = CANONICAL / 'TASK 3/task3_model/results/robustness_v2/final_release'
BASE = RELEASE.parents[2]
TASK2 = PROJECT / 'task2'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def load_final(scratch):
    scratch = Path(scratch).resolve()
    if ROOT.resolve() not in scratch.parents:
        raise ValueError('Solver evidence must stay inside TASK 4')
    if not RELEASE.is_dir():
        raise FileNotFoundError('Final Spatial release missing; no legacy fallback')
    sys.dont_write_bytecode = True
    for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
        os.environ.setdefault(name, '1')
    for name, expected in [('src', BASE / 'src/__init__.py'), ('v2', RELEASE / 'v2/__init__.py'),
                           ('spatial', RELEASE / 'spatial.py'), ('context', RELEASE / 'context.py')]:
        module = sys.modules.get(name)
        if module and Path(module.__file__).resolve() != expected.resolve():
            raise RuntimeError('Conflicting imported module: ' + name)
    manifest = read(RELEASE / 'submission_source_manifest.json')['artifacts']
    identities = {}
    for relative in ['spatial.py', 'queue_milp.py', 'recovery.py', 'v2/des.py', 'v2/milp.py']:
        expected = manifest.get(relative) or manifest.get(relative.replace('/', '\\'))
        actual = sha(RELEASE / relative)
        if expected != actual:
            raise RuntimeError('Frozen final source changed: ' + relative)
        identities[relative] = actual
    sys.path[:0] = [str(RELEASE), str(BASE)]
    engine = importlib.import_module('v2.des')
    spatial = importlib.import_module('spatial')
    queue = importlib.import_module('queue_milp')
    queue.ROOT = scratch
    if spatial.spatial_solve is not queue.spatial_solve:
        raise RuntimeError('Spatial solver identity mismatch')
    identities['engine_file'] = str(Path(engine.__file__).resolve())
    identities['spatial_file'] = str(Path(spatial.__file__).resolve())
    return engine, spatial, identities
