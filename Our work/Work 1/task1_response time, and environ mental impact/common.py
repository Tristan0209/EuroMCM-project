import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parent


DATASETS = ROOT.parents[1] / 'AI_datasets'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
