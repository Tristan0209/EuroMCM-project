from pathlib import Path


def paired_data(project):
    project = Path(project)
    final = project / 'task2/inputs/paired'
    return final if final.is_dir() else project / 'task2_router/data'


def frozen_features(project):
    project = Path(project)
    final = project / 'task2/inputs/frozen_features'
    return final if final.is_dir() else project / 'task2_neural/runs/minilm_gain_20261016_matched'
