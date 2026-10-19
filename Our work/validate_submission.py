from pathlib import Path, PureWindowsPath
import ast
import hashlib
import io
import json
import tokenize

ROOT = Path(__file__).resolve().parent


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def historical_path(value):
    parts = PureWindowsPath(value).parts
    index = parts.index('TASK 4')
    result = ROOT.joinpath(*parts[index:]).resolve()
    if ROOT not in result.parents:
        raise ValueError(value)
    return result


def main():
    manifest = read(ROOT / 'SUBMISSION_MANIFEST.json')
    for name, expected in manifest['files'].items():
        path = (ROOT / name).resolve()
        if ROOT not in path.parents or not path.is_file() or sha(path) != expected:
            raise ValueError('Submission integrity failed: ' + name)
    source_count = 0
    for path in ROOT.rglob('*.py'):
        source = path.read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        compile(tree, str(path), 'exec')
        if any(token.type == tokenize.COMMENT for token in tokenize.generate_tokens(io.StringIO(source).readline)):
            raise ValueError('Code comment: ' + str(path))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and ast.get_docstring(node, clean=False) is not None:
                raise ValueError('Docstring: ' + str(path))
        source_count += 1
    complete = read(ROOT / 'TASK 4/simple_assessment/runs/completion.json')
    if complete['status'] != 'PASS' or len(complete['rows']) != 24:
        raise ValueError('Incomplete historical Task 4 study')
    artifacts = 0
    for row in complete['rows']:
        path = historical_path(row['record'])
        record = read(path)
        if record['status'] != 'COMPLETE_VALIDATED' or record['world_id'] != row['world_id']:
            raise ValueError('Invalid saved execution: ' + str(path))
        for old_path, expected in record['artifact_hashes'].items():
            artifact = historical_path(old_path)
            if not artifact.is_file() or sha(artifact) != expected:
                raise ValueError('Historical artifact changed: ' + str(artifact))
            artifacts += 1
    print(json.dumps({'status': 'PASS', 'manifest_files': len(manifest['files']),
                      'python_files': source_count, 'comments': 0, 'docstrings': 0,
                      'historical_task4_executions': 24, 'historical_artifacts_checked': artifacts,
                      'scope': 'File integrity, Python syntax and saved Task 4 artifact hashes; no retraining or model-performance certification.'}, indent=2))


if __name__ == '__main__':
    main()
