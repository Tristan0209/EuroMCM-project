import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from inspect_interfaces import protected_snapshot
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    target = args.out.resolve()
    if HERE.resolve() not in target.parents or target.exists():
        raise ValueError('New evidence file inside integration_v3 required')
    before = json.loads((HERE.parent/'protected_before.json').read_text(encoding='utf-8'))
    current = protected_snapshot()
    changes = [p for p,h in before.items() if current.get(p)!=h]
    added = sorted(set(current)-set(before))
    value = {'status':'PASS' if not changes and not added else 'FAIL','protected_files':len(before),
             'changed_or_missing':changes,'added_upstream':added}
    target.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(value,ensure_ascii=False))
    sys.exit(0 if value['status']=='PASS' else 1)
