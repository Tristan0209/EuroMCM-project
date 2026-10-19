from pathlib import Path
import sys
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SIMPLE = HERE.parent
TASK4 = SIMPLE.parent
sys.path.insert(0,str(TASK4))
sys.path.insert(0,str(TASK4/'audit'))
from task4_core.sources import sha,BASE,RELEASE
from task4_core.serialization import save
from inspect_interfaces import protected_snapshot

if __name__=='__main__':
    target = HERE/'before.json'
    assert not target.exists(),'Preserve prior audit baseline'
    existing = [p for p in TASK4.rglob('*') if p.is_file() and HERE not in p.parents]
    save(target,{'old_task4':{str(p):sha(p) for p in existing},'upstream':protected_snapshot(),
        'canonical_release_used':{str(p):sha(p) for p in list(RELEASE.glob('*.py'))+
            list((RELEASE/'v2').glob('*.py'))+list((BASE/'results/data').glob('*')) if p.is_file()},
        'scope':'All existing TASK4 artifacts including 24 runs and old reports; all D upstream protected files; C release source and data',
        'new_DES_runs':0,'new_routing_runs':0,'training':0})
    print('Saved immutable pre-review snapshot',len(existing))
