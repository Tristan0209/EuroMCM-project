import ast
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from task4_core.sources import read,sha
from task4_core.serialization import save
from task4_core.registry import reusable

HERE = ROOT/'audit/integration_v3'
target = HERE/'delivery_validation_final.json'
assert not target.exists(), 'Keep previous delivery validation'
names = ['Task4_Implementation_Plan_ZH.md','Task4_Mathematical_Model_ZH.md','Task4_Execution_Checklist_ZH.md',
         'Task4_Review_Fixes_ZH.md','Task4_Core_Experiment_Protocol.md','Task4_Readiness_Report.md','Task4_Estimated_Runtime.md',
         'Task4_Pilot_Report_ZH.md','Task4_Integration_Validation.md','Task4_Scenario_Generator_Spec.md',
         'Task4_Runtime_Benchmark.md','Task4_Readiness_Report_v3.md']
broken = []
for name in names:
    p = ROOT/name
    text = p.read_text(encoding='utf-8')
    assert len(text)>1000 and '\ufffd' not in text and text.count('```')%2==0,name
    for href in re.findall(r'\]\(([^\n]+?)\)',text):
        path = href.strip('<>')
        if re.match(r'^[A-Za-z]:/',path) and not Path(path).exists():
            broken.append({'document':name,'href':href})
assert not broken,broken
code = list((ROOT/'task4_core').glob('*.py'))+list((ROOT/'tests').glob('*.py'))+[ROOT/'run_task4.py']
for p in code:
    ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
run = ROOT/'runs/pilot_v3_001'
records = [read(p) for p in sorted((run/'executions').glob('*/attempt_*/record.json'))]
assert len(records)==28 and all(reusable(r,r['config']) for r in records)
current_source_mismatches = []
for record in records:
    for path,expected in record['config']['source_hashes'].items():
        p = Path(path)
        if p.is_absolute() and (not p.is_file() or sha(p)!=expected):
            current_source_mismatches.append(path)
assert not current_source_mismatches,current_source_mismatches
ledger = [json.loads(line) for line in (run/'ledger.jsonl').read_text(encoding='utf-8').splitlines()]
assert sum(r['event']=='COMPLETE_VALIDATED' for r in ledger)==28
assert sum(r['event']=='REUSE_REFERENCE' for r in ledger)==28
assert not any('FAILED' in r['event'] for r in ledger)
tests = [(HERE/'baseline_001/contract_results.json','tests_run'),(HERE/'attempt_002/integration_results.json','tests_run'),
         (HERE/'analysis_001/results.json','tests_run'),(HERE/'additional_001/results.json','tests_run')]
assert all(read(p)['status']=='PASS' for p,_ in tests)
assert sum(read(p)[key] for p,key in tests)==42
value = {'status':'PASS','distinct_contracts_passed':42,'pilot_new_executions':28,'pilot_worlds':14,
         'reuse_references':28,'new_DES_during_resume':0,'formal_experiments':0,'training_executed':False,
         'broken_local_links':broken,'current_source_hash_mismatches':current_source_mismatches,
         'readiness':'NO_GO_DISK_SPACE','documents':{str(ROOT/n):sha(ROOT/n) for n in names},
         'code':{str(p):sha(p) for p in code},'upstream_protection':read(HERE/'protection_after_pilot.json'),
         'v2_snapshots':{str(p):sha(p) for p in (HERE/'documents_v2').glob('*.md')}}
save(target,value)
print(json.dumps({k:v for k,v in value.items() if k not in ['documents','code','v2_snapshots']},indent=2))
