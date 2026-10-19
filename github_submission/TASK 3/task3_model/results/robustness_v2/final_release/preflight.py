from context import *
import xml.etree.ElementTree as ET
import pandas as pd
def main():
    xml=ET.parse(ROOT/'audit/tests.xml');suites=list(xml.getroot().iter('testsuite'))
    assert sum(int(x.get('tests','0')) for x in suites)==83
    assert not sum(int(x.get('errors','0'))+int(x.get('failures','0')) for x in suites)
    inventory=[]
    for directory in [BASE/'src',A2/'v2',EXT,PREVIOUS]:
        for p in sorted(directory.glob('*.py')):
            content=p.read_text(encoding='utf-8-sig')
            inventory.append({'path':str(p.relative_to(PROJECT)),'sha256':sha(p),'lines':len(content.splitlines()),'read_entire_file':True})
    for directory in [BASE,PREVIOUS,PREVIOUS/'reports']:
        for p in sorted(directory.glob('*.md')):
            p.read_text(encoding='utf-8-sig');inventory.append({'path':str(p.relative_to(PROJECT)),'sha256':sha(p),'read_entire_file':True})
    for p in sorted((BASE/'results/data').glob('*')):
        if p.suffix=='.csv':
            f=pd.read_csv(p);inventory.append({'path':str(p.relative_to(PROJECT)),'rows':len(f),'columns':list(f),'sha256':sha(p)})
        elif p.suffix=='.json':read(p)
    save(ROOT/'audit/read_inventory.json',inventory)
    assert sha(ROOT/'v2/des.py')==sha(A2/'v2/des.py')
    save(ROOT/'audit/source_and_input_hashes.json',signature())
    save(ROOT/'audit/preflight_gate.json',{'status':'PASS','tests_passed':83,'event_engine_byte_identical':True,
        'old_counterexample':'fixed-integer LP infeasible: retained explicit rejection; synthetic recoverable examples repaired',
        'scope_preregistered':'configs/protocol.json; fresh 120 core cases, affected F/H reruns; seeds independent, CRN paired',
        'tolerance':1e-5})
    print('PASS: 83 tests; immutable production inputs/code recorded')
if __name__=='__main__':main()
