from context import *
import argparse,datetime,platform
def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');a=p.parse_args()
    manifest=ROOT/'freeze_manifest.json'
    if a.verify:
        values=read(manifest);changed=[]
        for path,expected in values['artifacts'].items():
            file=ROOT/path
            if not file.is_file() or sha(file)!=expected:changed.append(path)
        protected=read(ROOT/'audit/protected_before.json');external=[]
        for path,expected in protected.items():
            file=PROJECT/path
            if not file.is_file() or sha(file)!=expected:external.append(path)
        assert not changed and not external,{'changed_release_artifacts':changed,'changed_protected_files':external}
        print('PASS read-only verification:',len(values['artifacts']),'release artifacts;',len(protected),'protected old files')
        return
    assert not manifest.exists(),'Never overwrite a frozen release'
    assert read(ROOT/'audit/final_gate.json')['status']=='PASS'
    assert read(ROOT/'audit/reports_completed.json')['status']=='PASS'
    assert read(ROOT/'audit/protection_after.json')['status']=='PASS'
    save(ROOT/'FREEZE_STATUS.json',{'status':'FROZEN_CONDITIONAL_SIMULATION_RELEASE',
        'frozen_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'validated execution and complete evidence; NOT zero-SLA, all-solves-success or deployed-hardware certification',
        'core':'7335 x 24 seeds x 5 policies; all new executions',
        'F_H':'six full-cohort seeds; affected safety/control reruns completed',
        'gate':'audit/final_gate.json','validation':'validation_checklist.csv',
        'measured_wall_reproduction':'not guaranteed across hardware or background load',
        'dependencies':{'python':platform.python_version(),'numpy':'1.24.4','scipy':'1.9.1','highs_bundled':'1.2.2','pandas':'2.3.3','matplotlib':'3.5.2','pytest':'7.1.2'}})

    artifacts={str(f.relative_to(ROOT)):sha(f) for f in sorted(ROOT.rglob('*')) if f.is_file() and f!=manifest}
    save(manifest,{'algorithm':'SHA-256','artifacts':artifacts,'source_and_normalized_inputs':signature(),
        'original_inputs_and_predecessors':'audit/protected_before.json','manifest_self_hash':'not included; avoids recursion'})
    print('Frozen',len(artifacts),'artifacts; physical-world validity remains unverified')
if __name__=='__main__':main()
