from context import *
import argparse,shutil,subprocess
def main():
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--reproduce',metavar='NEW_DIRECTORY_NAME');a=p.parse_args()
    if not a.reproduce:
        subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'freeze.py'),'--verify'],check=True);return
    name=a.reproduce
    assert Path(name).name==name and name not in ['.','..'] and name.startswith('final_release_reproduction_')
    destination=ROOT.parent/name;assert not destination.exists(),'Never overwrite an existing release or reproduction'
    destination.mkdir()
    docs=[ROOT/name for name in ['README.md','REPRODUCIBILITY.md','FINAL_MODEL_FORMULATION.md']]
    for file in list(ROOT.glob('*.py'))+docs+list(ROOT.glob('requirements*.txt')):shutil.copy2(file,destination/file.name)
    for directory in ['v2','tests','configs']:shutil.copytree(ROOT/directory,destination/directory)
    (destination/'audit').mkdir()
    protected=read(ROOT/'audit/protected_before.json')
    for file in ROOT.rglob('*'):
        if file.is_file():protected[str(file.relative_to(PROJECT))]=sha(file)
    save(destination/'audit/protected_before.json',protected)
    shutil.copy2(ROOT/'audit/environment.json',destination/'audit/environment_reference.json')
    import platform
    save(destination/'audit/environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'packages':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)})
    env=dict(os.environ,**dict.fromkeys(['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'],'1'),PYTHONHASHSEED='20261017')
    commands=[['run_tests.py'],['preflight.py'],['experiments.py','--phase','pilot'],['experiments.py','--phase','core'],
        ['pipeline.py'],['normalize_tables.py'],['make_additional_figures.py'],['affected_request_audit.py'],['compare_events.py'],['validate_release.py'],['reports_release.py'],['freeze.py']]
    for command in commands:
        subprocess.run([sys.executable,'-B','-X','utf8',str(destination/command[0]),*command[1:]],check=True,env=env)
    print('Reproduction completed without altering the frozen release:',destination)
if __name__=='__main__':main()
