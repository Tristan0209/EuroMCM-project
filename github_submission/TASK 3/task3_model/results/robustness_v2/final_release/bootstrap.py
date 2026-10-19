from context import *
import platform, subprocess

def main():
    (ROOT/'audit').mkdir(exist_ok=True)
    assert not (ROOT/'audit/protected_before.json').exists(),'Initial bootstrap already completed; do not overwrite evidence or sources'
    old=read(PREVIOUS/'audit/protected_before.json')

    for directory in [A2,PREVIOUS]:
        for p in directory.rglob('*'):
            if p.is_file():old[str(p.relative_to(PROJECT))]=sha(p)
    save(ROOT/'audit/protected_before.json',old)
    (ROOT/'v2').mkdir(exist_ok=True)
    for name in ['__init__.py','des.py','milp.py']:
        (ROOT/'v2'/name).write_bytes((A2/'v2'/name).read_bytes())
    for name in ['queue_milp.py','policies.py','run_tests.py']:
        (ROOT/name).write_bytes((PREVIOUS/name).read_bytes())
    p=ROOT/'run_tests.py';s=p.read_text(encoding='utf-8')
    s=s.replace("str(ROOT/'tests'),","str(PREVIOUS/'tests'),str(ROOT/'tests'),")
    p.write_text(s,encoding='utf-8')
    s=(EXT/'spatial.py').read_text(encoding='utf-8')
    s='from context import *\nfrom v2.des import Simulation as FullSimulation\nfrom queue_milp import spatial_solve\n'+s[s.index('class SpatialSimulation'):]
    (ROOT/'spatial.py').write_text(s,encoding='utf-8')
    save(ROOT/'audit/environment.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
        'environment':{k:os.getenv(k) for k in ['PYTHONHASHSEED','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']},
        'packages':subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)})
if __name__=='__main__':main()
