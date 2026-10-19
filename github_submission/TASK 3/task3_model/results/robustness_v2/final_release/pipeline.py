from context import *
import subprocess,time
import pandas as pd
def execute(script,*args):
    name=script.replace('.py','')+'_'+('_'.join(args) or 'main')
    with (ROOT/'audit'/(name+'.log')).open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/script),*args],stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(name+' failed; inspect preserved log')
def main():
    while True:
        p=ROOT/'results/core/raw_runs.csv'
        if p.exists() and len(pd.read_csv(p))==120:break
        text=(ROOT/'audit/core.log').read_text(encoding='utf-8-sig',errors='replace')
        if 'Traceback (most recent call last)' in text:raise RuntimeError('Core failed; no F/H permitted')
        time.sleep(10)
    execute('analyze_release.py','core')
    execute('experiments.py','--phase','F');execute('analyze_release.py','F')
    execute('experiments.py','--phase','H');execute('analyze_release.py','H')
    execute('repeat_stability.py','--repeat','1');execute('repeat_stability.py','--repeat','2')
    save(ROOT/'audit/pipeline_completed.json',{'core':120,'F_references':72,'H_references':72,'fresh_deterministic_repeats':10,'status':'PASS'})
if __name__=='__main__':main()
