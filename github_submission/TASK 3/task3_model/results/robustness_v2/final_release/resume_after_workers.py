from context import *
import time,subprocess
def main():
    while True:
        completed=set()
        for p in (ROOT/'runs').glob('*/metrics.json'):
            try:m=read(p)
            except json.JSONDecodeError:continue
            if m['n']==7335 and m['gamma']==1.35 and m['delay_scale']==0. and m['strategy'] in ['Greedy','Spatial_MILP','MPC','E-Spatial','E-Full']:
                completed.add((m['seed'],m['strategy']))
        if len(completed)==120:break
        text=(ROOT/'audit/core_worker_12.log').read_text(encoding='utf-8-sig',errors='replace')
        if 'Traceback (most recent call last)' in text:raise RuntimeError('Independent producer failed; stop before new experiments')
        time.sleep(10)
    for name in ['core','pipeline']:
        old=ROOT/'audit'/(name+'.log');archived=ROOT/'audit'/(name+'_attempt1.log')
        assert not archived.exists();old.rename(archived)
    for command,log in [(['experiments.py','--phase','core'],'core'),(['pipeline.py'],'pipeline')]:
        with (ROOT/'audit'/(log+'.log')).open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/command[0]),*command[1:]],stdout=stream,stderr=subprocess.STDOUT,check=True)
    save(ROOT/'audit/orchestration_resume.json',{'status':'PASS','reason':'initial aggregator encountered an in-progress case and refused overwrite',
        'resolution':'waited for all 120 distinct metrics then rebuilt aggregate from same actual cases; no new statistical samples',
        'old_logs':'core_attempt1.log and pipeline_attempt1.log retained'})
if __name__=='__main__':main()
