from context import *
import time,subprocess
def main():
    while not (ROOT/'audit/pipeline_completed.json').exists():
        p=ROOT/'audit/pipeline.log'
        if p.exists() and 'Traceback (most recent call last)' in p.read_text(encoding='utf-8-sig',errors='replace'):
            raise RuntimeError('Experiment pipeline failed; no final audit or freeze')
        time.sleep(10)
    for name in ['run_tests','normalize_tables','make_additional_figures','affected_request_audit','compare_events','validate_release','reports_release']:
        with (ROOT/'audit'/(name+'_final.log')).open('w',encoding='utf-8') as stream:
            subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/(name+'.py'))],stdout=stream,stderr=subprocess.STDOUT,check=True)
        print('Completed final check:',name,flush=True)
    save(ROOT/'audit/finish_checks_completed.json',{'status':'PASS','freeze':'not performed; inspect reports first'})
if __name__=='__main__':main()
