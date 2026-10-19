import argparse
import subprocess
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-project",
        type=Path,
        default=Path(r"D:\project\EuroMCM-project"),
        help="EuroMCM project root directory"
    )
    parser.add_argument(
        '--out',
        type=Path,
        default=HERE / 'runs' / 'experiment'
    )
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();out=args.out.resolve()
    if out.exists() and not args.resume:raise FileExistsError('Choose a new directory or explicitly --resume')
    def run(script,*extra):
        subprocess.run([sys.executable,'-B',str(HERE/script),'--out',str(out),*map(str,extra)],check=True,cwd=HERE.parent)
    if not (out/'prepared.json').exists():run('prepare.py','--source-project',args.source_project,*(['--resume'] if out.exists() else []))
    run('fit_routers.py','--resume')
    if not (out/'model_choice.json').exists():run('evaluate.py','--resume')
    if not (out/'final_model_choice.json').exists():run('calibrate_risk.py')
    if not (out/'finite_sample_calibration.json').exists():run('finalize_risk.py')
    run('final_artifacts.py')
    if not (out/'resource_scenarios.csv').exists():run('resource_scenarios.py','--source-project',args.source_project)
    subprocess.run([sys.executable,'-B',str(HERE/'test_complete.py')],check=True,cwd=HERE.parent)
    run('verify.py')
    run('write_report.py')
    print('Complete solution:',out/'RESULTS_zh.md')


if __name__=='__main__':main()
