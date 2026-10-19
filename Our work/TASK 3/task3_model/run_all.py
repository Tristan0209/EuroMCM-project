import argparse
import subprocess
import sys
from src.util import ROOT,read,save


def tests():
    status=subprocess.call([sys.executable,'-B','-X','utf8','-m','pytest','-q','-p','no:cacheprovider','tests',
        '--junitxml='+str(ROOT/'results/audit/tests.xml')],cwd=ROOT)
    if status:raise SystemExit(status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['audit','test','pilot','scale','experiments','full-stress','report','all'],default='all')
    args=parser.parse_args()
    if args.phase in ['audit','all']:
        from src.phase0 import run
        run()
    if args.phase in ['test','pilot','all']:tests()
    if args.phase in ['pilot','all']:
        from src.experiments import scale
        scale([100])
    if args.phase in ['scale','all']:
        if not (ROOT/'results/audit/scale_gate.json').is_file():raise RuntimeError('Run pilot before larger scales')
        from src.experiments import scale
        scale([100,500,1000,7335])
    if args.phase in ['experiments','all']:
        gate=read(ROOT/'results/audit/scale_gate.json')
        if 7335 not in gate['sizes_completed']:raise RuntimeError('Full-scale correctness gate has not passed')
        from src.experiments import suite
        suite()
    if args.phase in ['full-stress','all']:
        if 7335 not in read(ROOT/'results/audit/scale_gate.json')['sizes_completed']:raise RuntimeError('Complete scale gate first')
        from src.experiments import full_stress
        full_stress()
    if args.phase in ['report','all']:
        from src.reporting import run
        run()


if __name__=='__main__':main()
