"""Run the complete recorded experiment pipeline."""
from pathlib import Path
import argparse, subprocess, sys, os
ROOT=Path(__file__).resolve().parent
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=min(12,os.cpu_count() or 1))
    p.add_argument('--check-mps',action='store_true')
    a=p.parse_args()
    if a.workers<1:p.error('--workers must be positive')
    (ROOT/'results').mkdir(exist_ok=True)
    stages=[
        ['src/test_protocol.py'],
        ['src/run_experiments.py','--phase','all','--workers',str(a.workers),'--seeds','30'],
        ['src/extra_experiments.py','--workers',str(a.workers)],
        ['src/synthetic.py'],
        ['src/analyze.py']+(['--check-mps'] if a.check_mps else []),
        ['verify_results.py'],
    ]
    for stage in stages:
        print('\nRunning '+' '.join(stage),flush=True)
        subprocess.run([sys.executable,*stage],cwd=ROOT,check=True)
    print('\nAll experiment stages and reproduction checks passed.',flush=True)
if __name__=='__main__':main()
