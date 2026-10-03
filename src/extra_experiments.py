import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):os.environ[k]="1"
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import itertools,json,argparse
import numpy as np
import pandas as pd
from protocol import simulate,link_sequence,initial_radius,METRICS
ROOT=Path(__file__).resolve().parents[1]
def task(args):
    experiment,key,seed=args
    meta=json.loads((ROOT/'data/streams.json').read_text());mm={x['key']:x for x in meta};m=mm[key]
    z=np.load(ROOT/'data/streams.npz');y=z[key][m['val_end']-1:];cal=z[key][int(.8*m['train_end'])-1:m['train_end']]
    tuning=json.loads((ROOT/'results/tuning.json').read_text())
    index=list(mm).index(key);s,p,u=link_sequence(len(y)-1,2,seed+100003*(index+1))
    cs,cp,cu=link_sequence(len(cal)-1,2,910117+index)
    rows=[]
    if experiment=='sensitivity':
        configurations=itertools.product([4],[4,8,16,32,64],[.003,.01,.03,.1,.3,1.],[0])
    else:
        configurations=[(mode,32,tuning[f"{key.split('_')[0]}|{mode}|32"],1) for mode in range(9)]
    for mode,H,factor,predictor in configurations:
        qi=initial_radius(cal,cs,2*m['q0'],predictor=predictor)
        out,_=simulate(y,s,p,u,mode,2*m['q0'],H,factor*m['q0'],qi,predictor=predictor)
        row=dict(experiment=experiment,stream=key,dataset=key.split('_')[0],seed=seed,mode=mode,H=H,eta_factor=factor,predictor=predictor,n=len(y)-1)
        row.update(dict(zip(METRICS,out)));rows.append(row)
    return rows
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=12);args=parser.parse_args()
    meta=json.loads((ROOT/'data/streams.json').read_text())
    tasks=list(itertools.product(['sensitivity','predictor'],[x['key'] for x in meta],range(10)))
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for x in pool.map(task,tasks,chunksize=4):rows.extend(x)
    pd.DataFrame(rows).to_csv(ROOT/'results/extra.csv.gz',index=False,float_format='%.12g',compression={'method':'gzip','mtime':0})
    print('Additional runs',len(rows),'prediction evaluations',sum(x['n'] for x in rows))
if __name__=='__main__':main()
