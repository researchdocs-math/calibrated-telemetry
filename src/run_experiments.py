import os
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS"): os.environ[k]="1"
from pathlib import Path
import json,time,argparse,itertools
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import pandas as pd
from protocol import simulate,link_sequence,initial_radius,MODES,METRICS
ROOT=Path(__file__).resolve().parents[1]
STREAMS=None;META=None;TUNED=None
FACTORS=[.003,.01,.03,.1,.3,1.]
def init():
    global STREAMS,META,TUNED
    STREAMS=np.load(ROOT/'data/streams.npz')
    META={m['key']:m for m in json.loads((ROOT/'data/streams.json').read_text())}
    p=ROOT/'results/tuning.json'
    TUNED=json.loads(p.read_text()) if p.exists() else {}
def task(args):
    phase,key,seed,kind,H=args
    m=META[key]; a=m['train_end'];b=m['val_end'];dataset=key.split('_')[0]
    y=STREAMS[key][a-1:b] if phase=='validation' else STREAMS[key][b-1:]
    s,pr,u=link_sequence(len(y)-1,kind,seed+100003*(list(META).index(key)+1)+(7000000 if phase=='validation' else 0))
    rows=[]
    for hm in ([1.,4.] if phase=='validation' else [.5,1.,2.,4.]):
        cal=STREAMS[key][int(.8*a)-1:a]
        cs,cp,cu=link_sequence(len(cal)-1,kind,910117+list(META).index(key))
        qinit=initial_radius(cal,cs,hm*m['q0'])
        for mode in range(9):
            factors=([.1] if mode==0 else FACTORS) if phase=='validation' else [TUNED[f'{dataset}|{mode}|{H}']]
            for factor in factors:
                out,_=simulate(y,s,pr,u,mode,hm*m['q0'],H,factor*m['q0'],qinit)
                row=dict(phase=phase,dataset=dataset,stream=key,seed=seed,channel=kind,H=H,h_factor=hm,mode=mode,eta_factor=factor,q_scale=m['q0'],n=len(y)-1,scale=m['scale'])
                row.update(dict(zip(METRICS,out)))
                rows.append(row)
    return rows
def run(phase,workers,seeds):
    meta=json.loads((ROOT/'data/streams.json').read_text())
    channels=range(3) if phase=='validation' else range(4)
    tasks=list(itertools.product([phase],[m['key'] for m in meta],range(seeds),channels,[8,32]))
    t=time.perf_counter();rows=[]
    print(phase,len(tasks),'jobs on',workers,'workers',flush=True)
    with ProcessPoolExecutor(max_workers=workers,initializer=init) as pool:
        for i,group in enumerate(pool.map(task,tasks,chunksize=4)):
            rows.extend(group)
            if (i+1)%500==0: print(phase,i+1,'/',len(tasks),'elapsed',round(time.perf_counter()-t,1),'s',flush=True)
    df=pd.DataFrame(rows);df.to_csv(ROOT/f'results/{phase}.csv.gz',index=False,float_format='%.12g',compression={'method':'gzip','mtime':0})
    print(phase,'done:',len(df),'runs,',int(df.n.sum()),'prediction evaluations;',round(time.perf_counter()-t,2),'seconds',flush=True)
    return df
def tune(df):
    df=df.copy();df['gap']=abs(df.coverage-.9);df['normalized_score']=df.quantile_loss/df.q_scale
    g=df.groupby(['dataset','mode','H','eta_factor'])[['gap','normalized_score']].mean().reset_index()
    chosen={}
    for (dataset,mode,H),d in g.groupby(['dataset','mode','H']):
        eligible=d[d.gap<=.02]
        row=eligible.sort_values(['normalized_score','eta_factor']).iloc[0] if len(eligible) else d.sort_values(['gap','normalized_score']).iloc[0]
        chosen[f'{dataset}|{mode}|{H}']=float(row.eta_factor)
    (ROOT/'results/tuning.json').write_text(json.dumps(chosen,indent=2))
    g.to_csv(ROOT/'results/tuning_scores.csv',index=False,float_format='%.12g')
    print('Tuning frozen:',chosen,flush=True)
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--workers',type=int,default=12);p.add_argument('--phase',choices=['all','validation','test'],default='all')
    p.add_argument('--seeds',type=int,default=30)
    args=p.parse_args();start=time.time()
    if args.phase in ('all','validation'):tune(run('validation',args.workers,3))
    if args.phase in ('all','test'):run('test',args.workers,args.seeds)
    (ROOT/'results/run_info.json').write_text(json.dumps(dict(workers=args.workers,test_seeds=args.seeds,elapsed_seconds=time.time()-start),indent=2))
if __name__=='__main__':main()
