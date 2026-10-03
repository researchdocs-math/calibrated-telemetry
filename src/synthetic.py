from pathlib import Path
import numpy as np,pandas as pd
from protocol import simulate,link_sequence,METRICS
ROOT=Path(__file__).resolve().parents[1]
def main():
    rng=np.random.default_rng(109912);n=50000
    a=rng.beta(2,5,n)
    y=np.concatenate([[0.],np.concatenate([.2*a[:10000],.3+.5*a[10000:20000],.1+.8*a[20000:30000],.15*a[30000:40000],.2+.6*a[40000:]])])
    rows=[];traces={}
    for channel in [2,3]:
        for seed in range(30):
            s,p,u=link_sequence(n,channel,22001+seed)
            for mode in range(9):
                out,tr=simulate(y,s,p,u,mode,2.,32,.002,.11,keep=(seed==0))
                row=dict(channel=channel,seed=seed,mode=mode,n=n)
                row.update(dict(zip(METRICS,out)));rows.append(row)
                if seed==0:traces[f'channel{channel}_mode{mode}']=tr
    pd.DataFrame(rows).to_csv(ROOT/'results/synthetic.csv',index=False,float_format='%.12g')
    np.savez_compressed(ROOT/'results/synthetic_traces.npz',**traces)
    print('Synthetic experiment:',len(rows),'runs')
if __name__=='__main__':main()
