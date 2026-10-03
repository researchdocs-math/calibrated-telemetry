import os
os.environ["VECLIB_MAXIMUM_THREADS"]="1"
from pathlib import Path
import json,argparse,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--check-mps',action='store_true');args=parser.parse_args()
    d=pd.read_csv(ROOT/'results/test.csv.gz');d['abs_gap']=abs(d.coverage-.9)
    group=['dataset','channel','H','h_factor','mode']
    metrics=['coverage','abs_gap','width','quantile_loss','bytes','empty_rate','full_rate','window_error']
    seed=d.groupby(group+['seed'])[metrics].mean().reset_index()
    means=seed.groupby(group)[metrics].mean()
    rng=np.random.default_rng(774109);counts=rng.multinomial(30,np.full(30,1/30),size=10000)/30.
    # All bootstrap columns use the same resampled seeds for paired comparisons.
    columns=[];matrix=[]
    for key,g in seed.groupby(group):
        g=g.sort_values('seed')
        for metric in metrics:
            columns.append((key,metric));matrix.append(g[metric].to_numpy())
    mat=np.asarray(matrix).T
    boot=counts@mat
    if args.check_mps:
        import torch,time
        if torch.backends.mps.is_available():
            start=time.perf_counter()
            tb=torch.from_numpy(counts.astype('float32')).to('mps')@torch.from_numpy(mat.astype('float32')).to('mps')
            torch.mps.synchronize()
            error=float(np.max(np.abs(tb.cpu().numpy()-boot)))
            (ROOT/'results/gpu_check.json').write_text(json.dumps(dict(device='Apple Metal (MPS)',matrix_shape=list(boot.shape),bootstrap_replicates=10000,max_absolute_difference=error,seconds=time.perf_counter()-start),indent=2))
            assert error<1e-4,error
            print('MPS bootstrap cross-check; max absolute error',error,flush=True)
    cis=np.quantile(boot,[.025,.975],axis=0)
    rows=[]
    for j,(key,metric) in enumerate(columns):
        row=dict(zip(group,key));row.update(metric=metric,mean=float(means.loc[key,metric]),low=float(cis[0,j]),high=float(cis[1,j]));rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT/'results/summary.csv',index=False,float_format='%.12g')
    # Paired seed differences at the fixed comparison configuration.
    rows=[]
    for dataset in ['beijing','intel']:
        x=seed[(seed.dataset==dataset)&(seed.channel==2)&(seed.H==32)&(seed.h_factor==2)]
        a=x[x['mode']==4].sort_values('seed')
        for mode in [0,1,2,3,5,6,7,8]:
            b=x[x['mode']==mode].sort_values('seed')
            for metric in metrics:
                diff=a[metric].to_numpy()-b[metric].to_numpy()
                lo,hi=np.quantile(counts@diff,[.025,.975])
                rows.append(dict(dataset=dataset,baseline=mode,metric=metric,difference=diff.mean(),low=lo,high=hi))
    pd.DataFrame(rows).to_csv(ROOT/'results/paired.csv',index=False,float_format='%.12g')
    cert=d[d['mode'].isin([3,4,5,7])]
    assert cert.identity_error.max()<1e-8
    assert np.all(cert.abs_gap<=cert.bound+1e-10)
    cet=d[d['mode']==4]
    K=.1*cet.H+.9*(cet.max_failure_run+1)
    assert np.all(cet.max_debt<=K+1e-8)
    assert np.all(abs(cet.terminal_debt)<=K+1e-8)
    assert np.all(cet.repair_success_rate*(.1*cet.H)<=.1+.8*(1-cet.coverage)+1e-8)
    report=dict(runs=len(d),evaluations=int(d.n.sum()),max_identity_error=float(cert.identity_error.max()),certificate_checks=len(cert),test_readings=int(d.drop_duplicates('stream').n.sum()))
    (ROOT/'results/validation_checks.json').write_text(json.dumps(report,indent=2))
    print(report)
if __name__=='__main__':main()
