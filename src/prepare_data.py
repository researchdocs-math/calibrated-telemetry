from pathlib import Path
import io,zipfile,gzip,json,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def prepare(raw):
    records=[]; data={}
    with zipfile.ZipFile(raw/'beijing.zip') as z:
        inner=zipfile.ZipFile(io.BytesIO(z.read('PRSA2017_Data_20130301-20170228.zip')))
        for name in sorted(inner.namelist()):
            if not name.endswith('.csv'): continue
            d=pd.read_csv(inner.open(name))
            values=d['PM2.5'].to_numpy(float)
            mask=np.isfinite(values)&(values>=0)&(values<=1000)
            dt=pd.to_datetime(d[['year','month','day','hour']])
            key='beijing_'+str(d.station.iloc[0])
            add(data,records,key,values[mask]/1000,dt[mask],1000,0,len(values)-mask.sum())
    # Aggregate irregular reports before removing empty bins: no fabricated targets.
    d=pd.read_csv(raw/'intel.txt.gz',sep=r'\s+',header=None,names=['date','time','epoch','mote','temperature','humidity','light','voltage'])
    d['datetime']=pd.to_datetime(d['date']+' '+d['time'],format='mixed',errors='coerce')
    d=d[d.datetime.notna()&d.mote.between(1,54)&d.temperature.between(-20,60)]
    for mote,g in d.groupby('mote'):
        g=g.sort_values('datetime').set_index('datetime').temperature.resample('5min').median()
        good=g.dropna()
        if len(good)<2000: continue
        add(data,records,'intel_'+str(int(mote)).zfill(2),(good.to_numpy()+20)/80,good.index,80,-20,len(g)-len(good))
    np.savez_compressed(ROOT/'data'/'streams.npz',**data)
    (ROOT/'data'/'streams.json').write_text(json.dumps(records,indent=2))
    print('Prepared',len(records),'streams and',sum(x['n'] for x in records),'observed readings',flush=True)
def add(data,records,key,y,dt,scale,offset,missing):
    # Split each observed stream chronologically, never use future values for training.
    y=np.asarray(y,dtype=np.float64)
    n=len(y); a=int(.5*n); b=int(.7*n)
    times=np.asarray(pd.DatetimeIndex(dt).asi8//10**9,dtype=np.int64)
    data[key]=y; data[key+'__time']=times
    initial=float(max(np.quantile(np.abs(np.diff(y[:a])),.9),1e-5))
    records.append(dict(key=key,n=n,train_end=a,val_end=b,test_n=n-b,scale=scale,offset=offset,q0=initial,missing_bins=int(missing),start=str(pd.Timestamp(times[0],unit='s')),end=str(pd.Timestamp(times[-1],unit='s')),median_gap_s=float(np.median(np.diff(times))),p95_gap_s=float(np.quantile(np.diff(times),.95))))
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--raw',type=Path,required=True)
    prepare(p.parse_args().raw)
