import sys
sys.dont_write_bytecode=True
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import datetime,json,hashlib
from functools import lru_cache
import numpy as np,pandas as pd
import statsmodels.api as sm
from scipy import stats,special
ROOT=Path(__file__).resolve().parents[1];OUT=Path(os.environ.get('R3_OUTPUT_ROOT',str(ROOT/'out/R3')));SEED=20261007;REPS=1999;EPS=1e-12
ASSETS=['BTC','ETH','SOL','BNB','XRP','DOGE','ADA','LINK','AVAX','LTC']
def log(s):
 with (ROOT/'logs/RUNLOG_R3.md').open('a') as f:f.write('- '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+s+'\n')
 print(s,flush=True)
def csv(d,p):
 d=d if isinstance(d,pd.DataFrame) else pd.DataFrame(d);p=OUT/p;p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False);return d
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def daily(asset,measure='rv5'):
 d=pd.read_parquet(ROOT/'data_clean/market_daily.parquet');d=d[(d.market=='futures/um')&(d.asset==asset)].sort_values('date').set_index('date')
 d=d.reindex(pd.date_range(d.index.min(),d.index.max(),freq='D'));d.loc[~d.valid_day.fillna(False).astype(bool),measure]=np.nan;return d
from forecast_core import har,forward_mean,qlike,hac_mean,compare_paths

def rolling(x,y,h,window=1000,form='log',start=1000):
 a=np.asarray(x,float);yy=np.asarray(y,float);a=np.column_stack([np.ones(len(a)),a]);good=np.isfinite(a).all(1)&np.isfinite(yy)&(yy>0)
 target=np.log(np.maximum(yy,EPS)) if form=='log' else yy
 clean=np.where(good[:,None],a,0);ty=np.where(good,target,0)
 xx=np.einsum('ni,nj->nij',clean,clean);xy=clean*ty[:,None]
 cx=np.concatenate([np.zeros((1,a.shape[1],a.shape[1])),np.cumsum(xx,axis=0)]);cy=np.vstack([np.zeros(a.shape[1]),np.cumsum(xy,axis=0)])
 pred=np.full(len(yy),np.nan);lasts=np.full(len(yy),np.datetime64('NaT'),dtype='datetime64[ns]')
 for t in range(start,len(yy)):
  end=t-h+1;begin=max(0,end-window)
  if not np.isfinite(a[t]).all():continue
  use=np.flatnonzero(good[begin:end])+begin
  if len(use)<max(100,5*a.shape[1]):continue
  mat=cx[end]-cx[begin];vec=cy[end]-cy[begin]
  # Scale cross-products to unit diagonal; rank deficiency uses Moore-Penrose.
  scale=np.sqrt(np.maximum(np.diag(mat),1e-24));beta=np.linalg.pinv(mat/scale[:,None]/scale[None,:],rcond=1e-10)@(vec/scale)/scale
  p=a[t]@beta
  if form=='log':p=np.exp(np.clip(p,-50,30))*np.mean(np.exp(np.clip(target[use]-a[use]@beta,-50,30)))
  pred[t]=max(p,EPS);lasts[t]=x.index[use[-1]]+pd.Timedelta(days=h+1)
 return pd.Series(pred,index=x.index),lasts

def mean_ci(x,lag=7):
 x=np.asarray(x,float);m,se,t=hac_mean(x,lag)
 if not np.isfinite(m) and np.isfinite(x).any():m=float(x[np.isfinite(x)].mean())
 return dict(mean=m,se=se,ci_low=m-1.96*se,ci_high=m+1.96*se,p=2*stats.norm.sf(abs(t)),n=int(np.isfinite(x).sum()))
def improvement(y,a,b,h):
 z=pd.DataFrame({'y':y,'a':a,'b':b}).dropna()
 if len(z)<30:return dict(n=len(z),status='insufficient')
 r=compare_paths(z.y,z.a,z.b,h)
 for label,la,lb in [('qlike',qlike(z.y,z.a),qlike(z.y,z.b)),('mse',(z.y-z.a)**2,(z.y-z.b)**2)]:
  ratio=np.mean(lb)/np.mean(la);influence=100*(ratio*np.asarray(la)-np.asarray(lb))/np.mean(la);m,se,t=hac_mean(influence,2*h)
  value=100*(1-ratio);r.update({label+'_improvement_pct':value,label+'_ci_low':value-1.96*se,label+'_ci_high':value+1.96*se})
 return r

@lru_cache(maxsize=2)
def grid(asset):
 p=OUT/f'shared/grid_{asset}.parquet'
 if p.exists():return pd.read_parquet(p)
 folder=ROOT/f'data_clean/minute/futures/um/{asset}'
 parts=[pd.read_parquet(p,columns=['timestamp','close','high','low','quote_volume']) for p in sorted(folder.glob('*.parquet')) if not p.name.startswith('._')]
 d=pd.concat(parts,ignore_index=True).sort_values('timestamp')
 if d.timestamp.duplicated().any():raise ValueError('Duplicate minute timestamps '+asset)
 d['timestamp']=pd.to_datetime(d.timestamp)+pd.Timedelta(minutes=1);d=d.set_index('timestamp')
 a=d.resample('5min',label='right',closed='right').agg(close=('close','last'),high=('high','max'),low=('low','min'),volume=('quote_volume','sum'),count=('close','count'))
 a['close']=d.close.reindex(a.index)
 a['logret']=np.log(a.close/a.close.shift(1));a.loc[(a['count']!=5)|(a['count'].shift(1)!=5),'logret']=np.nan
 a['rv']=a.logret**2;a['boundary_price_known']=a.close.notna();a.to_parquet(p);log(f'{asset} 5分钟网格 {len(a)} 行，完整 {int((a["count"]==5).sum())}');return a

@lru_cache(maxsize=2)
def hourly(asset):
 a=grid(asset);h=a.resample('h',label='right',closed='right').agg(close=('close','last'),rv=('rv',lambda v:v.sum(min_count=12)),volume=('volume','sum'),count=('count','sum'))
 h.loc[h['count']!=60,['close','rv']]=np.nan;return h

def iv_daily(currency):
 d=pd.read_parquet(ROOT/f'data_clean/dvol_hourly_{currency}.parquet').sort_values('public_at');d['date']=pd.to_datetime(d.public_at).dt.tz_localize(None).dt.floor('D')-pd.Timedelta(days=1)
 # Final bar is selected only if it was public no later than the next UTC boundary.
 d['bar_end']=pd.to_datetime(d.public_at).dt.tz_localize(None);d=d[d.bar_end<=d.date+pd.Timedelta(days=1)]
 return d.groupby('date').tail(1).set_index('date').close/100

def plot_init():
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':100});return plt

def freeze():
 p=OUT/'source_manifest.csv'
 if p.exists():return
 rows=[]
 for folder in ['data_clean','code','out/shared']:
  for f in (ROOT/folder).rglob('*'):
   if not f.is_file() or f.name.startswith('._') or '__pycache__' in f.parts or f.name.startswith(('r2_','r3_')):continue
   st=f.stat();full='minute' not in f.parts
   rows.append(dict(path=str(f.relative_to(ROOT)),bytes=st.st_size,mtime_ns=st.st_mtime_ns,sha256=sha(f) if full else '',verify='sha256' if full else 'size_mtime'))
 csv(rows,'source_manifest.csv');log('本轮共用来源冻结 '+str(len(rows))+' 文件')
