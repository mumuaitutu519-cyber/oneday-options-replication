import os,sys
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import json,hashlib,datetime
from functools import lru_cache
import numpy as np,pandas as pd
from scipy import stats,special
import statsmodels.api as sm
ROOT=Path(__file__).resolve().parents[1];OUT=Path(os.environ.get('R5_OUTPUT_ROOT',str(ROOT/'out/R5')));CHEN=OUT/'chen';CHEN.mkdir(parents=True,exist_ok=True)
R4=ROOT/'out/R4/chen';SEED=20261007;EPS=1e-12;CUTOFF=pd.Timestamp('2026-10-07');FORCE=os.environ.get('R5_FORCE_REBUILD')=='1'
TENORS=[1,2,3,7,14,30];REPS=4999
from forecast_core import qlike,hac_mean,dm_test,cw_test,mcs
from r4_common import ci,loss_compare
from r4_options import fullmeta,brackets,mfvariance
from r2_core import black,implied_w

def log(s):
 p=ROOT/'logs/RUNLOG_R5.md';p.parent.mkdir(exist_ok=True)
 with p.open('a') as f:f.write('- '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+str(s)+'\n')
 print(s,flush=True)
def csv(rows,name):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);p=CHEN/name;p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False);return d
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def rawday(c,date):return pd.read_parquet(ROOT/f'out/R2/raw/{c}/{pd.Timestamp(date).strftime("%Y-%m-%d")}/trades.parquet')
@lru_cache(maxsize=2)
def stationary_indices(n,reps=REPS,block=7):
 rng=np.random.default_rng(SEED);idx=np.empty((reps,n),dtype=np.int32);idx[:,0]=rng.integers(n,size=reps)
 for j in range(1,n):
  restart=rng.random(reps)<1/block;idx[:,j]=np.where(restart,rng.integers(n,size=reps),(idx[:,j-1]+1)%n)
 return idx
@lru_cache(maxsize=12)
def stationary_weights(n,block=7):
 idx=stationary_indices(n,block=block);offset=np.arange(REPS,dtype=np.int64)[:,None]*n
 return np.bincount((idx+offset).ravel(),minlength=REPS*n).reshape(REPS,n).astype(float)
def bootstrap_distribution(x,block=7):
 if isinstance(x,pd.Series) and isinstance(x.index,pd.DatetimeIndex):x=x.reindex(pd.date_range(x.index.min(),x.index.max())).to_numpy(float)
 else:x=np.asarray(x,float)
 valid=np.isfinite(x);weights=stationary_weights(len(x),block=block);den=weights@valid.astype(float);return (weights@np.where(valid,x,0))/np.where(den>0,den,np.nan)
def stationary_statistics(x,block=7):
 if not np.isfinite(x).any():return np.full(REPS,np.nan),np.full(REPS,np.nan)
 if isinstance(x,pd.Series) and isinstance(x.index,pd.DatetimeIndex):x=x.reindex(pd.date_range(x.index.min(),x.index.max())).to_numpy(float)
 else:x=np.asarray(x,float)
 valid=np.isfinite(x);order=np.flatnonzero(valid);order=order[np.argsort(x[order])];weights=stationary_weights(len(x),block=block);cum=np.cumsum(weights[:,order],axis=1);den=cum[:,-1];mean=(weights@np.where(valid,x,0))/np.where(den>0,den,np.nan)
 i=(cum>=np.ceil(den/2)[:,None]).argmax(1);j=(cum>=np.floor(den/2+1)[:,None]).argmax(1);median=(x[order[i]]+x[order[j]])/2;median[den==0]=np.nan;return mean,median
def bootstrap(x,stat='mean',block=7):
 n=int(np.isfinite(np.asarray(x,float)).sum())
 if n<10:return dict(boot_low=np.nan,boot_high=np.nan,bootstrap_n=n)
 z=bootstrap_distribution(x,block) if stat=='mean' else stationary_statistics(x,block)[1];lo,hi=np.nanquantile(z,[.025,.975]);return dict(boot_low=lo,boot_high=hi,bootstrap_n=n)
def gw_test(a,b,h=1):
 d=np.asarray(a)-np.asarray(b);d=d[np.isfinite(d)]
 if len(d)<30:return dict(gw_stat=np.nan,gw_p=np.nan)
 lag=d[:-h];scale=np.std(lag);Z=np.column_stack([np.ones(len(lag)),lag/max(scale,EPS)]);mom=Z*d[h:,None];n=len(mom);u=mom-mom.mean(0);S=u.T@u/n;L=max(7,2*h)
 for j in range(1,min(L,n-2)+1):
  g=u[j:].T@u[:-j]/n;S+=(1-j/(L+1))*(g+g.T)
 w=float(n*mom.mean(0)@np.linalg.pinv(S,rcond=1e-12)@mom.mean(0));return dict(gw_stat=w,gw_p=stats.chi2.sf(w,2))
def compare(y,a,b,h=1):
 z=pd.DataFrame(dict(y=y,a=a,b=b)).replace([np.inf,-np.inf],np.nan).dropna();z=z[(z>0).all(1)];r=loss_compare(z.y,z.a,z.b,h)
 if len(z)>=30:
  for name,l0,l1 in [('qlike',qlike(z.y,z.a),qlike(z.y,z.b)),('mse',(z.y-z.a)**2,(z.y-z.b)**2)]:r.update({name+'_'+k:v for k,v in gw_test(l0,l1,h).items()})
 return r
def periods(d):
 yield 'all',d
 if len(d):yield 'trimRV1',d[d.y<=d.y.quantile(.99)]
 for name,mask in [('2021-2022',d.date.dt.year.between(2021,2022)),('2023',d.date.dt.year.eq(2023)),('2024-2026',d.date.dt.year.between(2024,2026)),('ETF_pre',d.date<'2024-01-11'),('ETF_post',d.date>='2024-01-11')]:yield name,d[mask]
 for y,z in d.groupby(d.date.dt.year):
  if y!=2023:yield str(y),z
