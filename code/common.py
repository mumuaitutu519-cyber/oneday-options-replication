import os,sys,json,datetime
from pathlib import Path
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np,pandas as pd
import pyarrow as pa
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
from scipy import stats,special
import statsmodels.api as sm
from forecast_core import *
ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'data'; OUT=ROOT/'output'; WORK=OUT/'forecasts'; CHEN=WORK
WORK.mkdir(parents=True,exist_ok=True)
SEED=20261007;EPS=1e-12;FORCE=True;REPS=4999
HIST=['HAR','HAR_S','HAR_J','HAR_Q','HAR_clock','HAR_DVOL','GB','LASSO','RF']
CORE=HIST+['DVOL_raw','IV_raw','IV_calibrated','IV_HAR'];MODELS=CORE+['DVOL_calibrated','IV_GB']
def log(s):print(s,flush=True)
def csv(rows,name):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);p=WORK/Path(name).name;d.to_csv(p,index=False);return d
def ci(x,lag=7):
 x=np.asarray(x,float);x=x[np.isfinite(x)];m,se,t=hac_mean(x,lag)
 if not np.isfinite(m) and len(x):m=x.mean()
 return dict(n=len(x),mean=m,se=se,t=t,p=2*stats.norm.sf(abs(t)))
def compare(y,a,b,h=1):
 z=pd.DataFrame(dict(y=np.asarray(y),a=np.asarray(a),b=np.asarray(b))).replace([np.inf,-np.inf],np.nan).dropna();z=z[(z>0).all(1)]
 if len(z)<30:return dict(n=len(z),gain=np.nan,t=np.nan,p=np.nan)
 l0=qlike(z.y,z.a);l1=qlike(z.y,z.b);t,p=dm_test(l0,l1,h)
 return dict(n=len(z),gain=100*(1-l1.mean()/l0.mean()),t=t,p=p,loss=l1.mean())
def calendar_ci(values,dates,h=1):
 s=pd.Series(np.asarray(values,float),index=pd.DatetimeIndex(dates)).replace([np.inf,-np.inf],np.nan).sort_index();s=s.reindex(pd.date_range(s.index.min(),s.index.max()));x=s.to_numpy();ok=np.isfinite(x);v=x[ok];n=len(v);m=v.mean();u=np.where(ok,x-m,0);L=max(7,2*h);V=u@u
 for j in range(1,min(L,len(u)-1)+1):V+=2*(1-j/(L+1))*(u[j:]@u[:-j])
 se=np.sqrt(max(V,0))/n;t=m/se if se else np.nan
 return dict(n=n,mean=m,se=se,t=t,p=2*stats.norm.sf(abs(t)))
def load_input(key):
 """Lossless shared-feature storage: per-specification columns override the shared daily panel."""
 delta=pd.read_parquet(DATA/'forecast_inputs'/(key+'.parquet'));base=pd.read_parquet(DATA/('features_'+key.split('_')[0]+'.parquet'))
 assert delta.date.equals(base.date),key
 for col in delta:base[col]=delta[col]
 return base
