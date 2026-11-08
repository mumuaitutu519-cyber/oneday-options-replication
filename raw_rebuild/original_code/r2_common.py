import sys
sys.dont_write_bytecode=True
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import datetime as dt,hashlib,json
import numpy as np
import pandas as pd
from scipy.special import ndtr
ROOT=Path(__file__).resolve().parents[1];OUT=Path(os.environ.get('R2_OUTPUT',str(ROOT/'out/R2')));SEED=20261007;REPS=1999
TENORS=['10min-8h','8-24h','1-3d','3-7d','7-30d']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def log(message):
 with (ROOT/'logs/RUNLOG_R2.md').open('a') as f:f.write(f'- {dt.datetime.now(dt.timezone.utc).isoformat()} {message}\n')
 print(message,flush=True)
def csv(data,name):
 frame=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data);frame.to_csv(OUT/name,index=False);return frame
def black(F,K,w,kind):
 F=np.asarray(F,float);K=np.asarray(K,float);w=np.maximum(np.asarray(w,float),1e-24)
 s=np.sqrt(w);d1=(np.log(F/K)+w/2)/s;d2=d1-s
 return np.where(np.asarray(kind)=='call',F*ndtr(d1)-K*ndtr(d2),K*ndtr(-d2)-F*ndtr(-d1))
def implied_w(F,K,P,kind):
 F=np.asarray(F,float);K=np.asarray(K,float);P=np.asarray(P,float);kind=np.asarray(kind)
 intrinsic=np.where(kind=='call',np.maximum(F-K,0),np.maximum(K-F,0));upper=np.where(kind=='call',F,K)
 good=np.isfinite(F)&np.isfinite(K)&np.isfinite(P)&(F>0)&(K>0)&(P>intrinsic)&(P<upper)
 lo=np.zeros(len(F));hi=np.full(len(F),64.)
 for _ in range(60):
  mid=(lo+hi)/2;less=black(F,K,mid,kind)<P;lo=np.where(less,mid,lo);hi=np.where(less,hi,mid)
 w=(hi+lo)/2;w[~good]=np.nan
 return w,good
def tenor(hours):return pd.cut(hours,[1/6,8,24,72,168,720],labels=TENORS,include_lowest=True).astype('string')
def config(v=3,window=1000,trim=False,holiday=False,lag=0):return f'v{v}_w{window}_trim{int(trim)}_hol{int(holiday)}_lag{lag}'
PRIMARY=config()
def source_freeze():
 dest=ROOT/'logs/r2_first_round_manifest.csv'
 if dest.exists():return
 rows=[]
 # SHA freeze for all first-round code and outputs, input panels and original report.
 for folder in ['code','out','data_clean']:
  for p in (ROOT/folder).rglob('*'):
   rel=p.relative_to(ROOT)
   if not p.is_file() or p.name.startswith('._') or '__pycache__' in p.parts or 'R2' in p.parts or p.name.startswith('r2_'):continue
   s=p.stat();full=folder in ['code','out'] or len(rel.parts)==2
   rows.append(dict(path=str(rel),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(p) if full else '',verification='sha256' if full else 'stat_readonly_minute_part'))
 p=ROOT/'RESULTS_FOR_CLAUDE.md';s=p.stat();rows.append(dict(path=p.name,bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(p),verification='sha256'))
 pd.DataFrame(rows).to_csv(dest,index=False)
 log(f'第一轮只读清单冻结：{len(rows)} 个文件；代码/表/根面板/报告SHA256，分钟分区记录大小和mtime。')
def dm(difference,lag=7,horizon=1,one_sided=False):
 from scipy.stats import t
 x=np.asarray(difference,float);x=x[np.isfinite(x)];n=len(x)
 if n<max(20,2*lag):return np.nan,np.nan
 u=x-x.mean();variance=np.dot(u,u)/n
 for j in range(1,min(lag,n-1)+1):variance+=2*(1-j/(lag+1))*np.dot(u[j:],u[:-j])/n
 if variance<=0:return np.nan,np.nan
 correction=np.sqrt(max((n+1-2*horizon+horizon*(horizon-1)/n)/n,0));stat=x.mean()/np.sqrt(variance/n)*correction
 return float(stat),float(t.sf(stat,n-1) if one_sided else 2*t.sf(abs(stat),n-1))
def bootstrap_day(frame,base='err0',alternative='err1',date='date',reps=REPS):
 """Circular blocks of seven consecutive calendar days, retaining empty days."""
 d=frame.groupby(date)[[base,alternative]].agg(['sum','count']);d.index=pd.to_datetime(d.index)
 idx=pd.date_range(d.index.min(),d.index.max(),freq='D');d=d.reindex(idx,fill_value=0);n=len(d)
 if n<2:return np.nan,np.nan
 arr=np.column_stack([d[(base,'sum')],d[(alternative,'sum')],d[(base,'count')]])
 rng=np.random.default_rng(SEED);means=[];block=min(7,n)
 for _ in range(reps):
  starts=rng.integers(0,n,size=int(np.ceil(n/block)));indices=(starts[:,None]+np.arange(block))%n;total=arr[indices.ravel()[:n]].sum(0)
  if total[0]>0 and total[2]>0:means.append(100*(1-total[1]/total[0]))
 return tuple(np.quantile(means,[.025,.975])) if means else (np.nan,np.nan)
