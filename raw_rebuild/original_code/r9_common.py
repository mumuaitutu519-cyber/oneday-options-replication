import os,sys
sys.dont_write_bytecode=True
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']: os.environ[k]='1'
from pathlib import Path
import json,hashlib,datetime
from functools import lru_cache
import numpy as np,pandas as pd
from scipy import stats,special
ROOT=Path(__file__).resolve().parents[1]; O=ROOT/'out/R9'; O.mkdir(exist_ok=True)
R5=ROOT/'out/R5/chen'; R4=ROOT/'out/R4/chen'; TENORS=[1,2,3,7,14,30]; SEED=20261013; REPS=4999

def log(s):
 with (ROOT/'logs/RUNLOG_R9.md').open('a') as f:f.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+str(s)+'\n')
 with (O/'log.md').open('a') as f:f.write(str(s)+'\n')
 print(s,flush=True)
def save(d,name):
 d=d if isinstance(d,pd.DataFrame) else pd.DataFrame(d);d.to_csv(O/name,index=False);return d

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
@lru_cache(maxsize=2)
def weights(n):
 rng=np.random.default_rng(SEED);idx=np.empty((REPS,n),dtype=np.int32);idx[:,0]=rng.integers(n,size=REPS)
 for j in range(1,n):idx[:,j]=np.where(rng.random(REPS)<1/7,rng.integers(n,size=REPS),(idx[:,j-1]+1)%n)
 return np.bincount((idx+np.arange(REPS)[:,None]*n).ravel(),minlength=REPS*n).reshape(REPS,n).astype(np.int16)
def summary(s,h=1,boot=True):
 s=pd.Series(s).replace([np.inf,-np.inf],np.nan)
 if not isinstance(s.index,pd.DatetimeIndex):raise ValueError('calendar index required')
 if s.empty:return dict(n=0,mean=np.nan,median=np.nan,se=np.nan,p=np.nan,positive_share=np.nan,hac_lag=max(7,2*h),bootstrap_n=0,status='no_observations')
 s=s.sort_index().reindex(pd.date_range(s.index.min(),s.index.max()));x=s.to_numpy(float);ok=np.isfinite(x);n=ok.sum();v=x[ok];L=max(7,2*h)
 r=dict(n=int(n),mean=np.nan,median=np.nan,se=np.nan,p=np.nan,positive_share=np.nan,hac_lag=L,stationary_block=7,bootstrap_n=int(n))
 if not n:return r
 m=v.mean();u=np.where(ok,x-m,0);V=u@u
 for j in range(1,min(L,len(u)-1)+1):V+=2*(1-j/(L+1))*(u[j:]@u[:-j])
 se=np.sqrt(max(V,0))/n;t=m/se if se else np.nan
 r.update(mean=m,median=np.median(v),se=se,t=t,p=2*stats.norm.sf(abs(t)),p_greater=stats.norm.sf(t),p_less=stats.norm.cdf(t),ci_low=m-1.96*se,ci_high=m+1.96*se,positive_share=(v>0).mean(),q01=np.quantile(v,.01),q05=np.quantile(v,.05),q25=np.quantile(v,.25),q50=np.median(v),q75=np.quantile(v,.75),q95=np.quantile(v,.95),q99=np.quantile(v,.99))
 if boot and n>=10:
  w=weights(len(x));den=w@ok.astype(float);z=(w@np.where(ok,x,0))/np.where(den>0,den,np.nan);r['boot_low'],r['boot_high']=np.nanquantile(z,[.025,.975]);order=np.flatnonzero(ok);order=order[np.argsort(x[order])];cw=np.cumsum(w[:,order],axis=1);i=(cw>=np.ceil(den/2)[:,None]).argmax(1);j=(cw>=np.floor(den/2+1)[:,None]).argmax(1);med=(x[order[i]]+x[order[j]])/2;med[den==0]=np.nan;r['median_boot_low'],r['median_boot_high']=np.nanquantile(med,[.025,.975])
 return r

def black(F,K,w,kind):
 F=np.asarray(F,float);K=np.asarray(K,float);w=np.maximum(np.asarray(w,float),1e-24);s=np.sqrt(w);d1=(np.log(F/K)+w/2)/s;d2=d1-s
 return np.where(np.asarray(kind)=='call',F*special.ndtr(d1)-K*special.ndtr(d2),K*special.ndtr(-d2)-F*special.ndtr(-d1))
def implied(F,K,P,kind):
 F=np.asarray(F,float);K=np.asarray(K,float);P=np.asarray(P,float);kind=np.asarray(kind);intr=np.where(kind=='call',np.maximum(F-K,0),np.maximum(K-F,0));cap=np.where(kind=='call',F,K);good=np.isfinite(P)&(F>0)&(K>0)&(P>intr)&(P<cap);lo=np.zeros(len(F));hi=np.full(len(F),64.)
 for _ in range(60):
  mid=(lo+hi)/2;less=black(F,K,mid,kind)<P;lo=np.where(less,mid,lo);hi=np.where(less,hi,mid)
 w=(hi+lo)/2;w[~good]=np.nan;return w

def event_flags(d,start='observed_at',end='target_end'):
 ev=pd.read_csv(ROOT/'out/R3/shared/events_publication.csv');ev['at']=pd.to_datetime(ev.datetime_utc,utc=True).dt.tz_localize(None);ev['known']=pd.to_datetime(ev.known_at,utc=True,errors='coerce').dt.tz_localize(None)
 a=pd.to_datetime(d[start]);b=pd.to_datetime(d[end]);out=d.copy()
 for typ in ['CPI','NFP','FOMC']:
  rows=ev[ev.type==typ];ts=np.sort(rows['at'].to_numpy());out[typ]=np.searchsorted(ts,b,side='right')>np.searchsorted(ts,a,side='right');out[typ+'_known']=False
  for e in rows.itertuples():
   if pd.notna(e.known):out.loc[(a<e.at)&(b>=e.at)&(a>=e.known)&bool(e.ex_ante_supported),typ+'_known']=True
 # Any positive-duration intersection with a Saturday/Sunday, using UTC as all prices.
 saturday=a.dt.floor('D')+pd.to_timedelta((5-a.dt.dayofweek)%7,unit='D')
 out['weekend']=((a.dt.dayofweek>=5)|(saturday<b))&a.notna()&b.notna()&(b>a)
 out['macro']=out[['CPI','NFP','FOMC']].any(axis=1);out['rest']=~out[['CPI','NFP','FOMC','weekend']].any(axis=1)
 out['exclusive']=np.select([out.CPI,out.NFP,out.FOMC,out.weekend],['CPI','NFP','FOMC','weekend'],default='rest');return out
