import sys,os
sys.dont_write_bytecode=True
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
from pathlib import Path
import datetime,json,hashlib
import numpy as np,pandas as pd
from scipy import stats,special
import statsmodels.api as sm
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('R4_OUTPUT_ROOT',str(ROOT/'out/R4')))
FORCE=os.environ.get('R4_FORCE_REBUILD')=='1'
SEED=20261007;EPS=1e-12;CUTOFF=pd.Timestamp('2026-10-07');REPS=1999
ASSETS=['BTC','ETH','SOL','BNB','XRP','DOGE','ADA','LINK','AVAX','LTC']
from forecast_core import qlike,hac_mean,dm_test,cw_test,mcs,block_indices
from r3_common import grid,daily
from r3_clock import Clock
from r3_options import day,rawday,metadata,funding_events,delivery_prices

def log(message):
 p=ROOT/'logs/RUNLOG_R4.md';p.parent.mkdir(exist_ok=True)
 with p.open('a') as f:f.write('- '+datetime.datetime.now(datetime.timezone.utc).isoformat()+' '+message+'\n')
 print(message,flush=True)
def csv(rows,path):
 d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);p=OUT/path;p.parent.mkdir(parents=True,exist_ok=True);d.to_csv(p,index=False);return d

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()

# These R2 diagnostics are never read or imported by an R4 calculation. The
# broad archive snapshot still records their drift; it is separate from the
# integrity of every empirical input used in R4.
NONINPUT_R2={
 'code/r2_checks.py',
 'out/R2/mtm_daily_BTC.parquet','out/R2/mtm_daily_ETH.parquet',
 'out/R2/mtm_points_ETH_v3_w1000_trim0_hol0_lag0.parquet',
 'out/R2/mtm_points_ETH_v3_w1000_trim0_hol0_lag30.parquet',
 'out/R2/mtm_comparison.csv','out/R2/mtm_k_weekend.csv',
 'out/R2/mtm_by_year.csv','out/R2/mtm_weekend_all.csv',
 'out/R2/weekend_strategy_eligibility_BTC.csv',
 'out/R2/weekend_strategy_points_BTC.parquet',
 'out/R2/weekend_strategy_eligibility_ETH.csv',
 'out/R2/weekend_strategy_points_ETH.parquet',
 'out/R2/decay_hedging_summary.csv','out/R2/weekend_strategy_summary.csv',
 'out/R2/weekend_strategy_year.csv'}

def source_drift():
 manifest=pd.read_csv(OUT/'source_manifest.csv').fillna('');rows=[]
 for r in manifest.itertuples(index=False):
  p=ROOT/r.path;s=p.stat() if p.exists() else None;digest=sha(p) if s and r.sha256 else ''
  changed=s is None or s.st_size!=r.bytes or s.st_mtime_ns!=r.mtime_ns or (r.sha256 and digest!=r.sha256)
  if changed:rows.append(dict(path=r.path,baseline_bytes=r.bytes,current_bytes=s.st_size if s else np.nan,baseline_mtime_ns=r.mtime_ns,current_mtime_ns=s.st_mtime_ns if s else np.nan,baseline_sha256=r.sha256,current_sha256=digest,affects_R4=r.path not in NONINPUT_R2,classification='R4_dependency_changed' if r.path not in NONINPUT_R2 else 'unused_R2_diagnostic',evidence='source_process_evidence.json'))
 d=pd.DataFrame(rows,columns=['path','baseline_bytes','current_bytes','baseline_mtime_ns','current_mtime_ns','baseline_sha256','current_sha256','affects_R4','classification','evidence']);csv(d,'source_changes.csv');return d
def ci(x,lag=7):
 v=np.asarray(x,float);v=v[np.isfinite(v)];n=len(v)
 if not n:return dict(n=0,mean=np.nan,ci_low=np.nan,ci_high=np.nan,p=np.nan,se=np.nan)
 m,se,t=hac_mean(v,lag)
 if not np.isfinite(m):m=float(v.mean())
 return dict(n=n,mean=m,ci_low=m-1.96*se,ci_high=m+1.96*se,p=2*stats.norm.sf(abs(t)),se=se)
def loss_compare(y,base,pred,h=1):
 z=pd.DataFrame({'y':y,'base':base,'pred':pred}).replace([np.inf,-np.inf],np.nan).dropna();z=z[(z>0).all(1)];r={'n':len(z),'status':'estimated' if len(z)>=30 else 'insufficient'}
 if len(z)<30:return r
 for name,a,b in [('qlike',qlike(z.y,z.base),qlike(z.y,z.pred)),('mse',(z.y-z.base)**2,(z.y-z.pred)**2)]:
  den=float(a.mean());ratio=float(b.mean()/den);value=100*(1-ratio);influence=100*(ratio*np.asarray(a)-np.asarray(b))/den;co=ci(influence,max(7,2*h));t,p=dm_test(a,b,h)
  r.update({name+'_gain':value,name+'_low':value-1.96*co['se'],name+'_high':value+1.96*co['se'],name+'_dm_t':t,name+'_dm_p':p,name+'_base_loss':den,name+'_pred_loss':float(b.mean())})
 r['cw_t'],r['cw_p']=cw_test(z.y,z.base,z.pred,h);return r

def plot_init():
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 return plt

def rv_interval(a,start,end):
 lo=a.index.searchsorted(start,side='right');hi=a.index.searchsorted(end,side='right');z=a.rv.iloc[lo:hi];need=int((end-start).total_seconds()/300)
 return float(z.sum()) if len(z)==need and z.notna().all() else np.nan
