from common import *
import argparse,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from model_engine import predict_panel

def worker(cfg):
 start=time.monotonic();c=cfg['currency'];spec=cfg['spec'];dest=WORK/f'predictions_{c}_{spec}.parquet'
 if dest.exists():
  # Restore the original raw-benchmark start mask when resuming a completed fit.
  d=pd.read_parquet(dest);a=load_input(cfg['key']);v=a.DVOL_raw.copy()
  if 'DVOL_raw' in cfg['models'] or cfg['kind']=='R4':v.iloc[:(365 if cfg['window']=='expanding' else int(cfg['window']))]=np.nan
  d['DVOL_raw']=v;d.to_parquet(dest,index=False)
  return dict(key=cfg['key'],seconds=0,resumed=True)
 if cfg['kind']=='R4':
  from old_model_engine import run_spec
  run_spec(c,8,1,1000,'log')
 else:
  d=load_input(cfg['key'])
  predict_panel(d,c,spec,window=cfg['window'],models=cfg['models'],risk=cfg['risk'],minimum=cfg['minimum'])
 return dict(key=cfg['key'],seconds=time.monotonic()-start,resumed=False)

def main():
 cfg=json.loads((DATA/'forecast_specs.json').read_text());cfg.sort(key=lambda c:0 if c['kind'] in ['main','measurement','R4'] else 1);rows=[]
 with ProcessPoolExecutor(max_workers=int(os.environ.get('REPLICATION_WORKERS','4'))) as pool:
  futures={pool.submit(worker,c):c['key'] for c in cfg}
  for f in as_completed(futures):
   r=f.result();rows.append(r);print('DONE',r,flush=True);pd.DataFrame(rows).to_csv(OUT/'full_estimation_timing.csv',index=False)
 # Compare independently refitted forecasts against the historical issued forecasts.
 checks=[]
 for c in cfg:
  a=pd.read_parquet(WORK/f'predictions_{c["key"]}.parquet');b=pd.read_parquet(DATA/'forecast_cache'/(c['key']+'.parquet'))
  for m in list(dict.fromkeys(c['models']+['DVOL_raw'])):
   if m not in a or m not in b:continue
   x=a[m].to_numpy();y=b[m].to_numpy();same=np.isclose(x,y,rtol=1e-9,atol=1e-12,equal_nan=True)
   checks.append(dict(key=c['key'],model=m,n=len(x),equal=int(same.sum()),max_abs=float(np.nanmax(np.abs(x-y))) if np.isfinite(x-y).any() else 0))
 pd.DataFrame(checks).to_csv(ROOT/'verify/FULL_FORECAST_RECONCILIATION.csv',index=False)
if __name__=='__main__':main()
