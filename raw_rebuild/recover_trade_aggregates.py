import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import sys,json
import pandas as pd,numpy as np
from concurrent.futures import ProcessPoolExecutor
S=Path(os.environ['ONEDAY_RAW_PROJECT']).resolve();P=Path(__file__).resolve().parents[1];sys.path.insert(0,str(P/'raw_rebuild'))
from black import implied_w
META={}
def init():
 for c in ['BTC','ETH']:
  m=pd.read_parquet(S/f'data_clean/option_instruments_{c}.parquet');p=S/f'out/R2/instrument_supplements_{c}.parquet'
  if p.exists():m=pd.concat([m,pd.read_parquet(p)],ignore_index=True)
  META[c]=m.drop_duplicates('instrument_name')[['instrument_name','strike','expiration_timestamp','option_type','creation_timestamp']]
def run(job):
 c,date=job;date=pd.Timestamp(date);start=(date+pd.Timedelta(hours=8)).value//10**6;end=start+3600000;expiry=(date+pd.Timedelta(days=1,hours=8)).value//10**6
 f=S/f'out/R2/raw/{c}/{date:%Y-%m-%d}/trades.parquet';d=pd.read_parquet(f,filters=[('timestamp','>=',start),('timestamp','<',end)])
 for col in ['block_trade_id','block_rfq_id','combo_id','combo_trade_id']:
  if col in d:d=d[d[col].isna()|d[col].isin(['',0,False])]
 d=d.merge(META[c],on='instrument_name',how='inner',validate='many_to_one');d=d[(d.expiration_timestamp==expiry)&(d.creation_timestamp<=d.timestamp)&(d.price>0)&(d.index_price>0)]
 if not len(d):return []
 d['F']=pd.to_numeric(d.get('underlying_price',d.index_price),errors='coerce');d['F']=d.F.where(d.F>0,d.index_price);d=d[np.abs(np.log(d.strike/d.F))<=.05].copy()
 w,valid=implied_w(d.F,d.strike,d.price*d.index_price,d.option_type);tau=(d.expiration_timestamp-d.timestamp)/(365*86400000);d['sigma']=np.sqrt(w/tau);d=d[valid].copy();mw,valid=implied_w(d.F,d.strike,d.mark_price*d.index_price,d.option_type);d['mark_sigma']=np.sqrt(mw/((d.expiration_timestamp-d.timestamp)/(365*86400000)));d=d[valid].copy();d['difference']=100*(d.sigma-d.mark_sigma)
 rows=[]
 for side,z in [('all',d),('buy',d[d.direction=='buy']),('sell',d[d.direction=='sell'])]:
  if len(z):rows.append(dict(currency=c,date=date,side=side,trade_n=len(z),difference_sum=z.difference.sum(),difference_mean=z.difference.mean(),iv_median=z.sigma.median(),mark_iv_median=z.mark_sigma.median()))
 return rows
if __name__=='__main__':
 jobs=[]
 for c in ['BTC','ETH']:
  state=json.loads((S/f'out/R2/download_{c}.json').read_text());jobs.extend((c,str(d.date())) for d in pd.date_range(state['continuous_start'],state['latest']))
 rows=[]
 with ProcessPoolExecutor(max_workers=4,initializer=init) as pool:
  for i,part in enumerate(pool.map(run,jobs,chunksize=8)):
   rows.extend(part)
   if (i+1)%500==0:print('days',i+1,'/',len(jobs),flush=True)
 d=pd.DataFrame(rows);d.to_parquet(P/'data/atm_trade_daily_by_side.parquet',index=False,compression='zstd',compression_level=15);print('COMPLETE',d.shape,flush=True)
