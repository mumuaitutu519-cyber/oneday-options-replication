#!/usr/bin/env python3
from r2_common import *
import argparse,gzip
CLEAN_CODE_SHA=sha(Path(__file__))
def supplement_metadata(currency,metadata,names):
 from r2_hedge_data import fetch
 rows=[]
 for name in names:
  try:
   obj=fetch('get_instrument',dict(instrument_name=name),'instrument_'+name,domain='history.deribit.com');rows.append(obj)
  except Exception as e:log(f'合约元数据补取失败 {name}: {e}')
 if rows:
  extras=pd.DataFrame(rows);fields=[c for c in metadata.columns if c in extras];extras=extras[fields]
  dest=OUT/f'instrument_supplements_{currency}.parquet';prior=pd.read_parquet(dest) if dest.exists() else pd.DataFrame();pd.concat([prior,extras],ignore_index=True).drop_duplicates('instrument_name').to_parquet(dest,index=False);metadata=pd.concat([metadata,extras],ignore_index=True).drop_duplicates('instrument_name')
 return metadata
def clean_day(currency,day,metadata):
 folder=OUT/'raw'/currency/day;dest=OUT/'clean'/currency/(day+'.parquet');audit=dest.with_suffix('.audit.json')
 m=json.loads((folder/'COMPLETE.json').read_text());raw=folder/'trades.parquet'
 if sha(raw)!=m['parquet_sha256']:raise ValueError(f'Raw day hash mismatch {currency}/{day}')
 if dest.exists() and audit.exists():
  a=json.loads(audit.read_text())
  if a['raw_sha256']==m['parquet_sha256'] and a.get('clean_code_sha256')==CLEAN_CODE_SHA:return a
 d=pd.read_parquet(raw);counts=dict(currency=currency,date=day,raw_rows=len(d),duplicate_rows=int(d.trade_id.duplicated().sum()) if len(d) else 0)
 dest.parent.mkdir(parents=True,exist_ok=True)
 if not len(d):
  pd.DataFrame().to_parquet(dest,index=False);a=dict(**counts,clean_rows=0,raw_sha256=m['parquet_sha256'],clean_code_sha256=CLEAN_CODE_SHA);audit.write_text(json.dumps(a));return a
 d=d.drop_duplicates('trade_id');supplement=OUT/f'instrument_supplements_{currency}.parquet'
 if supplement.exists():metadata=pd.concat([metadata,pd.read_parquet(supplement)],ignore_index=True).drop_duplicates('instrument_name')
 missing_names=sorted(set(d.instrument_name)-set(metadata.instrument_name))
 if missing_names:metadata=supplement_metadata(currency,metadata,missing_names)
 d=d.merge(metadata,on='instrument_name',how='left',validate='many_to_one',suffixes=('','_meta'))
 counts['metadata_missing']=int(d.strike.isna().sum());counts['expiration_not_08utc']=int(((pd.to_datetime(d.expiration_timestamp,unit='ms',utc=True).dt.hour!=8)&d.expiration_timestamp.notna()).sum())
 missing=d.loc[d.strike.isna(),'instrument_name'].unique()
 if len(missing):csv(pd.DataFrame({'currency':currency,'date':day,'instrument_name':missing}),f'metadata_missing_{currency}_{day}.csv')
 for c in ['price','iv','index_price','underlying_price']:
  if c not in d:d[c]=np.nan
  d[c]=pd.to_numeric(d[c],errors='coerce')
 use=np.isfinite(d.underlying_price)&(d.underlying_price>0);d['F']=np.where(use,d.underlying_price,d.index_price);d['forward_source']=np.where(use,'underlying_price','index_price')
 d['actual_usd']=d.price*d.index_price;d['remaining_hours']=(d.expiration_timestamp-d.timestamp)/3600000
 d['tau']=d.remaining_hours/(24*365);d['log_moneyness']=np.log(d.strike/d.F)
 d['block']=False;d['combo']=False
 for c in ['block_trade_id','block_rfq_id','block_id']:
  if c in d:d['block']|=d[c].notna()&~d[c].isin(['',0,False])
 for c in ['combo_trade_id','combo_id','combo_order_id']:
  if c in d:d['combo']|=d[c].notna()&~d[c].isin(['',0,False])
 valid=np.isfinite(d.price)&(d.price>0)&np.isfinite(d.iv)&(d.iv>=0)&np.isfinite(d.index_price)&(d.index_price>0)&np.isfinite(d.F)&(d.F>0)&np.isfinite(d.strike)&(d.strike>0)&d.option_type.isin(['call','put'])&d.direction.isin(['buy','sell'])&(d.tau>0)
 counts['metadata_creation_after_trade']=int((d.creation_timestamp>d.timestamp).sum());valid&=d.creation_timestamp<=d.timestamp
 counts['field_invalid']=int((~valid).sum());d=d[valid].copy()
 d['w'],good=implied_w(d.F,d.strike,d.actual_usd,d.option_type);counts['black_price_bounds_invalid']=int((~good).sum());d=d[good].copy()
 d['sigma']=np.sqrt(d.w/d.tau);d['iv_deviation_points']=100*d.sigma-d.iv
 d['inversion_residual_usd']=black(d.F,d.strike,d.w,d.option_type)-d.actual_usd
 counts.update(valid_inversions=len(d),index_fallback=int(d.forward_source.eq('index_price').sum()),underlying_source=int(d.forward_source.eq('underlying_price').sum()),block_rows=int(d.block.sum()),combo_rows=int(d.combo.sum()),residual_max_usd=float(d.inversion_residual_usd.abs().max()) if len(d) else None)
 # All inversion diagnostics retain their true values before ATM/tenor filters.
 diagnostics=d[['trade_id','timestamp','remaining_hours','sigma','iv','iv_deviation_points','forward_source','inversion_residual_usd']].copy();diagnostics['date']=day
 diagnostics.to_parquet(dest.with_suffix('.iv.parquet'),index=False)
 d=d[(d.remaining_hours>=1/6)&(d.remaining_hours<=720)&(d.log_moneyness.abs()<=.10)].copy();d['currency']=currency;d['date']=pd.to_datetime(d.timestamp,unit='ms',utc=True).dt.tz_localize(None).dt.floor('D');d['hour_bucket']=pd.to_datetime(d.timestamp,unit='ms',utc=True).dt.tz_localize(None).dt.floor('h');d['tenor']=tenor(d.remaining_hours)
 columns=['currency','date','hour_bucket','timestamp','trade_id','instrument_name','expiration_timestamp','strike','option_type','direction','amount','F','index_price','underlying_price','forward_source','price','actual_usd','iv','w','sigma','tau','remaining_hours','log_moneyness','block','combo','iv_deviation_points','inversion_residual_usd']
 d=d[[c for c in columns if c in d]].sort_values(['timestamp','trade_id']);d.to_parquet(dest,index=False);counts['clean_rows']=len(d);counts['main_rows']=int(((d.log_moneyness.abs()<=.05)&~d.block&~d.combo).sum())
 a=dict(**counts,raw_sha256=m['parquet_sha256'],clean_code_sha256=CLEAN_CODE_SHA,clean_sha256=sha(dest));audit.write_text(json.dumps(a,ensure_ascii=False,indent=2));return a
def summaries(currency,days):
 audits=[];buckets=[];ivparts=[];yearcounts=[]
 for day in days:
  p=OUT/'clean'/currency/(day+'.parquet');audits.append(json.loads(p.with_suffix('.audit.json').read_text()))
  d=pd.read_parquet(p)
  if len(d):
   for band in [.02,.05,.10]:
    for include in [False,True]:
     g=d[d.log_moneyness.abs()<=band]
     if not include:g=g[~g.block&~g.combo]
     b=g.groupby(['expiration_timestamp','hour_bucket']).agg(timestamp=('timestamp','median'),n=('w','size'),w=('w','median'),F=('F','median'),remaining_hours=('remaining_hours','median')).reset_index();b['currency']=currency;b['band']=band;b['include_block']=include;b['tenor']=tenor(b.remaining_hours);b['date']=pd.to_datetime(b.hour_bucket).dt.floor('D');buckets.append(b)
   main=d[(d.log_moneyness.abs()<=.05)&~d.block&~d.combo].copy();main['tenor']=tenor(main.remaining_hours);g=main.groupby('tenor',observed=True).size().reset_index(name='n');g['currency']=currency;g['year']=int(day[:4]);yearcounts.append(g)
  q=p.with_suffix('.iv.parquet')
  if q.exists():
   z=pd.read_parquet(q);z['year']=int(day[:4]);z['currency']=currency;ivparts.append(z)
 csv(audits,f'cleaning_{currency}.csv')
 b=pd.concat(buckets,ignore_index=True) if buckets else pd.DataFrame();b.to_parquet(OUT/f'buckets_{currency}.parquet',index=False)
 if ivparts:
  z=pd.concat(ivparts,ignore_index=True);rows=[]
  for year,g in [('all',z)]+[(str(y),g) for y,g in z.groupby('year')]:
   x=g.iv_deviation_points.to_numpy();rows.append(dict(currency=currency,year=year,n=len(g),mean_points=x.mean(),median_points=np.median(x),mae_points=np.mean(abs(x)),p01_points=np.quantile(x,.01),p05_points=np.quantile(x,.05),p95_points=np.quantile(x,.95),p99_points=np.quantile(x,.99),residual_max_usd=g.inversion_residual_usd.abs().max(),index_fallback=int(g.forward_source.eq('index_price').sum())))
  csv(rows,f'iv_convention_{currency}.csv')
  z.to_parquet(OUT/f'iv_diagnostics_{currency}.parquet',index=False)
 if yearcounts:csv(pd.concat(yearcounts).groupby(['currency','year','tenor'],observed=True).n.sum().reset_index(),f'trade_counts_{currency}_year_tenor.csv')
 summary=json.loads((OUT/f'download_{currency}.json').read_text());selected=summary['continuous_days'];main=b[(b.band==.05)&~b.include_block&(b.n>=3)&(b.remaining_hours<=24)] if len(b) else b
 gate=dict(currency=currency,continuous_days=selected,start=summary['continuous_start'],end=summary['latest'],short_atm_hour_buckets=len(main),required_buckets=20000 if currency=='BTC' else 10000,iv_check_completed=(OUT/f'iv_convention_{currency}.csv').exists(),all_history_complete=summary['all_history_complete'],gate_pass=selected>=1000 and len(main)>=(20000 if currency=='BTC' else 10000) and (OUT/f'iv_convention_{currency}.csv').exists())
 return gate
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--incremental',action='store_true');args=ap.parse_args();gate=[]
 for currency in ['BTC','ETH']:
  metadata=pd.read_parquet(ROOT/f'data_clean/option_instruments_{currency}.parquet').drop(columns='currency',errors='ignore')
  supplement=OUT/f'instrument_supplements_{currency}.parquet'
  if supplement.exists():metadata=pd.concat([metadata,pd.read_parquet(supplement)],ignore_index=True).drop_duplicates('instrument_name')
  summary=OUT/f'download_{currency}.json'
  if summary.exists():
   z=json.loads(summary.read_text());days=[d.strftime('%Y-%m-%d') for d in pd.date_range(z['continuous_start'],z['latest'])]
  elif args.incremental:days=sorted(p.parent.name for p in (OUT/'raw'/currency).glob('*/COMPLETE.json'))
  else:raise FileNotFoundError(f'Download not complete {currency}')
  for i,day in enumerate(days):
   clean_day(currency,day,metadata)
   if (i+1)%100==0:log(f'真实成交清洗 {currency} {i+1}/{len(days)} 日')
  if not args.incremental:gate.append(summaries(currency,days))
 if gate:csv(gate,'gate1_data.csv');log(f'数据闸门：{gate}')
if __name__=='__main__':main()
