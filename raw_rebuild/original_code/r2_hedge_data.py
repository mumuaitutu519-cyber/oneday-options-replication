#!/usr/bin/env python3
from r2_common import *
import argparse,requests,gzip,time
def fetch(method,params,key,offline=False,domain='www.deribit.com'):
 offline=offline or os.environ.get('R2_OFFLINE')=='1'
 p=OUT/'raw/hedge'/f'{key}.json.gz';p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():
  with gzip.open(p,'rt') as f:obj=json.load(f)
  if 'result' in obj and 'error' not in obj:return obj['result']
 if offline:raise FileNotFoundError(p)
 url=f'https://{domain}/api/v2/public/{method}'
 for attempt in range(6):
  body=b'';error=None;status=None
  try:
   r=requests.get(url,params=params,timeout=45);status=r.status_code;body=r.content;r.raise_for_status();obj=r.json()
   if 'error' in obj or 'result' not in obj:raise ValueError(str(obj.get('error','missing_result')))
  except Exception as e:error=repr(e)
  receipt=dict(method=method,params=params,url=url,status=status,error=error,attempt=attempt,sha256_body=hashlib.sha256(body).hexdigest(),path=str(p.relative_to(ROOT)))
  with (ROOT/'logs/r2_hedge_requests.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
  if error is None:
   with p.open('wb') as f:
    with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0,compresslevel=1) as z:z.write(body)
   time.sleep(.5);return obj['result']
  p.with_suffix(f'.failure_{time.time_ns()}').write_bytes(body);time.sleep(min(30,2**attempt))
 raise RuntimeError(error)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--offline',action='store_true');args=ap.parse_args();quality=[]
 for currency in ['BTC','ETH']:
  metadata=fetch('get_instrument',dict(instrument_name=currency+'-PERPETUAL'),f'perpetual_contract_{currency}',args.offline);(OUT/f'perpetual_contract_{currency}.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2))
  rows=[];offset=0
  while True:
   r=fetch('get_delivery_prices',dict(index_name=currency.lower()+'_usd',count=1000,offset=offset),f'delivery_{currency}_{offset}',args.offline);d=r.get('data',[]);rows.extend(d)
   if not d or len(rows)>=r.get('records_total',len(rows)):break
   offset+=len(d)
  delivery=pd.DataFrame(rows).drop_duplicates('date');delivery['expiration_timestamp']=pd.to_datetime(delivery.date).astype('int64')//10**6+8*3600000;delivery.to_parquet(OUT/f'delivery_{currency}.parquet',index=False)
  # Exact exchange perpetual history, covering the complete clock input era.
  candles=[];funding=[]
  for month in pd.date_range('2020-01-01','2026-10-01',freq='MS'):
   end=min(month+pd.offsets.MonthBegin(1),pd.Timestamp('2026-10-07'));startms=int(month.timestamp()*1000);endms=int(end.timestamp()*1000)-1;key=month.strftime('%Y-%m')
   try:
    blocks=[]
    for part,bstart in enumerate(range(startms,endms+1,4000*300000)):
     bend=min(endms,bstart+4000*300000-1);r=fetch('get_tradingview_chart_data',dict(instrument_name=currency+'-PERPETUAL',start_timestamp=bstart,end_timestamp=bend,resolution='5'),f'perpetual5_{currency}_{key}_{part}',args.offline,domain='history.deribit.com');z=pd.DataFrame({c:r.get(c,[]) for c in ['ticks','open','high','low','close','volume']});z['public_ms']=z.ticks+300000;z=z[(z.volume>0)&(z.close>0)&(z.ticks>=bstart)&(z.ticks<=bend)];blocks.append(z)
    z=pd.concat(blocks,ignore_index=True).drop_duplicates('ticks');candles.append(z);quality.append(dict(currency=currency,month=key,kind='perpetual_5minute',rows=len(z),expected=(endms-startms+1)//300000,status='complete_response',missing_rate=1-len(z)/((endms-startms+1)//300000)))
   except Exception as e:quality.append(dict(currency=currency,month=key,kind='perpetual_5minute',rows=0,status='failed',reason=repr(e)))
   try:
    r=fetch('get_funding_rate_history',dict(instrument_name=currency+'-PERPETUAL',start_timestamp=startms-1,end_timestamp=endms),f'funding_full_{currency}_{key}',args.offline);z=pd.DataFrame(r);funding.append(z);quality.append(dict(currency=currency,month=key,kind='funding',rows=len(z),expected=(endms-startms+1)//3600000,status='complete_response'))
   except Exception as e:quality.append(dict(currency=currency,month=key,kind='funding',rows=0,status='failed',reason=repr(e)))
  pd.concat(candles,ignore_index=True).drop_duplicates('ticks').sort_values('ticks').to_parquet(OUT/f'deribit_perpetual_{currency}.parquet',index=False);pd.concat(funding,ignore_index=True).drop_duplicates('timestamp').sort_values('timestamp').to_parquet(OUT/f'deribit_funding_{currency}.parquet',index=False);log(f'Deribit实际对冲价格/结算/资金费 {currency} 完成')
 csv(quality,'hedge_data_quality.csv')
if __name__=='__main__':main()
