#!/usr/bin/env python3
"""Full daily historical option trades; resumable inclusive pagination."""
import sys
sys.dont_write_bytecode=True
import argparse, concurrent.futures, datetime as dt, gzip, hashlib, json, os, threading, time
from pathlib import Path
import requests
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'out/R2/raw'
LOCK=threading.Lock(); RATE=threading.Lock(); NEXT=0.0
HTTP_SLOTS=threading.BoundedSemaphore(2);LOCAL=threading.local()
URL='https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time'
END=pd.Timestamp('2026-10-07',tz='UTC')
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def write(p,obj):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.part');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(p)
def log(s):
 with LOCK:
  with (ROOT/'logs/RUNLOG_R2.md').open('a') as f:f.write(f'- {dt.datetime.now(dt.timezone.utc).isoformat()} {s}\n')
  print(s,flush=True)
def request(session,params,path):
 global NEXT
 if path.exists():
  with gzip.open(path,'rt') as f:obj=json.load(f)
  if 'result' in obj and 'error' not in obj:return obj['result']
 if os.environ.get('R2_OFFLINE')=='1':raise FileNotFoundError(f'Offline raw API response missing: {path}')
 for attempt in range(8):
  with RATE:
   wait=max(0,NEXT-time.monotonic())
   if wait:time.sleep(wait)
   NEXT=time.monotonic()+.35
  started=dt.datetime.now(dt.timezone.utc).isoformat();status=None;body=b'';error=None
  try:
   with HTTP_SLOTS:response=session.get(URL,params=params,timeout=50)
   status=response.status_code;body=response.content
   obj=response.json();response.raise_for_status()
   if 'error' in obj or 'result' not in obj:raise ValueError(str(obj.get('error','no_result')))
  except Exception as e:error=repr(e)
  path.parent.mkdir(parents=True,exist_ok=True)
  receipt={'started':started,'url':URL,'params':params,'status':status,'attempt':attempt,'bytes':len(body),'sha256_body':hashlib.sha256(body).hexdigest(),'error':error,'raw_path':str(path.relative_to(ROOT))}
  with LOCK:
   with (ROOT/'logs/r2_requests.jsonl').open('a') as f:f.write(json.dumps(receipt)+'\n')
  if error is None:
   tmp=path.with_suffix('.part')
   with tmp.open('wb') as f:
    with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:z.write(body)
   tmp.replace(path);return obj['result']
  failure=path.with_name(path.name+f'.failure_{time.time_ns()}')
  failure.write_bytes(body);time.sleep(min(60,2**attempt))
 raise RuntimeError(f'{params}: {error}')
def earliest(currency,session):
 params=dict(currency=currency,kind='option',start_timestamp=1262304000000,end_timestamp=int(END.timestamp()*1000)-1,count=1,sorting='asc',include_old='true')
 r=request(session,params,RAW/f'earliest_{currency}.json.gz');trades=r.get('trades',[])
 if not trades:raise RuntimeError(f'No historical trades {currency}')
 return pd.Timestamp(trades[0]['timestamp'],unit='ms',tz='UTC').floor('D')
def day(currency,date,session):
 folder=RAW/currency/date.strftime('%Y-%m-%d');done=folder/'COMPLETE.json';pq=folder/'trades.parquet'
 if done.exists():
  m=json.loads(done.read_text())
  if pq.exists() and sha(pq)==m['parquet_sha256']:return m
 start=int(date.timestamp()*1000);end=start+86400000-1;cursor=start;rows=[];seen=set();page=0
 while cursor<=end:
  params=dict(currency=currency,kind='option',start_timestamp=cursor,end_timestamp=end,count=10000,sorting='asc',include_old='true')
  path=folder/f'page_{page:05d}_{cursor}.json.gz';result=request(session,params,path)
  trades=result.get('trades',[])
  for r in trades:
   if not start<=r['timestamp']<=end:raise ValueError('API returned out-of-day timestamp')
   key=str(r['trade_id'])
   if key not in seen:seen.add(key);rows.append(r)
  page+=1
  if not trades or not result.get('has_more',False):break
  nxt=max(r['timestamp'] for r in trades)
  if nxt<=cursor:raise ValueError(f'Pagination stalled; do not skip timestamp {currency}/{date}/{cursor}')
  cursor=nxt # inclusive: simultaneous boundary trades retained and de-duplicated
 d=pd.DataFrame(rows)
 if len(d):d=d.sort_values(['timestamp','trade_id']).reset_index(drop=True)
 else:d=pd.DataFrame(columns=['timestamp','trade_id','instrument_name','price','iv','index_price','direction'])
 tmp=folder/'trades.parquet.part';d.to_parquet(tmp,index=False);tmp.replace(pq)
 m=dict(currency=currency,date=date.strftime('%Y-%m-%d'),complete=True,rows=len(d),pages=page,start_ms=start,end_ms=end,parquet_sha256=sha(pq),first_ms=int(d.timestamp.min()) if len(d) else None,last_ms=int(d.timestamp.max()) if len(d) else None,pagination='inclusive_timestamp_overlap_unique_trade_id')
 write(done,m);return m
def worker(currency,deadline):
 session=requests.Session();session.headers['User-Agent']='JOD variance-clock reproducible research'
 first=earliest(currency,session);results=[];failed=[]
 dates=pd.date_range(first,END-pd.Timedelta(days=1),freq='D')[::-1]
 def task(date):
  if time.time()>=deadline:return None,None
  if not hasattr(LOCAL,'session'):
   LOCAL.session=requests.Session();LOCAL.session.headers['User-Agent']='JOD variance-clock reproducible research'
  try:return day(currency,date,LOCAL.session),None
  except Exception as e:return None,dict(currency=currency,date=str(date.date()),error=repr(e))
 # Two adjacent daily files can progress independently; global HTTP_SLOTS
 # enforces at most two active requests across BOTH currency workers.
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as days_pool:
  for i,(m,error) in enumerate(days_pool.map(task,dates)):
   if m is None and error is None:log(f'{currency}: 36小时截止，保留最新连续完整区间');break
   if error:failed.append(error);log(f'{currency}/{error["date"]} 下载失败，继续其他日：{error["error"]}')
   else:results.append(m)
   if (i+1)%25==0:log(f'{currency} 连续倒序下载：{i+1}/{len(dates)} 日，累计 {sum(x["rows"] for x in results):,} 笔，当前 {dates[i].date()}')
 # Retry failed dates once, preserving all failed request receipts.
 for item in failed[:]:
  if time.time()>=deadline:break
  try:results.append(day(currency,pd.Timestamp(item['date'],tz='UTC'),session));failed.remove(item)
  except Exception:pass
 last=END-pd.Timedelta(days=1);continuous=[];by={x['date']:x for x in results}
 while last.strftime('%Y-%m-%d') in by:
  continuous.append(by[last.strftime('%Y-%m-%d')]);last-=pd.Timedelta(days=1)
 summary=dict(currency=currency,earliest_available=str(first.date()),latest=str((END-pd.Timedelta(days=1)).date()),requested_days=len(dates),complete_days=len(results),continuous_days=len(continuous),continuous_start=continuous[-1]['date'] if continuous else None,all_history_complete=len(results)==len(dates),rows=sum(x['rows'] for x in results),failures=failed)
 write(ROOT/f'out/R2/download_{currency}.json',summary)
 pd.DataFrame(sorted(results,key=lambda x:x['date'])).to_csv(ROOT/f'out/R2/download_manifest_{currency}.csv',index=False)
 log(f'{currency} 下载结束：{json.dumps(summary,ensure_ascii=False)}');return summary
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--hours',type=float,default=36);ap.add_argument('--offline',action='store_true');args=ap.parse_args();RAW.mkdir(parents=True,exist_ok=True)
 if args.offline:os.environ['R2_OFFLINE']='1'
 deadline=time.time()+args.hours*3600
 state=ROOT/'out/R2/download_state.json'
 if state.exists() and not args.offline:deadline=min(deadline,json.loads(state.read_text())['deadline_epoch'])
 elif not state.exists():write(state,dict(started=dt.datetime.now(dt.timezone.utc).isoformat(),deadline_epoch=deadline,count=10000,max_concurrency=2,max_request_rate=1/.35,seed=20261007))
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
  for result in pool.map(lambda c:worker(c,deadline),['BTC','ETH']):print(result,flush=True)
if __name__=='__main__':main()
