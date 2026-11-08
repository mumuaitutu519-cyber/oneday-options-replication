"""Optional public Binance monthly one-minute archive downloader. Not used by run_all.sh."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import argparse,hashlib,datetime,calendar
import requests

def fetch(job):
 market,symbol,month,dest=job;prefix='futures/um' if market=='perpetual' else 'spot';name=f'{symbol}-1m-{month}.zip';url=f'https://data.binance.vision/data/{prefix}/monthly/klines/{symbol}/1m/{name}';p=Path(dest)/market/symbol/name;p.parent.mkdir(parents=True,exist_ok=True)
 if not p.exists():
  with requests.get(url,stream=True,timeout=120) as r:
   r.raise_for_status()
   with p.with_suffix('.part').open('wb') as out:
    for block in r.iter_content(1024*1024):out.write(block)
  p.with_suffix('.part').replace(p)
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 checksum=requests.get(url+'.CHECKSUM',timeout=60);checksum.raise_for_status();expected=checksum.text.split()[0]
 if h.hexdigest()!=expected:raise ValueError(f'Checksum failed: {p}')
 return url,h.hexdigest(),p.stat().st_size

def main():
 a=argparse.ArgumentParser();a.add_argument('--destination',required=True);a.add_argument('--start',default='2019-12');a.add_argument('--end',default='2026-10');args=a.parse_args();begin=datetime.date.fromisoformat(args.start+'-01');end=datetime.date.fromisoformat(args.end+'-01');months=[]
 while begin<=end:months.append(begin.strftime('%Y-%m'));begin=datetime.date(begin.year+int(begin.month==12),begin.month%12+1,1)
 jobs=[(m,s,t,args.destination) for m in ['perpetual','spot'] for s in ['BTCUSDT','ETHUSDT'] for t in months]
 with ProcessPoolExecutor(max_workers=4) as pool:
  for row in pool.map(fetch,jobs):print(*row,flush=True)
if __name__=='__main__':main()
