#!/usr/bin/env python3
"""Verified official Binance archive ingestion and UTC realized measures.

Day t is data observed during UTC date t and available by 24:00 UTC t;
the forecast origin is 00:00 UTC t+1. Model code must shift this panel.
Minute timestamps are Binance bar opening times, so the first 5m return
uses close at 00:04 versus previous date 23:59. No missing price is filled.
Metrics have a conservative 10m publication lag: only stamps <=23:50 enter
day-t state. Funding uses only actual settlement records in the archive.
"""
import argparse, concurrent.futures as cf, hashlib, io, json, math, os, fcntl
from pathlib import Path
import re, time, urllib.request, urllib.error, urllib.parse, zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from scipy.special import gamma
from scipy.stats import norm

ROOT=Path(__file__).resolve().parents[1]
SOURCE_ROOT=ROOT
CONFIG=json.loads((ROOT/'config.json').read_text())
END=pd.Timestamp(CONFIG['end_date'])
ASSETS=CONFIG['assets']
RAW=ROOT/'data_raw'/'binance'; CLEAN=ROOT/'data_clean'/'minute'
OUT=ROOT/'out'/'shared'; LOG=ROOT/'logs'/'binance_run.md'
for p in [CLEAN,OUT]: p.mkdir(parents=True,exist_ok=True)
BASE='https://data.binance.vision/'
S3='https://s3-ap-northeast-1.amazonaws.com/data.binance.vision'
LOCAL_ARCHIVE=Path(os.environ.get('BINANCE_ARCHIVE',str(ROOT/'data_raw/binance')))
NS={'s':'http://s3.amazonaws.com/doc/2006-03-01/'}
MANIFEST=OUT/'download_manifest.csv'
OFFLINE=False
OFFLINE_TASKS=None

def log(s):
    s=time.strftime('%Y-%m-%d %H:%M:%S')+' '+s
    print(s,flush=True)
    with LOG.open('a') as f:f.write(s+'\n')

def fetch(url,timeout=45):
    if OFFLINE:raise RuntimeError('Network access forbidden in offline mode: '+url)
    req=urllib.request.Request(url,headers={'User-Agent':'JOD academic archive audit/1.0'})
    with urllib.request.urlopen(req,timeout=timeout) as r:return r.read()

def listing(prefix):
    marker=''; ans=[]
    for page in range(100):
        query=urllib.parse.urlencode({'prefix':prefix,'max-keys':'1000','marker':marker})
        for attempt in range(5):
            try:
                root=ET.fromstring(fetch(S3+'?'+query));break
            except Exception as e:
                if attempt==4:log('LIST_FAIL '+prefix+' '+repr(e));return ans
                time.sleep(min(2**attempt,12))
        for c in root.findall('s:Contents',NS):
            key=c.find('s:Key',NS).text
            if key.endswith('.zip'):ans.append((key,int(c.find('s:Size',NS).text)))
        if root.find('s:IsTruncated',NS).text!='true':break
        marker=root.find('s:NextMarker',NS)
        marker=marker.text if marker is not None else root.findall('s:Contents',NS)[-1].find('s:Key',NS).text
    return ans

def eligible(key,period):
    name=key.rsplit('/',1)[-1]
    x=re.search(r'(\d{4}-\d{2}(?:-\d{2})?)\.zip$',name)
    if not x:return False
    d=x.group(1)
    if period=='monthly':return d<=CONFIG['monthly_end']
    if '/metrics/' in key:return d<=str(END.date())
    return '2026-10-01'<=d<=str(END.date())

def discover(asset,market,kind,period):
    global OFFLINE_TASKS
    if OFFLINE:
        if OFFLINE_TASKS is None:
            source=SOURCE_ROOT/'out'/'shared'/'download_manifest.csv'
            if not source.exists():raise FileNotFoundError('Offline archive discovery requires '+str(source))
            OFFLINE_TASKS=pd.read_csv(source).to_dict('records')
        cols=['asset','market','kind','period','key','bytes_expected']
        return [{k:r[k] for k in cols} for r in OFFLINE_TASKS if r['asset']==asset and r['market']==market and r['kind']==kind and r['period']==period and eligible(r['key'],period)]
    symbol=asset+'USDT'
    prefix=f'data/{market}/{period}/{kind}/{symbol}/'
    if kind in ['klines','premiumIndexKlines','markPriceKlines']:prefix+='1m/'
    out=[]
    for key,size in listing(prefix):
        if eligible(key,period):out.append({'asset':asset,'market':market,'kind':kind,'period':period,'key':key,'bytes_expected':size})
    return out

def hashfile(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def download(task):
    p=RAW/task['key']
    check=Path(str(p)+'.CHECKSUM');receipt=Path(str(p)+'.verified.json')
    row=dict(task);row.update(status='failed',sha256='',checksum_match=False,crc_ok=False,error='')
    if OFFLINE:
        try:
            if not p.exists() or not check.exists():raise FileNotFoundError('Missing offline ZIP or official CHECKSUM')
            if p.stat().st_size!=int(task['bytes_expected']):raise ValueError('offline size mismatch')
            expected=check.read_text().strip().split()[0];actual=hashfile(p)
            if actual!=expected:raise ValueError('offline official SHA256 mismatch')
            with zipfile.ZipFile(p) as z:
                bad=z.testzip()
                if bad:raise ValueError('offline CRC mismatch '+bad)
            row.update(status='offline_verified',sha256=actual,checksum_match=True,crc_ok=True)
        except Exception as e:row['error']=repr(e)
        return row
    p.parent.mkdir(parents=True,exist_ok=True)
    if receipt.exists() and p.exists() and p.stat().st_size==task['bytes_expected']:
        try:
            saved=json.loads(receipt.read_text())
            if hashfile(p)!=saved['sha256']:raise ValueError('cached SHA256 mismatch')
            with zipfile.ZipFile(p) as z:
                if z.testzip():raise ValueError('cached CRC mismatch')
            row.update(saved);row['status']='cached';return row
        except Exception:pass
    for attempt in range(5):
        try:
            local=LOCAL_ARCHIVE/task['key']
            localcheck=Path(str(local)+'.CHECKSUM')
            if not p.exists() and local.exists() and local.stat().st_size==task['bytes_expected']:
                p.symlink_to(local)
            if not check.exists() and localcheck.exists():check.symlink_to(localcheck)
            if not check.exists():check.write_bytes(fetch(BASE+task['key']+'.CHECKSUM'))
            expected=check.read_text().strip().split()[0]
            if len(expected)!=64:raise ValueError('invalid official SHA256 checksum')
            part=Path(str(p)+'.part')
            if not p.exists() or p.stat().st_size!=task['bytes_expected']:
                n=part.stat().st_size if part.exists() else 0
                headers={'User-Agent':'JOD academic archive audit/1.0'}
                if n:headers['Range']=f'bytes={n}-'
                with urllib.request.urlopen(urllib.request.Request(BASE+task['key'],headers=headers),timeout=90) as r:
                    mode='ab' if n and r.status==206 else 'wb'
                    with part.open(mode) as f:
                        for b in iter(lambda:r.read(1<<20),b''):f.write(b)
                if part.stat().st_size!=task['bytes_expected']:raise ValueError('size mismatch')
                part.replace(p)
            actual=hashfile(p)
            if actual!=expected:
                p.unlink();raise ValueError('official SHA256 mismatch; invalid downloaded copy discarded')
            with zipfile.ZipFile(p) as z:
                bad=z.testzip()
                if bad:raise ValueError('CRC mismatch '+bad)
            row.update(status='verified',sha256=actual,checksum_match=True,crc_ok=True,error='')
            receipt.write_text(json.dumps({k:row[k] for k in ['status','sha256','checksum_match','crc_ok','error']}))
            return row
        except Exception as e:
            row['error']=repr(e)
            if isinstance(e,urllib.error.HTTPError) and e.code==404:break
            if attempt<4:time.sleep(min(2**attempt+np.random.default_rng(20261007+attempt).uniform(0,.5),15))
    return row

def save_manifest(new):
    with (OUT/'download_manifest.lock').open('w') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        old=pd.read_csv(MANIFEST).to_dict('records') if MANIFEST.exists() else []
        bykey={r['key']:r for r in old};bykey.update({r['key']:r for r in new})
        tmp=Path(str(MANIFEST)+'.tmp');pd.DataFrame(bykey.values()).sort_values('key').to_csv(tmp,index=False);tmp.replace(MANIFEST)
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def run_download(tasks,workers):
    rows=[];old=pd.read_csv(MANIFEST).to_dict('records') if MANIFEST.exists() else []
    bykey={r['key']:r for r in old}
    with cf.ThreadPoolExecutor(max_workers=workers) as pool:
        for i,row in enumerate(pool.map(download,tasks)):
            rows.append(row);bykey[row['key']]=row
            if (i+1)%50==0:
                if not OFFLINE or (i+1)%250==0:save_manifest(rows)
                log(f'DOWNLOAD {i+1}/{len(tasks)} bytes={sum(r["bytes_expected"] for r in rows)} verified={sum(r["status"]!="failed" for r in rows)}')
    save_manifest(rows)
    return rows

COLS=['timestamp','open','high','low','close','volume','close_time','quote_volume','trades','taker_buy_base','taker_buy_quote','ignore']
def read_kline(p):
    with zipfile.ZipFile(p) as z:
        name=next(x for x in z.namelist() if x.endswith('.csv'))
        df=pd.read_csv(z.open(name),header=None,names=COLS,dtype=str)
    if len(df) and not str(df.iloc[0,0]).isdigit():df=df.iloc[1:]
    for c in COLS[:-1]:df[c]=pd.to_numeric(df[c],errors='coerce')
    unit='us' if df['timestamp'].median()>1e14 else 'ms'
    df['timestamp']=pd.to_datetime(df['timestamp'],unit=unit,utc=True).dt.tz_localize(None)
    return df[['timestamp','open','high','low','close','volume','quote_volume','trades','taker_buy_quote']].sort_values('timestamp')

def clean_month(df):
    count={'raw_rows':len(df),'duplicate_rows':int(df.timestamp.duplicated().sum()),'bad_price_rows':0,'zero_run_rows':0,'mad_reversal_rows':0}
    source_raw=df.groupby('source_file').size().to_dict() if 'source_file' in df else {}
    source_dup=df.loc[df.timestamp.duplicated(keep='last')].groupby('source_file').size().to_dict() if 'source_file' in df else {}
    df=df.drop_duplicates('timestamp',keep='last').reset_index(drop=True)
    bad=(~np.isfinite(df.close))|(df.close<=0)|(df.high<df.low)|(df.open<=0)
    count['bad_price_rows']=int(bad.sum())
    zero=df.volume.eq(0).to_numpy();contig=df.timestamp.diff().eq(pd.Timedelta(minutes=1)).to_numpy()
    groups=np.cumsum(np.r_[True,(zero[1:]!=zero[:-1])|(~contig[1:])])
    lengths=pd.Series(groups).map(pd.Series(groups[zero]).value_counts()).fillna(0).to_numpy()
    removezero=zero&(lengths>=2);count['zero_run_rows']=int(removezero.sum())
    lr=np.log(df.close).diff();dates=df.timestamp.dt.normalize()
    med=lr.groupby(dates).transform('median');mad=(lr-med).abs().groupby(dates).transform('median')
    nxt=lr.shift(-1)
    adjacent=pd.Series(contig,index=df.index)&pd.Series(contig,index=df.index).shift(-1).fillna(False).astype(bool)
    rev=(lr.abs()>50*mad.clip(lower=1e-12))&(lr*nxt<0)&((lr+nxt).abs()<.2*lr.abs())&adjacent
    count['mad_reversal_rows']=int(rev.sum())
    # Remove only detected erroneous price; returns spanning removed rows remain missing.
    kept=~(bad|removezero|rev)
    if source_raw:
        fc=[]
        for file,ix in df.groupby('source_file').indices.items():
            fc.append({'file':file,'raw_rows':int(source_raw[file]),'duplicate_rows':int(source_dup.get(file,0)),'bad_price_rows':int(bad.to_numpy()[ix].sum()),'zero_run_rows':int(removezero[ix].sum()),'mad_reversal_rows':int(rev.to_numpy()[ix].sum()),'clean_rows':int(kept.to_numpy()[ix].sum())})
        count['_file_counts']=fc
    df=df.loc[kept].copy()
    count['clean_rows']=len(df)
    return df,count

def parzen(x):
    x=np.abs(x);return np.where(x<=.5,1-6*x*x+6*x*x*x,np.where(x<=1,2*(1-x)**3,0))

def measure_day(g,prev_close,prev_stamp,raw_minutes,asset,market,prev_tail=None):
    date=g.timestamp.iloc[0].normalize();minute=((g.timestamp-date).dt.total_seconds()/60).astype(int).to_numpy()
    price=np.full(1440,np.nan);price[minute]=g.close.to_numpy()
    p=np.r_[prev_close,price];r=np.diff(np.log(p))
    if prev_stamp!=date-pd.Timedelta(minutes=1):r[0]=np.nan
    def returns(step,offset=None):
        if offset is None:offset=step-1
        ix=np.arange(offset,1440,step);prior=ix-step
        end=price[ix];start=np.where(prior>=0,price[np.maximum(prior,0)],prev_close)
        ok=(prior>=0)|(prev_stamp==date-pd.Timedelta(minutes=1))
        if offset!=step-1:
            for j in np.where(prior<0)[0]:
                lookup=date+pd.Timedelta(minutes=int(prior[j]))
                start[j]=(prev_tail or {}).get(lookup,np.nan)
                # Every previous-date minute between the start and midnight
                # must exist, just as every current-date minute must exist.
                # Endpoint-only acceptance would cross a cleaned/raw gap.
                ok[j]=all(np.isfinite((prev_tail or {}).get(date+pd.Timedelta(minutes=k),np.nan))
                          for k in range(int(prior[j]),0))
        rr=np.log(end/start);rr[~ok]=np.nan
        # An endpoint return across any absent minute is not accepted.
        miss=np.cumsum(np.r_[0,~np.isfinite(price)])
        span=miss[ix+1]-miss[np.maximum(prior+1,0)]
        rr[span>0]=np.nan
        return rr
    x=returns(5);finite=x[np.isfinite(x)];n=len(finite)
    rv=float(np.nansum(x*x));bpv=float(np.nansum(np.abs(x[1:])*np.abs(x[:-1])))*(np.pi/2)*(288/287)
    rq=288/3*float(np.nansum(x**4));mu=2**(2/3)*gamma(7/6)/np.sqrt(np.pi)
    tq=288**2/(286*mu**3)*float(np.nansum(np.abs(x[:-2]*x[1:-1]*x[2:])**(4/3)))
    c=np.pi**2/4+np.pi-5
    z=(rv-bpv)/rv/np.sqrt(c/288*max(1,tq/max(bpv**2,1e-30))) if rv>0 else np.nan
    sig=bool(z>norm.ppf(.99));semipos=float(np.nansum(np.where(x>0,x*x,0)));semineg=float(np.nansum(np.where(x<0,x*x,0)))
    finite_r=np.where(np.isfinite(r),r,0);rob=float(finite_r@finite_r)
    for h in range(1,11):rob+=2*float(parzen(h/11))*(finite_r[h:]@finite_r[:-h])
    slot=np.arange(4,1440,5);clock=np.arange(1440)
    funding=np.minimum.reduce([abs(clock-v) for v in [0,480,960,1440]])<=30
    expiry=abs(clock-480)<=60
    res={'asset':asset,'market':market,'date':date,'rv5':rv,'rv_sub':float(np.mean([np.nansum(returns(5,o)**2) for o in range(5)])),
         'rv1':float(np.nansum(r*r)),'rv15':float(np.nansum(returns(15)**2)),'rv30':float(np.nansum(returns(30)**2)),
         'rv_robust':max(rob,0),'bpv':bpv,'bns_z':z,'bns_critical':float(norm.ppf(.99)),
         'jump':max(rv-bpv,0) if sig else 0,'jump_significant':sig,'semipos':semipos,'semineg':semineg,
         'signed_jump':semipos-semineg if sig else 0,'rq':rq,'tq':tq,
         'volume':float(g.volume.sum()),'quote_volume':float(g.quote_volume.sum()),'taker_buy_quote':float(g.taker_buy_quote.sum()),
         'trades':float(g.trades.sum()),'close':float(price[-1]) if np.isfinite(price[-1]) else np.nan,
         'daily_return':float(np.log(price[-1]/prev_close)) if np.isfinite(price[-1]) and np.isfinite(prev_close) and prev_stamp==date-pd.Timedelta(minutes=1) else np.nan,
         'asia':float(np.nansum(x[slot<480]**2)),'europe':float(np.nansum(x[(slot>=480)&(slot<960)]**2)),
         'america':float(np.nansum(x[slot>=960]**2)),'funding_window':float(np.nansum(x[funding[slot]]**2)),
         'expiry_window':float(np.nansum(x[expiry[slot]]**2)),'weekend':bool(date.dayofweek>=5),
         'raw_minutes':raw_minutes,'clean_minutes':len(g),'raw_missing_rate':1-raw_minutes/1440,
         'missing_rate':1-len(g)/1440,'valid_day':len(g)>=1433 and n>=285}
    for freq in [1,2,3,5,10,15,30,60]:res['signature_'+str(freq)]=float(np.nansum(returns(freq)**2))
    for h in range(24):res['hour_'+str(h)]=float(np.nansum(x[(slot>=h*60)&(slot<(h+1)*60)]**2))
    return res

def build_market(asset,market):
    tasks=[]
    for period in ['monthly','daily']:tasks+=discover(asset,market,'klines',period)
    log(f'DISCOVER {asset} {market} klines {len(tasks)}')
    rows=run_download(tasks,ARGS.workers)
    if OFFLINE and (not rows or any(r['status']=='failed' for r in rows)):
        raise RuntimeError('Offline market inputs incomplete: '+asset+' '+market)
    days=[];counts=[];prev_close=np.nan;prev_stamp=pd.NaT;prev_tail={};chunks=[]
    for row in sorted(rows,key=lambda x:x['key'].rsplit('/',1)[-1]):
        if row['status']=='failed':continue
        key=row['key'];raw=read_kline(RAW/key)
        raw=raw[raw.timestamp<END+pd.Timedelta(days=1)]
        if raw.empty:continue
        raw['source_file']=key;chunks.append(raw)
    if not chunks:
        log(f'NO_DATA {asset} {market}');return
    raw=pd.concat(chunks,ignore_index=True).sort_values('timestamp').reset_index(drop=True);del chunks
    raw_n=raw.drop_duplicates('timestamp').groupby(raw.drop_duplicates('timestamp').timestamp.dt.normalize()).size()
    clean,count=clean_month(raw)
    counts=count.pop('_file_counts',[])
    for countrow in counts:countrow.update(asset=asset,market=market)
    del raw
    # Clean on the entire series so zero-volume intervals and errors cross month boundaries.
    for key,gmonth in clean.groupby('source_file',sort=False):
        fname=key.rsplit('/',1)[-1].replace('.zip','.parquet');cleanpath=CLEAN/market/asset/fname;cleanpath.parent.mkdir(parents=True,exist_ok=True)
        gmonth.drop(columns='source_file').to_parquet(cleanpath,index=False,compression='zstd')
    for date,g in clean.groupby(clean.timestamp.dt.normalize()):
        days.append(measure_day(g,prev_close,prev_stamp,int(raw_n.get(date,0)),asset,market,prev_tail))
        prev_close=float(g.close.iloc[-1]);prev_stamp=g.timestamp.iloc[-1]
        prev_tail=dict(zip(g.timestamp.iloc[-5:],g.close.iloc[-5:]))
    if not days:
        log(f'NO_DATA {asset} {market}');return
    panel=pd.DataFrame(days).sort_values('date').drop_duplicates('date').set_index('date')
    panel=panel.reindex(pd.date_range(panel.index.min(),END,freq='D'))
    panel.index.name='date';panel=panel.reset_index();panel['asset']=asset;panel['market']=market
    for c in ['raw_minutes','clean_minutes']:panel[c]=panel[c].fillna(0)
    for c in ['raw_missing_rate','missing_rate']:panel[c]=panel[c].fillna(1)
    panel['valid_day']=panel.valid_day.fillna(False).astype(bool)
    panel['daily_return']=np.log(panel.close/panel.close.shift(1))
    target=ROOT/'data_clean'/'daily_parts';target.mkdir(exist_ok=True)
    dest=target/f'{asset}_{market.replace("/","_")}.parquet';temp=Path(str(dest)+'.tmp')
    panel.to_parquet(temp,index=False);temp.replace(dest)
    pd.DataFrame(counts).to_csv(OUT/f'cleaning_{asset}_{market.replace("/","_")}.csv',index=False)
    quality={'asset':asset,'market':market,'start':str(panel.date.min().date()),'end':str(panel.date.max().date()),
             'days':len(panel),'valid_days':int(panel.valid_day.sum()),'raw_missing_rate':1-panel.raw_minutes.sum()/(1440*len(panel)),
             'missing_rate':1-panel.clean_minutes.sum()/(1440*len(panel)),'failed_archives':sum(x['status']=='failed' for x in rows),
             'downloaded_archives':sum(x['status']!='failed' for x in rows),
             'predictor_day_coverage':float(panel.valid_day.mean()),
             'gate1_pass':bool(panel.valid_day.sum()>=1000 and (1-panel.clean_minutes.sum()/(1440*len(panel)))<.005 and panel.valid_day.mean()>=.95)}
    pd.DataFrame([quality]).to_csv(OUT/f'quality_{asset}_{market.replace("/","_")}.csv',index=False)
    consolidate()
    log('MARKET_READY '+json.dumps(quality))

def consolidate():
    with (OUT/'consolidate.lock').open('w') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX)
        _consolidate_unlocked()
        fcntl.flock(lock.fileno(),fcntl.LOCK_UN)

def _consolidate_unlocked():
    parts=sorted(p for p in (ROOT/'data_clean'/'daily_parts').glob('*.parquet') if not p.name.startswith('.'))
    if parts:
        panel=pd.concat([pd.read_parquet(p) for p in parts],ignore_index=True)
        dest=ROOT/'data_clean'/'market_daily.parquet';temp=Path(str(dest)+'.tmp')
        panel.to_parquet(temp,index=False);temp.replace(dest)
        align=[]
        for (asset,market),g in panel.groupby(['asset','market']):
            g=g.sort_values('date');expected=np.log(g.close/g.close.shift(1))
            error=(g.daily_return.isna()!=expected.isna())|((g.daily_return-expected).abs()>1e-14)
            align.append({'asset':asset,'market':market,'calendar_days':len(g),'day_end_close_available':int(g.close.notna().sum()),'consecutive_endpoints_available':int(expected.notna().sum()),'daily_return_available':int(g.daily_return.notna().sum()),'alignment_error_count':int(error.sum())})
        pd.DataFrame(align).to_csv(OUT/'daily_return_alignment_audit.csv',index=False)
        sig=[]
        for (a,m),g in panel.groupby(['asset','market']):
            for freq in [1,2,3,5,10,15,30,60]:sig.append({'asset':a,'market':m,'frequency_minutes':freq,'mean_rv':g.loc[g.valid_day,'signature_'+str(freq)].mean(),'days':int(g.valid_day.sum())})
        pd.DataFrame(sig).to_csv(OUT/'signature_curve.csv',index=False)
        common=[];commonsig=[]
        for asset,ap in panel.groupby('asset'):
            if set(ap.market.unique())!={'spot','futures/um'}:continue
            start=max(ap[ap.market==m].date.min() for m in ['spot','futures/um'])
            end=min(ap[ap.market==m].date.max() for m in ['spot','futures/um'])
            gp=ap[ap.date.between(start,end)]
            pair_valid=gp.pivot(index='date',columns='market',values='valid_day').fillna(False).all(axis=1)
            for market,g in gp.groupby('market'):
                missing=1-g.clean_minutes.sum()/(1440*len(g));rawmissing=1-g.raw_minutes.sum()/(1440*len(g))
                common.append({'asset':asset,'market':market,'start':str(start.date()),'end':str(end.date()),'calendar_days':len(g),'market_valid_days':int(g.valid_day.sum()),'pair_valid_days':int(pair_valid.sum()),'raw_missing_rate':rawmissing,'missing_rate':missing,'predictor_day_coverage':float(g.valid_day.mean()),'pair_predictor_coverage':float(pair_valid.mean()),'gate1_market_pair_pass':bool(pair_valid.sum()>=1000 and missing<.005 and pair_valid.mean()>=.95)})
                matched=g[g.date.isin(pair_valid[pair_valid].index)]
                for freq in [1,2,3,5,10,15,30,60]:commonsig.append({'asset':asset,'market':market,'frequency_minutes':freq,'mean_rv':matched['signature_'+str(freq)].mean(),'days':len(matched),'start':str(start.date()),'end':str(end.date())})
        if common:pd.DataFrame(common).to_csv(OUT/'common_sample_quality.csv',index=False)
        if commonsig:
            cs=pd.DataFrame(commonsig);cs.to_csv(OUT/'signature_curve_common.csv',index=False)
            pivot=cs.pivot(index=['asset','market'],columns='frequency_minutes',values='mean_rv')
            summary=pd.DataFrame({'rv1_over_rv5':pivot[1]/pivot[5],'rv15_over_rv5':pivot[15]/pivot[5],'rv30_over_rv5':pivot[30]/pivot[5],'rv60_over_rv5':pivot[60]/pivot[5]}).reset_index()
            summary.to_csv(OUT/'signature_summary.csv',index=False)
    for pattern,name in [('quality_*.csv','data_quality.csv'),('cleaning_*.csv','cleaning_counts.csv')]:
        prefix='quality' if name=='data_quality.csv' else 'cleaning'
        # Process audit tables also begin "cleaning_"; select only the exact
        # per-asset/per-market source receipts generated by build_market.
        selected={f'{prefix}_{asset}_{market.replace("/","_")}.csv'
                  for asset in ASSETS for market in ['spot','futures/um']}
        files=[p for p in OUT.glob(pattern) if p.name in selected]
        if files:
            combined=pd.concat([pd.read_csv(p) for p in files],ignore_index=True)
            if name=='cleaning_counts.csv':combined=combined.sort_values(['asset','market','file']).reset_index(drop=True)
            combined.to_csv(OUT/name,index=False)
    if parts:
        quality=pd.read_csv(OUT/'data_quality.csv')
        for i,r in quality.iterrows():
            g=panel[(panel.asset==r.asset)&(panel.market==r.market)]
            quality.loc[i,'predictor_day_coverage']=float(g.valid_day.mean())
            quality.loc[i,'gate1_pass']=bool(g.valid_day.sum()>=1000 and r.missing_rate<.005 and g.valid_day.mean()>=.95)
        fields=['asset','market','start','end','days','valid_days','raw_missing_rate','missing_rate','failed_archives','downloaded_archives','predictor_day_coverage','gate1_pass']
        quality=quality[fields].sort_values(['asset','market']).reset_index(drop=True)
        quality.to_csv(OUT/'data_quality.csv',index=False)
    rest=RAW/'rest'
    if rest.exists():
        receipts=[json.loads(p.read_text()) for p in rest.glob('*.receipt.json') if not p.name.startswith('.')]
        if receipts:pd.DataFrame(receipts).to_csv(OUT/'funding_rest_receipts.csv',index=False)

def read_other(p):
    with zipfile.ZipFile(p) as z:
        n=next(x for x in z.namelist() if x.endswith('.csv'));return pd.read_csv(z.open(n))

def funding_rest_tail(asset):
    start=pd.Timestamp(CONFIG['monthly_end'])+pd.offsets.MonthBegin(1)
    url='https://fapi.binance.com/fapi/v1/fundingRate?'+urllib.parse.urlencode({'symbol':asset+'USDT','startTime':int(start.tz_localize('UTC').timestamp()*1000),'endTime':int((END+pd.Timedelta(days=1)).tz_localize('UTC').timestamp()*1000)-1,'limit':1000})
    dest=RAW/'rest'/f'{asset}-fundingRate-{start.date()}_{END.date()}.json'
    if not OFFLINE:dest.parent.mkdir(exist_ok=True)
    receipt=Path(str(dest)+'.receipt.json')
    for attempt in range(4):
        try:
            if OFFLINE and (not dest.exists() or not receipt.exists()):raise FileNotFoundError('Offline funding JSON or original receipt missing')
            if dest.exists():raw=dest.read_bytes()
            else:
                raw=fetch(url);parsed=json.loads(raw)
                if not isinstance(parsed,list) or not parsed:raise ValueError('funding REST empty or invalid schema')
                dest.write_bytes(raw)
            rows=json.loads(raw)
            if len(rows)>=1000:raise ValueError('funding REST limit reached; pagination required')
            f=pd.DataFrame(rows)
            if not {'fundingTime','fundingRate'}.issubset(f):raise ValueError('funding REST schema mismatch')
            r={'asset':asset,'source':'official Binance public REST','url':url,'file':str(dest.relative_to(SOURCE_ROOT)),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'records':len(f),'official_checksum_available':False,'status':'verified_schema_and_local_sha256'}
            if receipt.exists() and json.loads(receipt.read_text())['sha256']!=r['sha256']:raise ValueError('cached funding REST SHA mismatch')
            if not OFFLINE:receipt.write_text(json.dumps(r,indent=2))
            log(f'FUNDING_REST_READY {asset} {len(f)} actual settled records')
            f=f.rename(columns={'fundingTime':'calc_time','fundingRate':'last_funding_rate'})
            return f
        except Exception as e:
            log('FUNDING_REST_FAIL '+asset+' '+repr(e))
            if OFFLINE:return None
            if attempt<3:time.sleep(min(2**attempt,8))
    return None

def build_leverage(asset):
    tasks=[]
    for kind in ['premiumIndexKlines','fundingRate']:
        for period in ['monthly','daily']:tasks+=discover(asset,'futures/um',kind,period)
    tasks+=discover(asset,'futures/um','metrics','daily')
    log(f'DISCOVER {asset} leverage {len(tasks)}')
    rows=run_download(tasks,ARGS.workers);premium=[];funding=[];metrics=[]
    if OFFLINE and (not rows or any(r['status']=='failed' for r in rows)):
        raise RuntimeError('Offline leverage inputs incomplete: '+asset)
    parse_failures=[]
    for row in sorted(rows,key=lambda x:x['key']):
        if row['status']=='failed':continue
        try:
            if row['kind']=='premiumIndexKlines':
                df=read_kline(RAW/row['key'])
                for date,g in df.groupby(df.timestamp.dt.normalize()):
                    g=g.drop_duplicates('timestamp').sort_values('timestamp');x=g.close.to_numpy();r=np.diff(x)
                    gap=np.diff(g.timestamp.to_numpy()).astype('timedelta64[m]').astype(int)
                    r=r[gap==1];usable=len(g)>=1433
                    premium.append({'date':date,'basis_abs':float(np.mean(abs(x))) if usable else np.nan,'basis_mean':float(np.mean(x)) if usable else np.nan,'basis_rv':float(r@r) if usable else np.nan,'basis_minutes':len(g)})
            elif row['kind']=='fundingRate':
                df=read_other(RAW/row['key']);funding.append(df)
            else:metrics.append(read_other(RAW/row['key']))
        except Exception as e:
            parse_failures.append(row['key']);log('PARSE_FAIL '+row['key']+' '+repr(e))
    if OFFLINE and parse_failures:raise RuntimeError('Offline archive parse failed: '+str(parse_failures))
    tail=funding_rest_tail(asset)
    if OFFLINE and tail is None:raise RuntimeError('Offline true settled funding REST input unavailable: '+asset)
    if tail is not None:funding.append(tail)
    basefile=ROOT/'data_clean'/'daily_parts'/f'{asset}_futures_um.parquet'
    for wait_attempt in range(180):
        if basefile.exists():break
        if wait_attempt%30==0:log('WAIT_MARKET_PART '+asset)
        time.sleep(2)
    base=pd.read_parquet(basefile)[['date','close','quote_volume','taker_buy_quote','daily_return']]
    if premium:
        p=pd.DataFrame(premium).groupby('date',as_index=False).last();base=base.merge(p,on='date',how='left')
    if funding:
        f=pd.concat(funding,ignore_index=True);tc=next((x for x in f if x in ['calc_time','funding_time','timestamp','fundingTime']),None)
        vc=next((x for x in f if x in ['last_funding_rate','funding_rate','fundingRate']),None)
        if tc and vc:
            f[tc]=pd.to_datetime(pd.to_numeric(f[tc]),unit='ms',utc=True).dt.tz_localize(None)
            f=f[f[tc]<END+pd.Timedelta(days=1)].drop_duplicates(tc).sort_values(tc);f['date']=f[tc].dt.normalize()
            f[vc]=pd.to_numeric(f[vc],errors='coerce')
            if 'funding_interval_hours' in f:
                # REST has no interval-hours field. Audit actual adjacent settlement spacing.
                observed=f[tc].diff().dt.total_seconds().div(3600).round()
                observed=observed.where(observed.between(1,8))
                f['funding_interval_hours']=f.funding_interval_hours.fillna(observed)
            d=f.groupby('date')[vc].agg(funding_rate='sum',funding_last='last',funding_abs=lambda x:float(abs(x).mean()),funding_settlements='count').reset_index()
            if 'funding_interval_hours' in f:
                d['funding_covered_hours']=f.groupby('date').funding_interval_hours.sum().to_numpy()
                incomplete=d.funding_covered_hours<24
                d.loc[incomplete,['funding_rate','funding_last','funding_abs']]=np.nan
            d['funding_sign']=np.sign(d.funding_last);base=base.merge(d,on='date',how='left')
            base['funding_signed']=base['funding_rate']
            f.to_parquet(ROOT/'data_clean'/f'funding_settled_{asset}.parquet',index=False)
        else:log('FUNDING_SCHEMA '+asset+' '+','.join(f.columns))
    if metrics:
        m=pd.concat(metrics,ignore_index=True);tc=next((x for x in m if x in ['create_time','timestamp','time']),None)
        if tc:
            m[tc]=pd.to_datetime(m[tc],utc=True).dt.tz_localize(None);m=m.drop_duplicates(tc).sort_values(tc)
            minute=m[tc].dt.hour*60+m[tc].dt.minute;m=m[minute<=1430];m['date']=m[tc].dt.normalize()
            numerical=[x for x in m if x not in [tc,'date','symbol']]
            for c in numerical:m[c]=pd.to_numeric(m[c],errors='coerce')
            for c in ['sum_open_interest','sum_open_interest_value','count_long_short_ratio']:
                if c in m:m.loc[m[c]<=0,c]=np.nan
            d=m.groupby('date')[numerical].last();d['metrics_observations']=m.groupby('date').size()
            d=d.reindex(pd.date_range(d.index.min(),d.index.max(),freq='D'));d.index.name='date';d=d.reset_index()
            if 'sum_open_interest' in d:d['oi_log_change']=np.log(d.sum_open_interest).diff()
            if 'sum_open_interest_value' in d:d['oi_value']=d.sum_open_interest_value
            if 'count_long_short_ratio' in d:d['account_long_short_ratio']=d.count_long_short_ratio
            base=base.merge(d,on='date',how='left');m.to_parquet(ROOT/'data_clean'/f'metrics_5m_{asset}.parquet',index=False)
    base['asset']=asset
    base['taker_imbalance_abs']=abs(2*base.taker_buy_quote/base.quote_volume-1)
    if 'oi_value' in base:base['oi_volume_ratio']=base.oi_value/base.quote_volume
    base['abs_return']=base.daily_return.abs()
    # Rolling scaling and first principal component belong in each model training window.
    dest=ROOT/'data_clean'/'leverage_parts';dest.mkdir(exist_ok=True)
    base.to_parquet(dest/f'{asset}.parquet',index=False)
    allp=pd.concat([pd.read_parquet(p) for p in sorted(dest.glob('*.parquet')) if not p.name.startswith('.')],ignore_index=True)
    allp.to_parquet(ROOT/'data_clean'/'leverage_daily.parquet',index=False)
    cov=[]
    for a,g in allp.groupby('asset'):
        for c in g:
            if c not in ['date','asset']:cov.append({'asset':a,'variable':c,'days':len(g),'available_days':int(g[c].notna().sum()),'coverage':g[c].notna().mean(),'start':str(g.loc[g[c].notna(),'date'].min()),'end':str(g.loc[g[c].notna(),'date'].max())})
    pd.DataFrame(cov).to_csv(OUT/'leverage_coverage.csv',index=False)
    log(f'LEVERAGE_READY {asset} days={len(base)} metrics={len(metrics)} funding={len(funding)} premium={len(premium)}')

def measurement_qa():
    date=pd.Timestamp('2024-01-01');rng=np.random.default_rng(CONFIG['seed']);out=[]
    for j in range(200):
        r=rng.normal(0,np.sqrt(.0004/1440),1440);price=100*np.exp(np.cumsum(r))
        g=pd.DataFrame({'timestamp':pd.date_range(date,periods=1440,freq='min'),'close':price,'volume':1,'quote_volume':100,'taker_buy_quote':50,'trades':1})
        out.append(measure_day(g,100,date-pd.Timedelta(minutes=1),1440,'QA','QA'))
    x=pd.DataFrame(out)
    result={'rv5_mean_over_true':x.rv5.mean()/.0004,'rq_mean_over_true_squared':x.rq.mean()/.0004**2,'tq_mean_over_true_squared':x.tq.mean()/.0004**2,'bns_reject_fraction':x.jump_significant.mean(),'session_add_error':abs(x.rv5-x.asia-x.europe-x.america).max(),'semivariance_add_error':abs(x.rv5-x.semipos-x.semineg).max(),'kernel_mean_over_true':x.rv_robust.mean()/.0004}
    pd.DataFrame([result]).to_csv(OUT/'measurement_qa.csv',index=False)
    assert .9<result['rv5_mean_over_true']<1.1
    assert .85<result['rq_mean_over_true_squared']<1.15
    assert .85<result['tq_mean_over_true_squared']<1.15
    assert result['session_add_error']<1e-15 and result['semivariance_add_error']<1e-15
    log('MEASUREMENT_QA_PASS synthetic Brownian paths seed='+str(CONFIG['seed']))

def plot_signature():
    if not (OUT/'signature_curve_common.csv').exists():return
    sig=pd.read_csv(OUT/'signature_curve_common.csv')
    if sig.asset.nunique()<len(ASSETS):return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,5,figsize=(15,6),constrained_layout=True)
    for ax,asset in zip(axes.ravel(),ASSETS):
        for market,label,color in [('spot','Spot','#1f77b4'),('futures/um','Perpetual','#d95f02')]:
            g=sig[(sig.asset==asset)&(sig.market==market)].sort_values('frequency_minutes')
            ax.plot(g.frequency_minutes,g.mean_rv*1e4,'o-',label=label,color=color,ms=3)
        ax.set_xscale('log');ax.set_xticks([1,5,15,30,60],labels=['1','5','15','30','60']);ax.set_title(asset);ax.grid(alpha=.15)
        ax.set_xlabel('Sampling interval (minutes)');ax.set_ylabel('Mean daily RV (×10⁻⁴)')
    axes[0,0].legend(frameon=False)
    fig.savefig(OUT/'signature_curve.png',dpi=300);plt.close(fig)

def terminal_verification():
    required=[ROOT/'data_clean'/'market_daily.parquet',ROOT/'data_clean'/'leverage_daily.parquet',MANIFEST,OUT/'cleaning_counts.csv',OUT/'daily_return_alignment_audit.csv',OUT/'funding_rest_receipts.csv']
    if not all(p.exists() for p in required):
        log('INGESTION_AUDIT_PENDING missing final panel or receipts');return
    m=pd.read_parquet(required[0]);l=pd.read_parquet(required[1]);d=pd.read_csv(required[2]);c=pd.read_csv(required[3]);a=pd.read_csv(required[4]);r=pd.read_csv(required[5])
    checks={'assets':m.asset.nunique(),'markets':m.groupby(['asset','market']).ngroups,'market_daily_rows':len(m),'leverage_daily_rows':len(l),'duplicate_market_dates':int(m.duplicated(['asset','market','date']).sum()),'download_archives':len(d),'failed_archives':int(d.status.eq('failed').sum()),'official_checksum_verified':int(d.checksum_match.sum()),'zip_crc_verified':int(d.crc_ok.sum()),'raw_minute_rows':int(c.raw_rows.sum()),'clean_minute_rows':int(c.clean_rows.sum()),'session_identity_max_error':float(abs(m.rv5-m.asia-m.europe-m.america).max()),'semivariance_identity_max_error':float(abs(m.rv5-m.semipos-m.semineg).max()),'daily_return_alignment_errors':int(a.alignment_error_count.sum()),'rest_assets':len(r),'rest_actual_settled_records':int(r.records.sum())}
    checks['all_pass']=bool(checks['assets']==10 and checks['markets']==20 and checks['duplicate_market_dates']==0 and checks['failed_archives']==0 and checks['official_checksum_verified']==checks['download_archives'] and checks['zip_crc_verified']==checks['download_archives'] and checks['session_identity_max_error']<1e-12 and checks['semivariance_identity_max_error']<1e-12 and checks['daily_return_alignment_errors']==0 and checks['rest_assets']==10)
    pd.DataFrame([checks]).to_csv(OUT/'ingestion_checks.csv',index=False)
    (ROOT/'logs'/'binance_terminal_verification.json').write_text(json.dumps(checks,indent=2))
    log('INGESTION_AUDIT '+json.dumps(checks))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=8);parser.add_argument('--assets',nargs='+',default=ASSETS)
    parser.add_argument('--stage',choices=['all','markets','leverage','consolidate'],default='all')
    parser.add_argument('--offline',action='store_true',help='Read-only verification of recorded ZIP/CHECKSUM/REST inputs; forbid all network access')
    parser.add_argument('--output-root',type=Path,help='Clean rebuild destination; original raw inputs remain read-only')
    ARGS=parser.parse_args();OFFLINE=ARGS.offline
    if OFFLINE:
        if not RAW.is_dir():raise FileNotFoundError('Offline raw source directory missing: '+str(RAW))
    else:RAW.mkdir(parents=True,exist_ok=True)
    if ARGS.output_root:
        ROOT=ARGS.output_root.resolve()
        if ROOT!=SOURCE_ROOT and SOURCE_ROOT not in ROOT.parents:raise ValueError('Output root must remain inside the project directory')
        ROOT.mkdir(parents=True,exist_ok=True);(ROOT/'logs').mkdir(exist_ok=True)
        CLEAN=ROOT/'data_clean'/'minute';OUT=ROOT/'out'/'shared';LOG=ROOT/'logs'/'binance_run.md';MANIFEST=OUT/'download_manifest.csv'
        for p in [CLEAN,OUT]:p.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(ROOT/'logs'/'matplotlib_binance')
    np.random.seed(CONFIG['seed'])
    log('START root='+str(ROOT)+' stage='+ARGS.stage+' seed='+str(CONFIG['seed']))
    execution_failures=[]
    if ARGS.stage in ['all','markets']:
        for asset in ARGS.assets:
            for market in ['spot','futures/um']:
                try:build_market(asset,market)
                except Exception as e:
                    execution_failures.append(asset+' '+market+' '+repr(e));log('MARKET_FAILURE '+asset+' '+market+' '+repr(e))
    if ARGS.stage in ['all','leverage']:
        for asset in ARGS.assets:
            try:build_leverage(asset)
            except Exception as e:
                execution_failures.append(asset+' leverage '+repr(e));log('LEVERAGE_FAILURE '+asset+' '+repr(e))
    consolidate();measurement_qa();plot_signature();terminal_verification();log('FINISH')
    if OFFLINE and execution_failures:raise RuntimeError('Offline reconstruction failed: '+str(execution_failures))
    if OFFLINE and ARGS.stage in ['all','consolidate'] and ARGS.assets==ASSETS:
        receipt=ROOT/'logs/binance_terminal_verification.json'
        if not receipt.exists() or not json.loads(receipt.read_text()).get('all_pass'):
            raise RuntimeError('Offline terminal ingestion verification did not pass')
