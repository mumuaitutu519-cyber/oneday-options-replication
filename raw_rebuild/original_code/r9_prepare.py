from r9_common import *

def inventory():
 p=O/'protected_before.csv'
 files=[]
 for directory in ['code','out','paper','data_clean']:
  for f in (ROOT/directory).rglob('*'):
   if not f.is_file() or f.name.startswith('._') or '__pycache__' in f.parts:continue
   rel=str(f.relative_to(ROOT))
   if rel.startswith(('out/R9/','code/r9_')):continue
   s=f.stat();files.append(dict(path=rel,bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(f) if directory in ['code','paper'] else ''))
 for f in ROOT.glob('RESULTS*'):
  s=f.stat();files.append(dict(path=str(f.relative_to(ROOT)),bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=sha(f)))
 if not p.exists():save(files,'protected_before.csv')
 else:
  old=pd.read_csv(p).fillna('');cur=pd.DataFrame(files).fillna('');m=old.merge(cur,on='path',suffixes=('_before','_now'));assert len(m)==len(old) and (m.bytes_before==m.bytes_now).all() and (m.mtime_ns_before==m.mtime_ns_now).all() and (m.sha256_before==m.sha256_now).all(),'protected inputs drifted'
 log(f'保护清单 {len(files)} 文件；预注册SHA256 {sha(O/"PREREG.md")}')

def decomp(r,loc):
 n=len(r);rv=r@r;bv=np.pi/2*n/(n-1)*np.sum(abs(r[1:]*r[:-1]));mu=2**(2/3)*special.gamma(7/6)/np.sqrt(np.pi);tq=n*n/(n-2)/mu**3*np.sum(abs(r[2:]*r[1:-1]*r[:-2])**(4/3));med=np.pi/(6-4*np.sqrt(3)+np.pi)*n/(n-2)*np.sum(np.median(np.stack([abs(r[:-2]),abs(r[1:-1]),abs(r[2:])]),axis=0)**2);tr=np.sum(r[np.abs(r)<=4*loc]**2);rule=bool((abs(r)>4*loc).any());z=(1-bv/rv)/np.sqrt((np.pi**2/4+np.pi-5)/n*max(1,tq/max(bv*bv,1e-30))) if rv>0 and bv>0 else np.nan
 return dict(rv=rv,bv_raw=bv,bv=min(bv,rv),trv_raw=tr,trv=min(tr,rv),medrv_raw=med,medrv=min(med,rv),tq=tq,bns_z=z,jump1=z>stats.norm.ppf(.99),jump5=z>stats.norm.ppf(.95),jump_rule=rule,max_abs_return=np.max(abs(r)),n_returns=n)

def prepare_rv(c):
 grid=pd.read_parquet(ROOT/f'out/R3/shared/grid_{c}.parquet');price=pd.read_parquet(R5/f'prices_binance_perp_{c}.parquet').close;features=pd.read_parquet(R5/f'features_{c}.parquet');dates=pd.DatetimeIndex(features.date);rows=[]
 # Searchsorted slices avoid scanning the entire price history for every window.
 rvret=grid.logret.to_numpy();rvtime=grid.index;rvloc=np.sqrt((np.pi/2*(grid.logret.abs()*grid.logret.abs().shift(1))).rolling(12,min_periods=12).mean().shift(1)).to_numpy();lp=np.log(price.to_numpy());pfirst=price.index[0];minute_ns=60*10**9
 for variant in ['rv5','subsample','rv15']:
  for date in dates:
   origin=date+pd.Timedelta(hours=9)
   for h in TENORS:
    ends=[('fixed',origin+pd.Timedelta(days=h))]+([('expiry23',date+pd.Timedelta(days=1,hours=8))] if h==1 else [])
    for window,end in ends:
     if end>=pd.Timestamp('2026-10-07'):continue
     vals=[]
     for phase in range(5) if variant=='subsample' else [0]:
      if variant=='rv5':
       i=rvtime.searchsorted(origin,side='right');j=rvtime.searchsorted(end,side='right');r=rvret[i:j];local=rvloc[i:j]
       if len(r)!=int((end-origin).total_seconds()/300):continue
      else:
       step=15 if variant=='rv15' else 5;i=int((origin.value-pfirst.value)//minute_ns);j=int((end.value-pfirst.value)//minute_ns)
       history=90 if step==5 else 210
       if i-history<0 or j>=len(lp):continue
       pts=np.arange(i-history+phase,j+1,step);pts=pts[(pts<i)|((pts>i)&(pts<j))];pts=np.unique(np.r_[pts,i,j]);pp=lp[pts]
       if np.isnan(pp).any():continue
       rr=np.diff(pp);a=np.r_[np.nan,np.pi/2*np.abs(rr[1:]*rr[:-1])];loc=pd.Series(a).rolling(12,min_periods=12).mean().shift(1).to_numpy();keep=pts[1:]>i;r=rr[keep];local=np.sqrt(loc[keep])
      if len(r)<3 or np.isnan(r).any() or np.isnan(local).any():continue
      vals.append(decomp(r,local))
     if len(vals)!=(5 if variant=='subsample' else 1):continue
     rr={k:np.mean([x[k] for x in vals]) for k in vals[0]};rr.update(currency=c,date=date,tenor=h,observed_at=origin,target_end=end,variant=variant,window=window,classification='事后');rr['jump1']=rr['bns_z']>stats.norm.ppf(.99);rr['jump5']=rr['bns_z']>stats.norm.ppf(.95);rr['jump_rule']=any(x['jump_rule'] for x in vals);rows.append(rr)
   if date.day==1 and date.month in [1,7]:log(f'{c} {variant} RV {date.date()}')
 d=pd.DataFrame(rows);d.to_parquet(O/f'rv_{c}.parquet',index=False)
 one=d[(d.variant=='rv5')&(d.window=='expiry23')].set_index('date');parts=[]
 for h in TENORS:
  for date in dates:
   z=one.reindex(pd.date_range(date,periods=h))
   if z.rv.isna().any():continue
   r=dict(currency=c,date=date,tenor=h,variant='rv5',window='daily23_sum',classification='事后',observed_at=date+pd.Timedelta(hours=9),target_end=date+pd.Timedelta(days=h,hours=8))
   for k in ['rv','bv','trv','medrv','bv_raw','trv_raw','medrv_raw','n_returns']:r[k]=z[k].sum()
   for k in ['jump1','jump5','jump_rule']:r[k]=z[k].astype(bool).any()
   r['max_abs_return']=z.max_abs_return.max();parts.append(r)
 pd.DataFrame(parts).to_parquet(O/f'rv_daily23_{c}.parquet',index=False)
 log(f'{c} RV完成 {len(d)}窗口')

def integrate(g,F,sig,tau,cover=1.,minimum=5):
 g=g[((g.option_type=='put')&(g.strike<F))|((g.option_type=='call')&(g.strike>=F))];k=g.groupby('strike').agg(q=('usd','median'),iv=('sigma','median'));K=k.index.to_numpy(float);need=sig*np.sqrt(tau);r=dict(total=np.nan,center=np.nan,strike_n=len(K),coverage=False)
 if len(K)<minimum or K.min()>F*np.exp(-cover*need) or K.max()<F*np.exp(cover*need):return r
 x=K/F;y=k.q.to_numpy()/F
 if not np.any(np.isclose(x,1,atol=1e-12)):
  pos=np.searchsorted(x,1);x=np.insert(x,pos,1.);y=np.insert(y,pos,black([1.],[1.],[sig*sig*tau],['call'])[0])
 xx=np.concatenate([np.geomspace(x[j],x[j+1],17)[:-1] for j in range(len(x)-1)]+[x[-1:]]);yy=np.interp(xx,x,y);inside=2*np.trapezoid(yy/xx**2,xx)
 xl=np.geomspace(np.exp(-max(6.,12*np.sqrt(tau)*float(k.iv.iloc[0]))),x[0],160);xr=np.geomspace(x[-1],np.exp(max(6.,12*np.sqrt(tau)*float(k.iv.iloc[-1]))),160);yl=black(np.ones(len(xl)),xl,np.full(len(xl),float(k.iv.iloc[0])**2*tau),np.full(len(xl),'put'));yr=black(np.ones(len(xr)),xr,np.full(len(xr),float(k.iv.iloc[-1])**2*tau),np.full(len(xr),'call'));total=inside+2*(np.trapezoid(yl/xl**2,xl)+np.trapezoid(yr/xr**2,xr));bounds=np.exp(np.array([-1.,1.])*need);cx=xx[(xx>bounds[0])&(xx<bounds[1])];cx=np.r_[bounds[0],cx,bounds[1]];cy=np.interp(cx,x,y);center=2*np.trapezoid(cy/cx**2,cx)
 return dict(total=total,center=center,strike_n=len(K),coverage=True)

def prepare_quotes(c):
 meta=pd.read_parquet(ROOT/f'data_clean/option_instruments_{c}.parquet');extra=ROOT/f'out/R2/instrument_supplements_{c}.parquet'
 if extra.exists():meta=pd.concat([meta,pd.read_parquet(extra)],ignore_index=True)
 meta=meta.drop_duplicates('instrument_name')[['instrument_name','strike','expiration_timestamp','option_type','creation_timestamp']]
 reference=pd.read_parquet(R4/f'expiry_quotes_{c}.parquet');reference=reference[reference.window_hour==8];dates=sorted(reference.date.unique());rows=[];aud=[];inputs=[]
 configs=[('trade_native','trade',.05,1.,5,60),('mark_native','mark',.05,1.,5,60),('mark_band02','mark',.02,1.,5,60),('mark_band10','mark',.10,1.,5,60),('mark_cover15','mark',.05,1.5,5,60),('mark_strikes8','mark',.05,1.,8,60),('mark_last15','mark',.05,1.,5,15)]
 for i,date0 in enumerate(dates):
  date=pd.Timestamp(date0);origin=date+pd.Timedelta(hours=9);path=ROOT/f'out/R2/raw/{c}/{date:%Y-%m-%d}/trades.parquet'
  if not path.exists():continue
  inputs.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path),bytes=path.stat().st_size));raw=pd.read_parquet(path);raw=raw[(raw.timestamp>= (date+pd.Timedelta(hours=8)).value/1e6)&(raw.timestamp<origin.value/1e6)&(raw.price>0)&(raw.index_price>0)].copy()
  for col in ['combo_id','combo_trade_id','block_trade_id','block_rfq_id']:
   if col in raw:raw=raw[raw[col].isna()|raw[col].isin(['',0,False])]
  raw=raw.merge(meta,on='instrument_name',how='inner',validate='many_to_one');raw=raw[(raw.creation_timestamp<=raw.timestamp)&(raw.expiration_timestamp>raw.timestamp)].copy();raw['F']=raw.index_price;raw['trade_F']=pd.to_numeric(raw.get('underlying_price',raw.index_price),errors='coerce');raw['trade_F']=raw.trade_F.where(raw.trade_F>0,raw.index_price);raw['tau']=(raw.expiration_timestamp-raw.timestamp)/(365*86400000)
  for pricekind in ['trade','mark']:
   usd=raw['price' if pricekind=='trade' else 'mark_price']*raw.index_price;F=raw.trade_F if pricekind=='trade' else raw.F;raw[pricekind+'_w']=implied(F,raw.strike,usd,raw.option_type);raw[pricekind+'_sigma']=np.sqrt(raw[pricekind+'_w']/raw.tau);raw[pricekind+'_usd']=usd
  for variant,kind,band,cover,mink,mins in configs:
   g=raw[raw[kind+'_w'].notna()&(raw.timestamp>=(origin-pd.Timedelta(minutes=mins)).value/1e6)].copy();g['sigma']=g[kind+'_sigma'];g['usd']=g[kind+'_usd'];g['ref']=g.trade_F if kind=='trade' else g.F;g['lm']=np.log(g.strike/g.ref)
   for exp,z in g.groupby('expiration_timestamp'):
    tau=(exp-origin.value/1e6)/(365*86400000);atm=z[z.lm.abs()<=band]
    if len(atm)<3 or tau<=0:continue
    sig=atm.sigma.median();F=atm.ref.median();r=integrate(z,F,sig,tau,cover,mink);rows.append(dict(currency=c,date=date,variant=variant,observed_at=origin,expiry=pd.Timestamp(exp,unit='ms'),atm=sig*sig*tau,last_trade=z.timestamp.max(),n_atm=len(atm),F=F,**r))
  if (i+1)%250==0:log(f'{c} 报价积分 {i+1}/{len(dates)}')
 q=pd.DataFrame(rows);q.to_parquet(O/f'quotes_{c}.parquet',index=False);save(inputs,f'raw_source_manifest_{c}.csv');r=q[q.variant=='trade_native'].merge(reference[['date','expiry','atm_w','mf_w']],on=['date','expiry']);r['atm_error']=r.atm-r.atm_w;r['total_error']=r.total-r.mf_w;save(r[['currency','date','expiry','atm_error','total_error']],f'quote_reconciliation_{c}.csv');log(f'{c} 报价完成 {len(q)}桶')

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['all','rv','quotes'],default='all');a=ap.parse_args();inventory()
 import platform,importlib.metadata
 (O/'runtime.json').write_text(json.dumps(dict(python=sys.version,executable=sys.executable,platform=platform.platform(),seed=SEED,reps=REPS,packages={k:importlib.metadata.version(k) for k in ['numpy','pandas','scipy','pyarrow','matplotlib']}),indent=2))
 cache=O/'prepare_receipt.json'
 if a.stage=='all' and cache.exists() and os.environ.get('R9_REBUILD')!='1':
  receipt=json.loads(cache.read_text());valid=all((ROOT/r['path']).exists() and sha(ROOT/r['path'])==r['sha256'] for r in receipt['artifacts'])
  for manifest in ['source_manifest.csv','raw_source_manifest_BTC.csv','raw_source_manifest_ETH.csv']:
   if not valid:break
   valid=all((ROOT/r.path).exists() and sha(ROOT/r.path)==r.sha256 for r in pd.read_csv(O/manifest).itertuples())
  if valid:
   log('原始输入与已验证中间量哈希一致；复用报价与RV分解，重新计算全部统计与报告。R9_REBUILD=1可从原始记录重建中间量。');sys.exit(0)
 # Source manifests include all price / feature / strategy / event inputs, hashes.
 sources=[]
 for c in ['BTC','ETH']:
  for p in [R5/f'features_{c}.parquet',R5/f'premium_panel_{c}.parquet',R5/f'prices_binance_perp_{c}.parquet',R5/f'strategy_path_{c}.parquet',R4/f'expiry_quotes_{c}.parquet',ROOT/f'out/R3/shared/grid_{c}.parquet',R5/f'expiry_quotes_{c}.parquet',R5/f'term_panel_{c}.parquet',ROOT/f'data_clean/option_instruments_{c}.parquet',ROOT/f'out/R2/instrument_supplements_{c}.parquet']:
   if p.exists():sources.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
  for p in [R5/f'strategy_availability_{c}.csv',R5/f'measurement_panel_{c}_binance_perp_subsample.parquet',R5/f'measurement_panel_{c}_binance_perp_rv15.parquet']:sources.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
 for p in [R5/'term_premium.csv',R5/'strategy_summary.csv',ROOT/'out/R3/shared/events_publication.csv',ROOT/'out/R3/shared/events.csv',ROOT/'out/R3/shared/source_requests.jsonl',O/'PREREG.md']:sources.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size))
 save(sources,'source_manifest.csv')
 for c in ['BTC','ETH']:
  if a.stage in ['all','rv']:prepare_rv(c)
  if a.stage in ['all','quotes']:prepare_quotes(c)
