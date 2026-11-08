from r5_common import *
from r4_common import grid,rv_interval
from r4_strategy import quote_pair
VARIANTS=['mark','bilateral','last15','min10','band02','band10','blocks','weighted','dropmove1']

def weighted_median(x,w):
 o=np.argsort(x);x=np.asarray(x)[o];w=np.asarray(w)[o];return x[np.searchsorted(np.cumsum(w),w.sum()/2)] if np.isfinite(w).all() and (w>=0).all() and w.sum()>0 else np.nan

def bucket(g,when,expiry,variant):
 band=.02 if variant=='band02' else .10 if variant=='band10' else .05
 z=g[np.abs(g.log_moneyness)<=band];minimum=10 if variant=='min10' else 3
 if len(z)<minimum:return None
 tau=(expiry-when.value/1e6)/(365*86400000)
 if tau<=0:return None
 if variant=='bilateral':
  buy=z[z.direction=='buy'];sell=z[z.direction=='sell']
  if min(len(buy),len(sell))<3:return None
  sigma=(buy.sigma.median()+sell.sigma.median())/2
 elif variant=='weighted':sigma=weighted_median(z.sigma.to_numpy(),pd.to_numeric(z.amount,errors='coerce').to_numpy())
 else:sigma=z.sigma.median()
 if not np.isfinite(sigma):return None
 return dict(observed_at=when,expiry=pd.Timestamp(expiry,unit='ms'),expiry_ms=int(expiry),tau=tau,atm_sigma=sigma,atm_w=sigma*sigma*tau,F=z.F.median(),n_atm=len(z),n_buy=int((z.direction=='buy').sum()),n_sell=int((z.direction=='sell').sum()),last_trade=int(z.timestamp.max()))

def leg_pair(raw,date,mode):
 when=date+pd.Timedelta(hours=9);expiry=date+pd.Timedelta(days=1,hours=8);g=raw[(raw.timestamp>=(date+pd.Timedelta(hours=8)).value/1e6)&(raw.timestamp<when.value/1e6)]
 if mode in ['sell','buy']:g=g[g.direction==mode]
 q=quote_pair(g,expiry.value/1e6,when)
 if q is None:return None
 legs=[g[(g.expiration_timestamp==expiry.value/1e6)&(g.strike==q['strike'])&(g.option_type==typ)].sort_values('timestamp').iloc[-1] for typ in ['call','put']]
 if mode=='mark':
  if not all(bool(r.mark_valid) for r in legs):return None
  q['premium']=sum(r.mark_price*r.index_price for r in legs);q['sigma']=np.mean([r.mark_sigma for r in legs])
 spreads=[]
 for r in legs:
  z=raw[(raw.instrument_name==r.instrument_name)&(raw.timestamp<=when.value/1e6)].sort_values('timestamp');v=z.price.diff().dropna().to_numpy()
  spreads.append(np.sqrt(max(-np.cov(v[:-1],v[1:])[0,1],0))*q['F'] if len(v)>=10 else np.nan)
 q['halfspread']=float(np.sum(spreads));q['spread_known']=bool(np.isfinite(spreads).all());q['call_premium']=legs[0].price*legs[0].index_price if mode!='mark' else legs[0].mark_price*legs[0].index_price;q['put_premium']=legs[1].price*legs[1].index_price if mode!='mark' else legs[1].mark_price*legs[1].index_price
 return dict(currency=raw.currency.iloc[0],date=date,entry=when,expiry=expiry,entry_mode=mode,**q)

def run(c):
 done=CHEN/f'options_complete_{c}.json'
 if done.exists() and not FORCE:return
 meta=fullmeta(c);base=pd.read_parquet(R4/f'expiry_quotes_{c}.parquet');base=base[base.window_hour==8];dates=sorted(base.date.unique());qrows=[];flatrows=[];legs=[];status=[];quality=[];a=grid(c)
 for i,date0 in enumerate(dates):
  date=pd.Timestamp(date0);when=date+pd.Timedelta(hours=9);raw=rawday(c,date);raw=raw[(raw.timestamp>=date.value/1e6)&(raw.timestamp<when.value/1e6)&(raw.price>0)&(raw.index_price>0)].copy()
  for col in ['combo_id','combo_trade_id']:
   if col in raw:raw=raw[raw[col].isna()|raw[col].isin(['',0,False])]
  raw=raw.merge(meta,on='instrument_name',how='inner',validate='many_to_one');raw=raw[(raw.creation_timestamp<=raw.timestamp)&(raw.expiration_timestamp>raw.timestamp)].copy();raw['F']=raw.index_price;raw['actual_usd']=raw.price*raw.index_price;raw['tau']=(raw.expiration_timestamp-raw.timestamp)/(365*86400000);raw['currency']=c
  w,valid=implied_w(raw.F,raw.strike,raw.actual_usd,raw.option_type);raw['sigma']=np.sqrt(w/raw.tau);raw['trade_valid']=valid;mw,mv=implied_w(raw.F,raw.strike,raw.mark_price*raw.index_price,raw.option_type);raw['mark_sigma']=np.sqrt(mw/raw.tau);raw['mark_valid']=mv;raw['log_moneyness']=np.log(raw.strike/raw.F)
  raw['block']=False
  for col in ['block_trade_id','block_rfq_id']:
   if col in raw:raw['block']|=~(raw[col].isna()|raw[col].isin(['',0,False]))
  regular=raw[~raw.block];actual=regular[regular.trade_valid];window=actual[actual.timestamp>=(date+pd.Timedelta(hours=8)).value/1e6]
  atm=window[(window.expiration_timestamp==(date+pd.Timedelta(days=1,hours=8)).value/1e6)&(window.log_moneyness.abs()<=.05)]
  quality.append(dict(currency=c,date=date,trade_n=len(atm),sell_n=int((atm.direction=='sell').sum()),buy_n=int((atm.direction=='buy').sum()),mark_valid_n=int(atm.mark_valid.sum()),trade_mark_mean=(100*(atm.sigma-atm.mark_sigma)).where(atm.mark_valid).mean(),trade_mark_median=(100*(atm.sigma-atm.mark_sigma)).where(atm.mark_valid).median()))
  p0=a.close.get(date+pd.Timedelta(hours=8),np.nan);p1=a.close.get(when,np.nan);move=abs(np.log(p1/p0)) if p0>0 and p1>0 else np.nan
  for variant in VARIANTS:
   if variant=='mark':g=regular[(regular.timestamp>=(date+pd.Timedelta(hours=8)).value/1e6)&regular.mark_valid].copy();g['sigma']=g.mark_sigma
   elif variant=='blocks':g=raw[(raw.timestamp>=(date+pd.Timedelta(hours=8)).value/1e6)&raw.trade_valid].copy()
   elif variant=='last15':g=window[window.timestamp>=(date+pd.Timedelta(hours=8,minutes=45)).value/1e6]
   elif variant=='dropmove1' and (not np.isfinite(move) or move>.01):g=window.iloc[:0]
   else:g=window
   count=0
   for expiry,z in g.groupby('expiration_timestamp'):
    r=bucket(z,when,expiry,variant)
    if r:qrows.append(dict(currency=c,date=date,window_hour=8,variant=variant,**r));count+=1
   status.append(dict(currency=c,date=date,variant=variant,expiry_buckets=count,window_move=move,status='estimated' if count else 'no_eligible_expiry'))
  for q in base[(base.date==date)&base.coverage_ok].itertuples(index=False):
   g=window[window.expiration_timestamp==q.expiry_ms].copy();g['actual_usd']=black(np.repeat(q.F,len(g)),g.strike,np.repeat(q.atm_w,len(g)),g.option_type);g['sigma']=q.atm_sigma;r=mfvariance(g,q.F,q.atm_sigma,q.tau)
   if r['strike_n']!=q.strike_n:raise AssertionError((c,date,q.expiry,r['strike_n'],q.strike_n))
   flatrows.append(dict(currency=c,date=date,expiry=q.expiry,flat_grid_w=r['mf_w'],atm_w=q.atm_w,actual_mf_w=q.mf_w,strike_n=q.strike_n,flat_grid_ratio=r['mf_w']/q.atm_w,variant='flat_grid'))
  for mode in ['last','sell','buy','mark']:
   r=leg_pair(actual,date,mode)
   status.append(dict(currency=c,date=date,variant='strategy_'+mode,status='estimated' if r else 'same_strike_two_legs_unavailable'))
   if r:legs.append(r)
  if (i+1)%200==0:log(f'R5 option quotes {c}: {i+1}/{len(dates)} dates')
 q=pd.DataFrame(qrows);q.to_parquet(CHEN/f'expiry_quotes_{c}.parquet',index=False);csv(flatrows,f'flat_grid_expiry_{c}.csv');csv(status,f'option_availability_{c}.csv');csv(quality,f'trade_quality_{c}.csv');csv(legs,f'strategy_entries_{c}.csv')
 pans=[]
 for (variant,date),g in q.groupby(['variant','date']):
  g=g.sort_values('expiry');when=date+pd.Timedelta(hours=9)
  for h in TENORS:
   target=when+pd.Timedelta(days=h);b=brackets(g,target)
   if b is None:continue
   l,u,weight=b;w=(1-weight)*l.atm_w+weight*u.atm_w;rv=rv_interval(a,when,target) if target<CUTOFF else np.nan
   pans.append(dict(currency=c,date=date,window_hour=8,variant=variant,tenor=h,observed_at=when,target_end=target,lo_expiry=l.expiry,hi_expiry=u.expiry,hi_weight=weight,n_atm=int(l.n_atm+u.n_atm),last_trade=max(l.last_trade,u.last_trade),atm_w=w,rv_total=rv,iv_annual=w*365/h,rv_annual=rv*365/h,vrp=w-rv,vrp_annual=(w-rv)*365/h,volpts=100*(np.sqrt(w*365/h)-np.sqrt(rv*365/h))))
 pd.DataFrame(pans).to_parquet(CHEN/f'term_panel_{c}.parquet',index=False)
 bterm=pd.read_parquet(R4/f'term_panel_{c}.parquet');bterm=bterm[(bterm.window_hour==8)&bterm.mf_coverage];flat=pd.DataFrame(flatrows);f=flat.set_index(['date','expiry']).flat_grid_w;diagnostic=[]
 for r in bterm.itertuples(index=False):
  lo=f.get((r.date,r.lo_expiry),np.nan);hi=f.get((r.date,r.hi_expiry),np.nan);fw=(1-r.hi_weight)*lo+r.hi_weight*hi;correct=r.mf_w-(fw-r.atm_w)
  diagnostic.append(dict(currency=c,date=r.date,tenor=r.tenor,variant='flat_grid',atm_w=r.atm_w,actual_mf_w=r.mf_w,flat_price_grid_w=fw,actual_ratio=r.mf_w/r.atm_w,flat_grid_ratio=fw/r.atm_w,flat_grid_error_pct=100*(fw/r.atm_w-1),corrected_mf_w=correct,multiplicative_mf_w=r.mf_w/(fw/r.atm_w),nonpositive=correct<=0,rv_total=r.rv_total))
 csv(diagnostic,f'numerical_grid_diagnostics_{c}.csv');done.write_text(json.dumps(dict(currency=c,quote_rows=len(q),term_rows=len(pans),entry_rows=len(legs),raw_dates=len(dates))));log(f'R5 {c} mark/filters/full-grid completed')

if __name__=='__main__':
 for c in ['BTC','ETH']:run(c)
