from r4_common import *
from r2_core import implied_w,black
TENORS=[1,2,3,7,14,30];WINDOWS=[8,9,12,16]

def fullmeta(currency):
 d=pd.read_parquet(ROOT/f'data_clean/option_instruments_{currency}.parquet')
 p=ROOT/f'out/R2/instrument_supplements_{currency}.parquet'
 if p.exists():d=pd.concat([d,pd.read_parquet(p)],ignore_index=True)
 return d.drop_duplicates('instrument_name')[['instrument_name','strike','expiration_timestamp','option_type','creation_timestamp']]

def mfvariance(g,ref,sigma,tau):
 z=g[((g.option_type=='put')&(g.strike<ref))|((g.option_type=='call')&(g.strike>=ref))].copy()
 k=z.groupby('strike').agg(q=('actual_usd','median'),iv=('sigma','median'));K=k.index.to_numpy(float);need=sigma*np.sqrt(tau)
 info=dict(mf_w=np.nan,strike_n=len(K),kmin=float(K.min()/ref) if len(K) else np.nan,kmax=float(K.max()/ref) if len(K) else np.nan,coverage_ok=False,tail_extended=False,tail_share=np.nan)
 if len(K)<5 or K.min()>ref*np.exp(-need) or K.max()<ref*np.exp(need):return info
 x=K/ref;y=k.q.to_numpy()/ref
 # Integral of actual OTM prices within observed support. Boundary wings use flat IV.
 lower=np.exp(-max(6.,12*np.sqrt(tau)*float(k.iv.iloc[0])));upper=np.exp(max(6.,12*np.sqrt(tau)*float(k.iv.iloc[-1])))
 xl=np.geomspace(lower,x[0],160);xr=np.geomspace(x[-1],upper,160)
 yl=black(np.ones(len(xl)),xl,np.repeat(float(k.iv.iloc[0])**2*tau,len(xl)),np.repeat('put',len(xl)))
 yr=black(np.ones(len(xr)),xr,np.repeat(float(k.iv.iloc[-1])**2*tau,len(xr)),np.repeat('call',len(xr)))
 # Add model ATM connection at F if missing; observed prices remain authoritative elsewhere.
 if not np.any(np.isclose(x,1,atol=1e-12)):
  mid=black(np.array([1.]),np.array([1.]),np.array([sigma**2*tau]),np.array(['call']))[0]
  pos=np.searchsorted(x,1);x=np.insert(x,pos,1.);y=np.insert(y,pos,mid)
 # 16 logarithmic quadrature nodes per observed interval; actual price linear in strike.
 xx=np.concatenate([np.geomspace(x[j],x[j+1],17)[:-1] for j in range(len(x)-1)]+[x[-1:]])
 yy=np.interp(xx,x,y);inside=2*np.trapezoid(yy/xx**2,xx);tails=2*(np.trapezoid(yl/xl**2,xl)+np.trapezoid(yr/xr**2,xr));value=inside+tails
 info.update(mf_w=value,coverage_ok=True,tail_extended=True,tail_share=tails/value if value>0 else np.nan);return info

def rowquote(g,when,expiry):
 atm=g[g.log_moneyness.abs()<=.05];tau=(expiry-when.value/1e6)/(365*86400000)
 if len(atm)<3 or tau<=0:return None
 sig=float(atm.sigma.median());ref=float(atm.F.median());r=dict(observed_at=when,expiry=pd.Timestamp(expiry,unit='ms'),expiry_ms=int(expiry),tau=tau,atm_w=sig**2*tau,atm_sigma=sig,F=ref,n_atm=len(atm),n_buy=int((atm.direction=='buy').sum()),n_sell=int((atm.direction=='sell').sum()),last_trade=int(atm.timestamp.max()),buy_w=np.nan,sell_w=np.nan)
 for direction in ['buy','sell']:
  z=atm[atm.direction==direction]
  if len(z)>=3:r[direction+'_w']=float(z.sigma.median())**2*tau
 r.update(mfvariance(g,ref,sig,tau));return r

def brackets(z,target):
 lo=z[z.expiry<=target];hi=z[z.expiry>=target]
 if not len(lo) or not len(hi):return None
 l=lo.iloc[-1];u=hi.iloc[0]
 if l.expiry==u.expiry:
  above=z[z.expiry>target];below=z[z.expiry<target]
  if len(above):u=above.iloc[0]
  elif len(below):l=below.iloc[-1]
  else:return None
 if l.expiry==u.expiry:return None
 weight=float((target-l.expiry)/(u.expiry-l.expiry));return l,u,weight

def run(currency):
 dest=OUT/f'chen/expiry_quotes_{currency}.parquet'
 if not FORCE and dest.exists() and (OUT/f'chen/options_complete_{currency}.json').exists():return
 meta=fullmeta(currency);state=json.loads((ROOT/f'out/R2/download_{currency}.json').read_text());quotes=[];dailyinfo=[];markparts=[]
 dates=pd.date_range(state['continuous_start'],min(state['latest'],'2026-10-06'))
 for i,date in enumerate(dates):
  raw=rawday(currency,date.strftime('%Y-%m-%d'))
  if not len(raw):continue
  ts=pd.to_datetime(raw.timestamp,unit='ms');take=ts.dt.hour.isin(WINDOWS);raw=raw[take].copy()
  for name in ['block_trade_id','block_rfq_id','combo_id','combo_trade_id']:
   if name in raw:raw=raw[raw[name].isna()|raw[name].isin(['',0,False])]
  raw=raw.merge(meta,on='instrument_name',how='inner',validate='many_to_one');raw=raw[(raw.expiration_timestamp>raw.timestamp)&(raw.creation_timestamp<=raw.timestamp)&(raw.price>0)&(raw.index_price>0)]
  if not len(raw):continue
  raw['F']=pd.to_numeric(raw.get('underlying_price',raw.index_price),errors='coerce');raw['F']=raw.F.where(raw.F>0,raw.index_price)
  raw['actual_usd']=raw.price*raw.index_price;raw['tau']=(raw.expiration_timestamp-raw.timestamp)/(365*86400000)
  w,valid=implied_w(raw.F,raw.strike,raw.actual_usd,raw.option_type);raw['sigma']=np.sqrt(w/raw.tau);raw=raw[valid].copy();raw['log_moneyness']=np.log(raw.strike/raw.F);raw['hour']=pd.to_datetime(raw.timestamp,unit='ms').dt.hour
  for hour,g0 in raw.groupby('hour'):
   when=date+pd.Timedelta(hours=int(hour)+1)
   for expiry,g in g0.groupby('expiration_timestamp'):
    r=rowquote(g,when,expiry)
    if r:quotes.append(dict(currency=currency,date=date,window_hour=int(hour),**r))
   if hour==8:
    atm=g0[(g0.log_moneyness.abs()<=.05)&(g0.expiration_timestamp==(date+pd.Timedelta(days=1,hours=8)).value/1e6)].copy()
    dailyinfo.append(dict(currency=currency,date=date,atm_trades=len(atm),buy_n=int((atm.direction=='buy').sum()),sell_n=int((atm.direction=='sell').sum()),raw_observation_n=len(g0)))
    if len(atm) and 'mark_price' in atm:
     mw,mvalid=implied_w(atm.F,atm.strike,atm.mark_price*atm.index_price,atm.option_type);atm['mark_iv']=np.sqrt(mw/atm.tau);atm['mark_valid']=mvalid;atm['difference_volpts']=100*(atm.sigma-atm.mark_iv)
     markparts.append(atm[['trade_id','timestamp','direction','sigma','mark_iv','mark_valid','difference_volpts','price','mark_price']].assign(currency=currency,date=date))
  if (i+1)%100==0:log(f'辰 原始全期限 {currency} {i+1}/{len(dates)}日')
 q=pd.DataFrame(quotes);q.to_parquet(dest,index=False);csv(dailyinfo,f'chen/data_daily_{currency}.csv')
 pd.concat(markparts,ignore_index=True).to_parquet(OUT/f'chen/trade_mark_{currency}.parquet',index=False)
 a=grid(currency);pan=[];availability=[]
 for (date,hour),g in q.groupby(['date','window_hour']):
  g=g.sort_values('expiry');when=date+pd.Timedelta(hours=int(hour)+1)
  for tenor in TENORS:
   target=when+pd.Timedelta(days=tenor);b=brackets(g,target)
   if b is None:availability.append(dict(currency=currency,date=date,window_hour=hour,tenor=tenor,status='two_expiries_unavailable'));continue
   l,u,weight=b;r=dict(currency=currency,date=date,window_hour=hour,tenor=tenor,observed_at=when,target_end=target,lo_expiry=l.expiry,hi_expiry=u.expiry,hi_weight=weight,n_atm=int(l.n_atm+u.n_atm),last_trade=max(l.last_trade,u.last_trade),tail_extended=bool(l.tail_extended or u.tail_extended),mf_coverage=bool(l.coverage_ok and u.coverage_ok),tail_share=(1-weight)*l.tail_share+weight*u.tail_share)
   for metric in ['atm_w','mf_w','buy_w','sell_w']:r[metric]=(1-weight)*l[metric]+weight*u[metric] if np.isfinite(l[metric]) and np.isfinite(u[metric]) else np.nan
   r['rv_total']=rv_interval(a,when,target) if target<CUTOFF else np.nan;r['iv_annual']=r['atm_w']*365/tenor;r['rv_annual']=r['rv_total']*365/tenor;r['vrp']=r['atm_w']-r['rv_total'];r['vrp_annual']=r['iv_annual']-r['rv_annual'];r['volpts']=100*(np.sqrt(r['iv_annual'])-np.sqrt(r['rv_annual']));pan.append(r)
 p=pd.DataFrame(pan);p.to_parquet(OUT/f'chen/term_panel_{currency}.parquet',index=False);csv(availability,f'chen/term_availability_{currency}.csv')
 (OUT/f'chen/options_complete_{currency}.json').write_text(json.dumps(dict(seed=SEED,quote_rows=len(q),term_rows=len(p),dates=len(dates))))
 log(f'辰 {currency} 原始成交统一期限表完成，{len(q)} 到期日桶、{len(p)} 固定期限行')

def main():
 from concurrent.futures import ProcessPoolExecutor
 with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(run,['BTC','ETH']))
if __name__=='__main__':main()
