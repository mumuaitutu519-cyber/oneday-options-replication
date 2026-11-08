from r4_common import *
from r4_options import fullmeta,brackets

FREQUENCIES={'hourly':1,'8hour':8,'unhedged':None}

def quote_pair(g,expiry,when):
 z=g[(g.expiration_timestamp==expiry)&(np.abs(np.log(g.strike/g.F))<=.05)];valid=[]
 for k,v in z.groupby('strike'):
  if {'call','put'}<=set(v.option_type):valid.append((abs(np.log(k/v.F.median())),k,v))
 if not valid:return None
 _,k,z=min(valid,key=lambda x:x[0]);legs=[];spreads=[]
 for typ in ['call','put']:
  r=z[z.option_type==typ].sort_values('timestamp').iloc[-1];legs.append(r);alltr=g[(g.instrument_name==r.instrument_name)&(g.timestamp<=when.value/1e6)].sort_values('timestamp');changes=alltr.price.diff().dropna().to_numpy()
  spreads.append(float(np.sqrt(max(-np.cov(changes[:-1],changes[1:])[0,1],0))*r.index_price) if len(changes)>=10 else np.nan)
 return dict(strike=k,premium=sum(r.actual_usd for r in legs),sigma=np.mean([r.sigma for r in legs]),F=np.mean([r.F for r in legs]),first_trade=min(r.timestamp for r in legs),last_trade=max(r.timestamp for r in legs),halfspread=np.sum(spreads),spread_known=np.isfinite(spreads).all(),call_trade_id=str(legs[0].trade_id),put_trade_id=str(legs[1].trade_id))

def cashbook(c,when,expiry,q,frequency):
 a=grid(c);deltafreq=FREQUENCIES[frequency]
 times=pd.DatetimeIndex([when,expiry]) if deltafreq is None else pd.date_range(when,expiry,freq=f'{deltafreq}h').union(pd.DatetimeIndex([expiry]));S=a.close.reindex(times).to_numpy();delivery=delivery_prices(c).get(int(expiry.value/1e6),np.nan)
 if not np.isfinite(S).all():return None,'underlying_boundary_unavailable'
 if not np.isfinite(delivery):return None,'official_delivery_unavailable'
 tau=(expiry-times[:-1]).total_seconds().to_numpy()/(365*86400);w=np.maximum(q['sigma']**2*tau,EPS);d=np.zeros(len(times)-1) if deltafreq is None else 2*special.ndtr((np.log(S[:-1]/q['strike'])+w/2)/np.sqrt(w))-1;hedge=float(d@np.diff(S));payoff=abs(delivery-q['strike']);gamma=payoff-hedge;turn=np.abs(d[0])*S[0]+np.abs(np.diff(d))@S[1:-1]+np.abs(d[-1])*S[-1];fund=funding_events(c);fund=fund[(fund.index>when)&(fund.index<=expiry)];positions=np.searchsorted(times.to_numpy(),fund.index.to_numpy(),side='left')-1;marks=pd.to_numeric(fund.markPrice,errors='coerce').to_numpy();marks=np.where(np.isfinite(marks),marks,a.close.reindex(fund.index.floor('5min')).to_numpy())
 if np.any(~np.isfinite(marks)&(d[positions]!=0)):return None,'funding_notional_unavailable'
 funding_cash=np.where(d[positions]==0,0,d[positions]*marks*fund.last_funding_rate.to_numpy());fcost=float(np.sum(funding_cash));spread=q['halfspread']/S[0];rate_turn=turn/S[0]+2*q['F']/S[0];p=q['premium']/S[0];g=gamma/S[0];f=fcost/S[0];r=dict(premium=p,gamma=g,hedge=hedge/S[0],payoff=payoff/S[0],funding=f,gross=p-g-f,hedge_turnover=turn/S[0],option_turnover=2*q['F']/S[0],cost_turnover=rate_turn,halfspread=spread,delivery=delivery,spot0=S[0],strike=q['strike'],sigma=q['sigma'],funding_n=len(fund),first_trade=q['first_trade'],last_trade=q['last_trade'],call_trade_id=q['call_trade_id'],put_trade_id=q['put_trade_id'])
 # Exact realized option-price identity, separate from the variance approximation.
 wp=q['sigma']**2*(expiry-when).total_seconds()/(365*86400);prices=2*special.ndtr((np.log(S/q['strike'])+np.maximum(q['sigma']**2*(expiry-times).total_seconds().to_numpy()/(365*86400),EPS)/2)/np.sqrt(np.maximum(q['sigma']**2*(expiry-times).total_seconds().to_numpy()/(365*86400),EPS)))-1
 nodew=np.maximum(q['sigma']**2*tau,EPS);gam=2*np.exp(-.5*((np.log(S[:-1]/q['strike'])+nodew/2)/np.sqrt(nodew))**2)/np.sqrt(2*np.pi)/(S[:-1]*np.sqrt(nodew));dt=np.diff(times.view('int64'))/1e9/(365*86400);variance_bridge=.5*np.sum(gam*S[:-1]**2*(q['sigma']**2*dt-(np.diff(np.log(S)))**2))/S[0] if deltafreq is not None else np.nan;r['gamma_variance_bridge']=variance_bridge;r['bridge_residual']=r['gross']+r['funding']-variance_bridge
 book=pd.DataFrame(dict(currency=c,entry=when,expiry=expiry,frequency=frequency,time=times[:-1],end=times[1:],S0=S[:-1],S1=S[1:],delta=d,hedge_gain=d*np.diff(S),strike=q['strike'],sigma=q['sigma'],delivery_price=delivery))
 fundbook=pd.DataFrame(dict(currency=c,entry=when,expiry=expiry,frequency=frequency,time=fund.index,rate=fund.last_funding_rate.to_numpy(),mark=marks,delta=d[positions],funding_cash=funding_cash))
 return (r,book,fundbook),'estimated'

def signal(c):
 from r4_forecast import features,ols_predict
 d=features(c,8);q=pd.read_parquet(OUT/f'chen/expiry_quotes_{c}.parquet');q=q[(q.window_hour==8)&(q.expiry==q.date+pd.Timedelta(days=1,hours=8))].set_index('date');d['iv']=q.atm_w.reindex(d.date).to_numpy();a=grid(c);d['label_end']=d.date+pd.Timedelta(days=1,hours=8);d['rv']=[rv_interval(a,w,e) if e<CUTOFF else np.nan for w,e in zip(d.observed_at,d.label_end)];d['signed_vrp']=d.iv-d.rv;X=d[['rv24','rv7','rv30','rv1','rv8','iv','dvol','prior_return']].to_numpy();y=d.signed_vrp.to_numpy();good=np.isfinite(X).all(1)&np.isfinite(y);forecast=np.full(len(d),np.nan);trainends=[]
 for t in range(1000,len(d)):
  if not np.isfinite(X[t]).all():continue
  tr=np.flatnonzero(good[max(0,t-1000):t])+max(0,t-1000);tr=tr[d.label_end.iloc[tr].to_numpy()<=d.observed_at.iloc[t].to_datetime64()]
  if len(tr)<200:continue
  xx=X[tr];mu=xx.mean(0);sd=xx.std(0);sd[sd<1e-15]=1;A=np.column_stack([np.ones(len(tr)),(xx-mu)/sd]);b=np.linalg.lstsq(A,y[tr],rcond=1e-10)[0];forecast[t]=np.r_[1.,(X[t]-mu)/sd]@b;trainends.append(dict(currency=c,date=d.date.iloc[t],origin=d.observed_at.iloc[t],last_label=d.label_end.iloc[tr[-1]],ntrain=len(tr)))
 d['forecast']=forecast;csv(d,f'chen/vrp_signal_{c}.csv');csv(trainends,f'chen/vrp_signal_audit_{c}.csv');prior=pd.read_csv(ROOT/f'out/R3/chen/vrp_forecast_{c}.csv',parse_dates=['date']);prior=prior.merge(d[['date','signed_vrp','forecast','iv','rv']],on='date',how='left',validate='one_to_one');csv(prior,f'chen/vrp_signal_comparison_{c}.csv');return d.set_index('date').forecast

def run(c):
 complete=OUT/f'chen/strategy_complete_{c}.json'
 if not FORCE and complete.exists():return
 quotes=pd.read_parquet(OUT/f'chen/expiry_quotes_{c}.parquet');quotes=quotes[quotes.window_hour==8];tp=pd.read_parquet(OUT/f'chen/term_panel_{c}.parquet');tp=tp[(tp.window_hour==8)&tp.tenor.isin([7,30])];meta=fullmeta(c);signals=signal(c);results=[];attempts=[];books=[];fundbooks=[]
 for i,(date,z) in enumerate(quotes.groupby('date')):
  when=date+pd.Timedelta(hours=9)
  if when<grid(c).index.min() or when>=CUTOFF:continue
  raw=rawday(c,date.strftime('%Y-%m-%d')).copy()
  for col in ['block_trade_id','block_rfq_id','combo_id','combo_trade_id']:
   if col in raw:raw=raw[raw[col].isna()|raw[col].isin(['',0,False])]
  raw=raw[(raw.timestamp>=date.value/1e6)&(raw.timestamp<when.value/1e6)&(raw.price>0)&(raw.index_price>0)].merge(meta,on='instrument_name',how='inner',validate='many_to_one');raw['F']=raw.index_price;raw['actual_usd']=raw.price*raw.index_price
  from r2_core import implied_w
  w,valid=implied_w(raw.F,raw.strike,raw.actual_usd,raw.option_type);raw['sigma']=np.sqrt(w/((raw.expiration_timestamp-raw.timestamp)/(365*86400000)));raw=raw[valid].copy();raw['hour']=pd.to_datetime(raw.timestamp,unit='ms').dt.hour
  for tenor in [1,7,30]:
   if tenor==1:legs=[(date+pd.Timedelta(days=1,hours=8),1.)]
   else:
    row=tp[(tp.date==date)&(tp.tenor==tenor)]
    if not len(row):attempts.append(dict(currency=c,date=date,tenor=tenor,status='two_expiries_unavailable'));continue
    r=row.iloc[0];legs=[(r.lo_expiry,1-r.hi_weight),(r.hi_expiry,r.hi_weight)]
   if any(e>=CUTOFF for e,weight in legs if weight>0):attempts.append(dict(currency=c,date=date,tenor=tenor,status='after_data_cutoff'));continue
   qs=[quote_pair(raw[raw.hour==8],e.value/1e6,when) for e,weight in legs]
   if any(q is None for q in qs):attempts.append(dict(currency=c,date=date,tenor=tenor,status='same_strike_call_put_unavailable'));continue
   # Roll uses same contracts' past trades in the current UTC day.
   for q in qs:
    spreads=[]
    for typ in ['call','put']:
     g=raw[(raw.strike==q['strike'])&(raw.expiration_timestamp==legs[qs.index(q)][0].value/1e6)&(raw.option_type==typ)].sort_values('timestamp');x=g.price.diff().dropna().to_numpy();spreads.append(np.sqrt(max(-np.cov(x[:-1],x[1:])[0,1],0))*q['F'] if len(x)>=10 else np.nan)
    q['halfspread']=float(np.sum(spreads));q['spread_known']=np.isfinite(spreads).all()
   for freq in FREQUENCIES:
    cc=[];status='estimated'
    for (expiry,weight),q in zip(legs,qs):
     if weight==0:continue
     result,state=cashbook(c,when,expiry,q,freq)
     if state!='estimated':status=state;break
     r,book,fundbook=result;cc.append((weight,r));book['tenor']=tenor;book['weight']=weight;books.append(book);fundbook['tenor']=tenor;fundbook['weight']=weight;fundbooks.append(fundbook)
    attempts.append(dict(currency=c,date=date,tenor=tenor,frequency=freq,status=status))
    if status!='estimated':continue
    value={key:sum(weight*r[key] for weight,r in cc) for key in ['premium','gamma','hedge','payoff','funding','gross','hedge_turnover','option_turnover','cost_turnover','halfspread','gamma_variance_bridge','bridge_residual']};last=max(r['last_trade'] for _,r in cc);signalvalue=signals.get(date,np.nan)
    for bps in [0,2,5,10]:
     for spread in ['none','Roll']:
      half=value['halfspread'] if spread=='Roll' else 0.
      if not np.isfinite(half):continue
      cost=value['cost_turnover']*bps/10000+half;seller=value['gross']-cost
      for side,sign in [('sell',1),('buy',-1)]:
       pnl=seller if side=='sell' else -seller-2*cost
       for selection in ['all','signal_window','positive_vrp']:
        if selection!='all' and (tenor!=1 or not np.isfinite(signalvalue)):continue
        if selection=='positive_vrp' and signalvalue<=0:continue
        results.append(dict(currency=c,date=date,entry=when,last_quote=last,max_expiry=max(e for e,_ in legs),tenor=tenor,frequency=freq,bps=bps,spread=spread,side=side,selection=selection,pnl=pnl,usd_10000=pnl*10000,premium=sign*value['premium']*10000,gamma=sign*value['gamma']*10000,funding=sign*value['funding']*10000,transaction=cost*10000,hedge_turnover=value['hedge_turnover'],option_turnover=value['option_turnover'],halfspread=half*10000,gamma_variance_bridge=sign*value['gamma_variance_bridge']*10000,bridge_residual=sign*value['bridge_residual']*10000,vrp_signal=signalvalue,breakeven_bps=(value['gross']-half)/value['cost_turnover']*10000))
  if (i+1)%100==0:log(f'辰 策略 {c} {i+1}观察日')
 p=pd.DataFrame(results);p.to_parquet(OUT/f'chen/strategy_path_{c}.parquet',index=False);csv(attempts,f'chen/strategy_availability_{c}.csv')
 pd.concat(books,ignore_index=True).to_parquet(OUT/f'chen/strategy_cashbook_{c}.parquet',index=False)
 pd.concat(fundbooks,ignore_index=True).to_parquet(OUT/f'chen/strategy_funding_{c}.parquet',index=False)
 complete.write_text(json.dumps(dict(seed=SEED,rows=len(p))));log('辰 '+c+'全部策略现金流完成')

def summarize():
 rows=[];worst=[];breaks=[];signals=[]
 for c in ['BTC','ETH']:
  p=pd.read_parquet(OUT/f'chen/strategy_path_{c}.parquet');keys=['tenor','frequency','bps','spread','side','selection']
  for key,g in p.groupby(keys):
   d=dict(zip(keys,key));ref=p[(p.tenor==key[0])&(p.frequency==key[1])&(p.bps==key[2])&(p.spread==key[3])&(p.side==key[4])&(p.selection==('all' if key[5]=='all' else 'signal_window'))]
   for split,z in [('all',g)]+[(str(y),z) for y,z in g.groupby(g.date.dt.year)]:
    max_days=float(((z.max_expiry-z.entry).dt.total_seconds()/86400).max());lag=max(7,2*int(key[0])+14,int(np.ceil(2*max_days)));start=ref.date.min() if split=='all' else max(ref.date.min(),pd.Timestamp(split+'-01-01'));end=ref.date.max() if split=='all' else min(ref.date.max(),pd.Timestamp(split+'-12-31'));idx=pd.date_range(start,end);dailyusd=z.set_index('date').usd_10000.reindex(idx).fillna(0);mc=ci(dailyusd.to_numpy(),lag);trade=ci(z.usd_10000.to_numpy(),lag);row=dict(currency=c,split=split,**d,**mc,trade_n=len(z),hac_lag=lag,max_actual_days=max_days,trade_mean=trade['mean'],trade_low=trade['ci_low'],trade_high=trade['ci_high'],annual_sharpe=np.sqrt(365)*dailyusd.mean()/dailyusd.std() if dailyusd.std()>0 else np.nan,worst_day=z.usd_10000.min())
    for col in ['premium','gamma','funding','transaction','halfspread','bridge_residual']:row[col+'_mean']=z[col].sum()/len(idx) if len(idx) else np.nan
    rows.append(row)
   worst.append(g.nsmallest(10,'usd_10000'))
  zero=p[(p.side=='sell')&(p.bps==0)&(p.selection=='all')]
  for key,g in zero.groupby(['tenor','frequency','spread']):
   den=g.hedge_turnover+g.option_turnover;ratio=g.usd_10000.mean()/den.mean();influence=(g.usd_10000-ratio*den)/den.mean();co=ci(influence.to_numpy(),max(7,2*int(key[0])+14,int(np.ceil(2*((g.max_expiry-g.entry).dt.total_seconds()/86400).max()))));breaks.append(dict(currency=c,tenor=key[0],frequency=key[1],spread=key[2],bps=ratio,ci_low=ratio-1.96*co['se'],ci_high=ratio+1.96*co['se'],n=len(g)))
  v=pd.read_csv(OUT/f'chen/vrp_signal_comparison_{c}.csv');z=v.dropna(subset=['forecast','vrp_forecast']);signals.append(dict(currency=c,n_common=len(z),R3_positive=int((z.vrp_forecast>0).sum()),R4_positive=int((z.forecast>0).sum()),R3_mean=z.vrp_forecast.mean(),R4_mean=z.forecast.mean(),target_mean=z.signed_vrp.mean(),unit='total_variance',target_has_negative=True,log_target=False))
 csv(rows,'chen/strategy_summary.csv');csv(pd.concat(worst,ignore_index=True),'chen/worst_ten_days.csv');csv(breaks,'chen/breakeven_cost.csv');csv(signals,'chen/vrp_signal_audit_summary.csv');log('辰 策略全规格、逐年、最差日与盈亏平衡表完成')
if __name__=='__main__':
 from concurrent.futures import ProcessPoolExecutor
 with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(run,['BTC','ETH']))
 summarize()
