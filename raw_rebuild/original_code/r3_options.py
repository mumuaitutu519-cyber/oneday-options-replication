#!/usr/bin/env python3
from r3_common import *
import time
COLUMNS=['currency','date','hour_bucket','timestamp','instrument_name','expiration_timestamp','strike','option_type','F','index_price','price','actual_usd','iv','w','sigma','remaining_hours','log_moneyness','block','combo']
def archive_state(currency):
 p=ROOT/f'out/R2/download_{currency}.json'
 if not p.exists():return None
 m=json.loads(p.read_text());start=m.get('continuous_start');end=min(m.get('latest','2026-10-06'),'2026-10-06');m['latest']=end
 if not start:return None
 days=pd.date_range(start,end).strftime('%Y-%m-%d');missing=[d for d in days if not (ROOT/f'out/R2/clean/{currency}/{d}.audit.json').exists()]
 return dict(**m,clean_missing=len(missing),first_missing=missing[:3])
@lru_cache(maxsize=16)
def day(currency,date):
 dstr=pd.Timestamp(date).strftime('%Y-%m-%d');p=ROOT/f'out/R2/clean/{currency}/{dstr}.parquet';a=p.with_suffix('.audit.json')
 if not p.exists() or not a.exists():return pd.DataFrame(columns=COLUMNS)
 d=pd.read_parquet(p)
 if not len(d):return pd.DataFrame(columns=COLUMNS)
 return d

def select(d,band=.05,include=False):
 if not len(d):return d
 keep=d.log_moneyness.abs()<=band
 if not include:keep&=~d.block&~d.combo
 return d[keep]

def snapshot(currency):
 dest=OUT/f'shared/option_buckets_{currency}.parquet'
 if dest.exists():return pd.read_parquet(dest)
 state=archive_state(currency)
 if state is None or state['clean_missing']:raise RuntimeError('Shared archive not ready '+str(state))
 allb=[];manifest=[]
 for i,dt in enumerate(pd.date_range(state['continuous_start'],state['latest'])):
  d=day(currency,dt);p=ROOT/f'out/R2/clean/{currency}/{dt.strftime("%Y-%m-%d")}.parquet'
  if len(d):
   for band in [.02,.05,.10]:
    for inc in [False,True]:
     g=select(d,band,inc);g=g[g.hour_bucket.dt.hour==8];b=g.groupby(['hour_bucket','expiration_timestamp']).agg(timestamp=('timestamp','median'),last_timestamp=('timestamp','max'),w=('w','median'),sigma=('sigma','median'),n=('w','size'),F=('F','median'),strike=('strike','median'),n_call=('option_type',lambda x:(x=='call').sum()),n_put=('option_type',lambda x:(x=='put').sum())).reset_index();b['band']=band;b['include_block']=inc;allb.append(b)
  audit=json.loads(p.with_suffix('.audit.json').read_text());manifest.append(dict(currency=currency,date=dt,raw_sha256=audit['raw_sha256'],clean_sha256=sha(p),rows=len(d),path=str(p.relative_to(ROOT))))
  if (i+1)%250==0:log(f'共享期权 {currency} 扫描 {i+1}/{state["continuous_days"]}日')
 csv(manifest,f'shared/option_source_{currency}.csv');b=pd.concat(allb,ignore_index=True) if allb else pd.DataFrame();b['currency']=currency;b.to_parquet(dest,index=False);csv([state],f'shared/archive_{currency}.csv');return b

def observed(currency,when,expiry,band=.05,include=False,minn=3,width=30):
 when=pd.Timestamp(when);parts=[day(currency,date) for date in pd.date_range((when-pd.Timedelta(minutes=width)).floor('D'),(when+pd.Timedelta(minutes=width)).floor('D'))];d=pd.concat(parts,ignore_index=True);d=select(d,band,include)
 if not len(d):return None
 center=when.value/1e6;d=d[(d.timestamp>=center-width*60000)&(d.timestamp<=center)&(d.expiration_timestamp==pd.Timestamp(expiry).value/1e6)]
 if len(d)<minn:return None
 remaining=(pd.Timestamp(expiry)-when).total_seconds()/(365*86400)
 return dict(w=(d.sigma**2*remaining).median(),iv=d.sigma.median(),F=d.F.median(),n=len(d),last_timestamp=d.timestamp.max(),mean_timestamp=d.timestamp.median())

def pair(currency,when,expiry,band=.05):
 when=pd.Timestamp(when);parts=[day(currency,date) for date in pd.date_range((when-pd.Timedelta(minutes=30)).floor('D'),when.floor('D'))];d=pd.concat(parts,ignore_index=True);g=select(d,band);stamp=when.value/1e6
 if not len(g):return None
 g=g[(g.timestamp>=stamp-1800000)&(g.timestamp<=stamp)&(g.expiration_timestamp==pd.Timestamp(expiry).value/1e6)]
 valid=[]
 for k,z in g.groupby('strike'):
  if {'call','put'}<=set(z.option_type):valid.append((abs(np.log(k/z.F.median())),k,z))
 if not valid:return None
 _,k,z=min(valid,key=lambda x:x[0]);legs=[];spreads=[]
 whole=select(day(currency,when.floor('D')),.1,True)
 whole=whole[whole.timestamp<=stamp]
 for typ in ['call','put']:
  row=z[z.option_type==typ].sort_values('timestamp').iloc[-1];legs.append(row)
  x=whole[whole.instrument_name==row.instrument_name].sort_values('timestamp').price.diff().dropna().to_numpy()
  spread=np.sqrt(max(-np.cov(x[:-1],x[1:])[0,1],0))*row.index_price if len(x)>=10 else np.nan;spreads.append(spread)
 return dict(strike=k,premium=sum(r.actual_usd for r in legs),sigma=np.mean([r.sigma for r in legs]),F=np.mean([r.F for r in legs]),last_timestamp=max(r.timestamp for r in legs),first_timestamp=min(r.timestamp for r in legs),halfspread=sum(spreads),spread_known=all(np.isfinite(spreads)))

@lru_cache(maxsize=2)
def metadata(currency):
 return pd.read_parquet(ROOT/f'data_clean/option_instruments_{currency}.parquet')[['instrument_name','strike','expiration_timestamp','option_type']]

@lru_cache(maxsize=8)
def rawday(currency,date):
 path=ROOT/f'out/R2/raw/{currency}/{date}/trades.parquet'
 return pd.read_parquet(path)

def exit_pair(currency,when,expiry,strike):
 when=pd.Timestamp(when);stamp=when.value/1e6;whole=rawday(currency,when.strftime('%Y-%m-%d'));meta=metadata(currency);meta=meta[(meta.expiration_timestamp==pd.Timestamp(expiry).value/1e6)&(meta.strike==strike)];g=whole[whole.instrument_name.isin(meta.instrument_name)].copy()
 for col in ['block_trade_id','combo_id','combo_trade_id']:
  if col in g:g=g[g[col].isna()]
 g=g[(g.timestamp<=stamp)&(g.price>0)&(g.index_price>0)];whole=g.copy();g=g[g.timestamp>=stamp-1800000].merge(meta,on='instrument_name',validate='many_to_one');legs=[];spreads=[]
 for typ in ['call','put']:
  z=g[g.option_type==typ]
  if not len(z):return None
  row=z.sort_values('timestamp').iloc[-1];legs.append(row);changes=whole[whole.instrument_name==row.instrument_name].sort_values('timestamp').price.diff().dropna().to_numpy();spreads.append(np.sqrt(max(-np.cov(changes[:-1],changes[1:])[0,1],0))*row.index_price if len(changes)>=10 else np.nan)
 return dict(strike=strike,premium=sum(r.price*r.index_price for r in legs),sigma=np.nan,F=np.mean([r.index_price for r in legs]),last_timestamp=max(r.timestamp for r in legs),first_timestamp=min(r.timestamp for r in legs),halfspread=sum(spreads),spread_known=all(np.isfinite(spreads)))

@lru_cache(maxsize=2)
def funding_events(currency):
 d=pd.read_parquet(ROOT/f'data_clean/funding_settled_{currency}.parquet');return d.sort_values('calc_time').set_index('calc_time')

@lru_cache(maxsize=2)
def delivery_prices(currency):
 path=OUT/f'shared/delivery_{currency}.parquet'
 if not path.exists():path=ROOT/f'out/R2/delivery_{currency}.parquet'
 return pd.read_parquet(path).set_index('expiration_timestamp').delivery_price.to_dict()

def hedge_trade(currency,when,expiry,quote,bps=5,exit_time=None,exit_quote=None):
 h=grid(currency);when=pd.Timestamp(when);expiry=pd.Timestamp(expiry);end=pd.Timestamp(exit_time) if exit_time is not None else expiry
 times=pd.date_range(when,end,freq='h');times=times.union(pd.DatetimeIndex([end]));S=h.close.reindex(times).to_numpy();valid=np.isfinite(S).all()
 if not valid or len(S)<2:return None
 K=quote['strike'];vol=quote['sigma'];tau=(expiry-times[:-1]).total_seconds().to_numpy()/(365*86400);w=np.maximum(vol**2*tau,EPS);delta=2*special.ndtr((np.log(S[:-1]/K)+w/2)/np.sqrt(w))-1
 delivery=delivery_prices(currency).get(int(expiry.value/1e6),np.nan)
 if exit_quote is None and not np.isfinite(delivery):return dict(status='delivery_price_unavailable')
 payoff=abs(delivery-K) if exit_quote is None else exit_quote['premium'];hedging=np.sum(delta*np.diff(S));turn=np.abs(delta[0])*S[0]+np.sum(np.abs(np.diff(delta))*S[1:-1])+np.abs(delta[-1])*S[-1]
 funds=funding_events(currency);funds=funds[(funds.index>when)&(funds.index<=end)];positions=np.searchsorted(times.to_numpy(),funds.index.to_numpy(),side='left')-1;marks=pd.to_numeric(funds.markPrice,errors='coerce').to_numpy();proxy=h.close.reindex(funds.index.floor('5min')).to_numpy();marks=np.where(np.isfinite(marks),marks,proxy)
 if not np.isfinite(marks).all():return dict(status='funding_notional_price_unavailable')
 fc=np.sum(delta[positions]*marks*funds.last_funding_rate.to_numpy())
 optcost=quote['halfspread']+(exit_quote['halfspread'] if exit_quote is not None else 0)
 if not np.isfinite(optcost):return dict(status='option_halfspread_unavailable')
 option_fee=(2*quote['F']+(2*exit_quote['F'] if exit_quote is not None else 0))*bps/10000;transaction=turn*bps/10000+optcost+option_fee;pnl=quote['premium']+hedging-payoff-fc-transaction
 return dict(status='estimated',pnl=pnl/S[0],usd_10000=pnl/S[0]*10000,premium=quote['premium'],option_halfspread=optcost,option_fee=option_fee/S[0],total_transaction_cost=transaction/S[0],turnover=turn/S[0],funding=fc/S[0],funding_settlements=len(funds),entry=when,exit=end,delivery_price=delivery if exit_quote is None else np.nan,strategy_contract='fixed_strike_delivery_actual_funding_full_cost')
if __name__=='__main__':
 for c in ['BTC','ETH']:snapshot(c)
