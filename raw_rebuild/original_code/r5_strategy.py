from r5_common import *
from r4_common import grid
from r3_options import funding_events,delivery_prices
from r5_measurement import prices
FREQ={'15min':15,'hourly':60,'4hour':240}

def engine(c):
 p=CHEN/f'strategy_path_{c}.parquet'
 if p.exists() and not FORCE and (CHEN/f'strategy_native_source_{c}.json').exists():
  existing=pd.read_parquet(p)
  if 'option_fee_cap' in existing:existing.drop(columns='option_fee_cap').to_parquet(p,index=False)
  return
 for old in (CHEN/f'strategy_parts_{c}').glob('*.parquet'):
  if not old.name.startswith('._'):old.unlink(missing_ok=True)
 entries=pd.read_csv(CHEN/f'strategy_entries_{c}.csv',parse_dates=['date','entry','expiry']);pred=pd.read_parquet(CHEN/f'predictions_{c}_w500_log_hour8_h1.parquet').set_index('date');data={'binance':grid(c).close,'deribit':prices(c,'deribit')};D=delivery_prices(c);bf=funding_events(c);df=pd.read_parquet(ROOT/f'out/R2/deribit_funding_{c}.parquet');df.index=pd.to_datetime(df.timestamp,unit='ms');df=df.sort_index();events=pd.to_datetime(pd.read_csv(ROOT/'out/R3/shared/events.csv').datetime_utc,utc=True).dt.tz_localize(None).sort_values().to_numpy();rows=[];status=[];books=[];part=0;folder=CHEN/f'strategy_parts_{c}';folder.mkdir(exist_ok=True)
 def flush():
  nonlocal rows,books,part
  if rows:pd.DataFrame(rows).to_parquet(folder/f'path_{part:04d}.parquet',index=False)
  if books:pd.concat(books,ignore_index=True).to_parquet(folder/f'book_{part:04d}.parquet',index=False)
  rows=[];books=[];part+=1
 for i,q in enumerate(entries.itertuples(index=False)):
  if q.expiry>=CUTOFF:continue
  delivery=D.get(int(q.expiry.value/1e6),np.nan)
  if not np.isfinite(delivery):status.append(dict(currency=c,date=q.date,entry_mode=q.entry_mode,status='official_delivery_missing'));continue
  for tool,s in data.items():
   for vol in ['entry_IV','HAR','IV_calibrated']:
    if vol=='entry_IV':sigma=q.sigma
    else:
     value=pred.at[q.date,vol] if q.date in pred.index else np.nan;sigma=np.sqrt(value/((q.expiry-q.entry).total_seconds()/(365*86400)))
    if not np.isfinite(sigma):status.append(dict(currency=c,date=q.date,entry_mode=q.entry_mode,tool=tool,vol=vol,status='preorigin_forecast_missing'));continue
    for freq,minutes in FREQ.items():
     times=pd.date_range(q.entry,q.expiry,freq=f'{minutes}min').union(pd.DatetimeIndex([q.expiry]));S=s.reindex(times).to_numpy();baseS=data['binance'].get(q.entry,np.nan)
     if not np.isfinite(S).all() or not np.isfinite(baseS):status.append(dict(currency=c,date=q.date,entry_mode=q.entry_mode,tool=tool,vol=vol,frequency=freq,status='price_boundary_missing'));continue
     tau=(q.expiry-times[:-1]).total_seconds().to_numpy()/(365*86400);w=np.maximum(sigma*sigma*tau,EPS);delta=2*special.ndtr((np.log(S[:-1]/q.strike)+w/2)/np.sqrt(w))-1
     if tool=='binance':
      hedge=float(delta@np.diff(S));turn=float(abs(delta[0])*S[0]+abs(np.diff(delta))@S[1:-1]+abs(delta[-1])*S[-1]);fund=bf[(bf.index>q.entry)&(bf.index<=q.expiry)];pos=np.searchsorted(times.to_numpy(),fund.index.to_numpy(),side='left')-1;marks=pd.to_numeric(fund.markPrice,errors='coerce').to_numpy();marks=np.where(np.isfinite(marks),marks,s.reindex(fund.index.floor('5min')).to_numpy());fcash=delta[pos]*marks*fund.last_funding_rate.to_numpy();contracts=delta*S[:-1]
     else:
      contracts=delta*S[:-1];hedge=float(np.sum(contracts*(1/S[:-1]-1/S[1:]))*delivery);turn=float(abs(contracts[0])+abs(np.diff(contracts)).sum()+abs(contracts[-1]));fund=df[(df.index>q.entry)&(df.index<=q.expiry)];pos=np.searchsorted(times.to_numpy(),fund.index.to_numpy(),side='left')-1;marks=fund.index_price.to_numpy();fcash=contracts[pos]/marks*fund.interest_1h.to_numpy()*delivery
     if len(fund)==0 or (tool=='deribit' and len(fund)!=23) or not np.isfinite(fcash).all():status.append(dict(currency=c,date=q.date,entry_mode=q.entry_mode,tool=tool,vol=vol,frequency=freq,status='funding_notional_missing'));continue
     funding=float(np.sum(fcash));payoff=abs(delivery-q.strike);gross=q.premium-payoff+hedge-funding;optionturn=2*q.F;variant=f'{q.entry_mode}_{tool}_{vol}_{freq}';macro=np.searchsorted(events,q.expiry.to_datetime64(),side='right')>np.searchsorted(events,q.entry.to_datetime64(),side='right');weekend=q.date.dayofweek>=5
     for spread in ['none','Roll']:
      half=q.halfspread if spread=='Roll' else 0
      if not np.isfinite(half):continue
      for bps in [0,2,5,10]:
       cost=(turn+optionturn)*bps/10000+half
       sides=['sell'] if q.entry_mode=='sell' else ['buy'] if q.entry_mode=='buy' else ['sell','buy']
       for side in sides:
        sign=1 if side=='sell' else -1;pnl=(sign*gross-cost)/baseS;rows.append(dict(currency=c,date=q.date,entry=q.entry,last_quote=q.last_trade,max_expiry=q.expiry,tenor=1,frequency=freq,bps=bps,spread=spread,side=side,selection='all',variant=variant,entry_mode=q.entry_mode,tool=tool,hedge_vol=vol,pnl=pnl,usd_10000=pnl*10000,premium=sign*q.premium/baseS*10000,gamma=sign*(payoff-hedge)/baseS*10000,funding=sign*funding/baseS*10000,transaction=cost/baseS*10000,hedge_turnover=turn/baseS,option_turnover=optionturn/baseS,halfspread=half/baseS*10000,gamma_variance_bridge=np.nan,bridge_residual=np.nan,vrp_signal=np.nan,breakeven_bps=(sign*gross-half)/(turn+optionturn)*10000,macro=bool(macro),weekend=weekend,sigma=sigma,delivery=delivery,spot0=baseS,strike=q.strike,funding_n=len(fund),official_fee_status='Binance_official_table_unavailable_grid_only'))
     books.append(pd.DataFrame(dict(currency=c,date=q.date,variant=variant,time=times[:-1],end=times[1:],S0=S[:-1],S1=S[1:],delta=delta,inverse_contract_notional=contracts,hedge_cash=delta*np.diff(S) if tool=='binance' else contracts*(1/S[:-1]-1/S[1:])*delivery,delivery=delivery)))
  if (i+1)%200==0:flush();log(f'R5 strategy {c}: {i+1}/{len(entries)} entries')
 flush();csv(status,f'strategy_availability_{c}.csv')
 import pyarrow.parquet as pq
 for prefix,destination in [('path',p),('book',CHEN/f'strategy_cashbook_{c}.parquet')]:
  writer=None
  for chunk in sorted(folder.glob(prefix+'_*.parquet')):
   table=pq.read_table(chunk)
   if writer is None:writer=pq.ParquetWriter(destination.with_suffix('.tmp.parquet'),table.schema,compression='snappy')
   writer.write_table(table)
  if writer:writer.close();destination.with_suffix('.tmp.parquet').replace(destination)
 (CHEN/f'strategy_native_source_{c}.json').write_text(json.dumps(dict(price_sha256=sha(CHEN/f'prices_deribit_{c}.parquet'),source='raw_returned_chart_bars')))
 log(f'R5 {c} strategy all variants computed')

def summarize():
 summaries=[];breaks=[];worst=[];paired=[];bootcache={};calendarcache={};events=pd.to_datetime(pd.read_csv(ROOT/'out/R3/shared/events.csv').datetime_utc,utc=True).dt.tz_localize(None).sort_values().to_numpy()
 for c in ['BTC','ETH']:
  p=pd.read_parquet(CHEN/f'strategy_path_{c}.parquet');availability=pd.read_csv(CHEN/f'strategy_availability_{c}.csv',parse_dates=['date']).fillna('');keys=['variant','tenor','frequency','bps','spread','side','selection'];reference=p[(p.entry_mode=='last')&(p.tool=='binance')&(p.hedge_vol=='entry_IV')&(p.frequency=='hourly')&(p.bps==0)&(p.spread=='none')&(p.side=='sell')].set_index('date')
  for key,g in p.groupby(keys,sort=True):
   info=dict(zip(keys,key));g=g.sort_values('date').set_index('date',drop=False);splits=[('all',g),('macro',g[g.macro]),('no_macro',g[~g.macro]),('weekend',g[g.weekend]),('weekday',g[~g.weekend])]+[(str(y),z) for y,z in g.groupby(g.date.dt.year)]
   calkey=(c,key[0],key[4])
   if calkey not in calendarcache:
    idxall=pd.date_range(g.date.min(),g.date.max());rr=g.iloc[0];bad=availability[availability.entry_mode.eq(rr.entry_mode)]
    for col,val in [('tool',rr.tool),('vol',rr.hedge_vol),('frequency',rr.frequency)]:
     if col in bad:bad=bad[bad[col].eq(val)|bad[col].eq('')]
    idxall=idxall.difference(pd.DatetimeIndex(bad.date));start=(idxall+pd.Timedelta(hours=9)).to_numpy();end=(idxall+pd.Timedelta(days=1,hours=8)).to_numpy();ismacro=np.searchsorted(events,end,side='right')>np.searchsorted(events,start,side='right');calendarcache[calkey]=(idxall,ismacro)
   idxall,ismacro=calendarcache[calkey]
   for split,z in splits:
    if split=='all':idx=idxall
    elif split=='macro':idx=idxall[ismacro]
    elif split=='no_macro':idx=idxall[~ismacro]
    elif split=='weekend':idx=idxall[idxall.dayofweek>=5]
    elif split=='weekday':idx=idxall[idxall.dayofweek<5]
    else:idx=idxall[idxall.year==int(split)]
    if len(idx)<10:continue
    # Idle calendar days are zero; unavailable market/funding days remain outside eligible sample.
    usd=z.set_index('date').usd_10000.reindex(idx).fillna(0);trade=ci(z.usd_10000);co=ci(usd);cachekey=(c,key[0],key[4],key[5],split)
    turn=(z.set_index('date').hedge_turnover+z.set_index('date').option_turnover).reindex(idx).fillna(0)
    if cachekey not in bootcache:bootcache[cachekey]=(bootstrap_distribution(usd+key[3]*turn),bootstrap_distribution(turn))
    bb,bt=bootcache[cachekey];lo,hi=np.nanquantile(bb-key[3]*bt,[.025,.975]);bs=dict(boot_low=lo,boot_high=hi,bootstrap_n=len(usd));cs=usd.cumsum();draw=cs-cs.cummax().clip(lower=0);row=dict(currency=c,split=split,**info,**co,trade_n=len(z),hac_lag=7,max_actual_days=23/24,trade_mean=trade['mean'],trade_low=trade['ci_low'],trade_high=trade['ci_high'],annual_sharpe=np.sqrt(365)*usd.mean()/usd.std() if usd.std()>0 else np.nan,worst_day=z.usd_10000.min(),skew=stats.skew(usd,bias=False),q01=usd.quantile(.01),q05=usd.quantile(.05),max_drawdown_usd=float(-draw.min()),**bs)
    for col in ['premium','gamma','funding','transaction','halfspread','bridge_residual']:row[col+'_mean']=z[col].sum(min_count=1)/len(idx)
    summaries.append(row)
   worst.append(g.nsmallest(10,'usd_10000').assign(worst_rank=np.arange(1,min(10,len(g))+1)))
   if key[3]==0:
    turn=g.hedge_turnover+g.option_turnover;gross=g.usd_10000;ratio=gross.mean()/turn.mean();influence=(gross-ratio*turn)/turn.mean();co=ci(influence);bs=bootstrap(influence)
    breaks.append(dict(currency=c,tenor=1,frequency=key[2],spread=key[4],side=key[5],variant=key[0],bps=ratio,ci_low=ratio-1.96*co['se'],ci_high=ratio+1.96*co['se'],n=len(g),boot_low=ratio+bs['boot_low'],boot_high=ratio+bs['boot_high']))
   if key[3]==0 and key[4]=='none' and key[5]=='sell':
    z=g.set_index('date').join(reference[['usd_10000']],rsuffix='_base',how='inner');co=ci(z.usd_10000-z.usd_10000_base);paired.append(dict(currency=c,variant=key[0],base='last_binance_entry_IV_hourly',n_common=len(z),mean_variant=z.usd_10000.mean(),mean_base=z.usd_10000_base.mean(),difference=co['mean'],difference_p=co['p']))
  # Exact replication on dates with the same contracts, entry and accounting tool.
  old=pd.read_parquet(R4/f'strategy_path_{c}.parquet');old=old[(old.tenor==1)&(old.frequency=='hourly')&(old.bps==0)&(old.spread=='none')&(old.side=='sell')&(old.selection=='all')];z=reference.reset_index().merge(old[['date','usd_10000','premium','gamma','funding']],on='date',suffixes=('','_R4'))
  csv(z,f'strategy_R4_paired_{c}.csv')
 csv(summaries,'strategy_summary.csv');csv(breaks,'breakeven_cost.csv');csv(pd.concat(worst,ignore_index=True),'worst_ten_days.csv');csv(paired,'strategy_variant_paired.csv');log('R5 strategy stationary CIs and risk summaries completed')
if __name__=='__main__':
 for c in ['BTC','ETH']:engine(c)
 summarize()
