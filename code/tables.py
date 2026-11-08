from common import *
import argparse,re
from functools import lru_cache
from decimal import Decimal,ROUND_HALF_UP
from concurrent.futures import ProcessPoolExecutor
BLOCKS=json.loads((ROOT/'verify/manuscript_tables.json').read_text());CELLS={};DETAIL=[]
LABEL={'HAR-S':'HAR_S','HAR-J':'HAR_J','HAR-Q':'HAR_Q','HAR-C':'HAR_clock','HAR-IV30':'HAR_DVOL','IV30':'DVOL_raw','IV':'IV_raw','IV-C':'IV_calibrated','IV+HAR':'IV_HAR'}
@lru_cache(maxsize=4)
def premium(c):return pd.read_parquet(DATA/f'premium_{c}.parquet')
@lru_cache(maxsize=8)
def pred(c,spec):
 if os.environ.get('REPLICATION_MODE','quick')=='full':p=pd.read_parquet(WORK/f'predictions_{c}_{spec}.parquet')
 else:p=pd.read_parquet(DATA/'forecast_cache'/f'{c}_{spec}.parquet')
 return p
@lru_cache(maxsize=2)
def basepred(c):return pred(c,'w500_log_hour8_h1').dropna(subset=['y']+CORE)
def put(b,r,c,value,source='',p=None):
 CELLS[(b,r,c)]=dict(value=value,p=p,source=source)
def stat(b,r,c,co,scale=1,source='',with_t=True):
 put(b,r,c,co['mean']*scale,source,p=co['p'])
 if with_t:put(b,r+1,c,co['t'],source)
def gain(b,r,c,y,a,z,h=1,source=''):
 co=compare(y,a,z,h);put(b,r,c,co['gain'],source,p=co['p']);put(b,r+1,c,co['t'],source)
def common_variant(c,kind):
 p=pred(c,kind+'_w500_log_hour8_h1')[['date','y','IV_raw','IV_calibrated','IV_HAR']];return p.merge(pred(c,'w500_log_hour8_h1')[['date']+HIST+['DVOL_raw']],on='date',validate='one_to_one').dropna()
def mcs_job(job):
 c,conf=job;d=basepred(c);L=np.column_stack([qlike(d.y,d[m]) for m in CORE]);return c,conf,mcs(L,1,alpha=1-conf,reps=1999)[0]
def tables_forecasts():
 with ProcessPoolExecutor(max_workers=int(os.environ.get('REPLICATION_WORKERS','4'))) as pool:
  mcsets={(c,conf):active for c,conf,active in pool.map(mcs_job,[(c,v) for c in ['BTC','ETH'] for v in [.9,.75]])}
 for c,b in [('BTC',3),('ETH',4)]:
  d=basepred(c);best=min(HIST,key=lambda m:qlike(d.y,d[m]).mean());L=np.column_stack([qlike(d.y,d[m]) for m in CORE]);sets={conf:mcsets[(c,conf)] for conf in [.9,.75]}
  for i,m in enumerate(CORE):
   r=1+2*i;src=f'out/R5/chen/forecast_comparisons.csv currency={c},spec=w500_log_hour8_h1,split=all,model={m}'
   put(b,r,2,qlike(d.y,d[m]).mean(),src)
   if m!='HAR':gain(b,r,3,d.y,d.HAR,d[m],source=src+',base=HAR')
   if m!=best:gain(b,r,4,d.y,d[best],d[m],source=src+',base='+best)
   for col,conf in [(5,.9),(6,.75)]:put(b,r,col,'Yes' if i in sets[conf] else 'No','out/R5/chen/model_confidence_set.csv '+src)
  # Table 3 robustness rows, against each row's actual complete-case sample.
  for i,row in enumerate(BLOCKS[4]['rows']):
   lab=row[0]
   if not lab or lab in ['']:continue
   if lab=='Main':z=d
   elif lab in ['Window 365','Expanding']:z=pred(c,('w365' if lab=='Window 365' else 'wexpanding')+'_log_hour8_h1').dropna(subset=['y']+CORE)
   elif lab=='Window 1,000':z=pred(c,'w1000_log_hour8_h1').dropna(subset=['y','HAR','HAR_S','HAR_J','HAR_Q','HAR_clock','HAR_DVOL','GB','DVOL_raw','DVOL_calibrated','IV_raw','IV_calibrated','IV_HAR','IV_GB'])
   elif lab=='Same days':old=pred(c,'w1000_log_hour8_h1').dropna(subset=['y','HAR','HAR_S','HAR_J','HAR_Q','HAR_clock','HAR_DVOL','GB','DVOL_raw','DVOL_calibrated','IV_raw','IV_calibrated','IV_HAR','IV_GB']);z=d[d.date.isin(old.date)]
   elif lab=='Trimmed':z=d[d.y<=d.y.quantile(.99)]
   elif lab=='2021-2022':z=d[d.date.dt.year.between(2021,2022)]
   elif lab=='2023':z=d[d.date.dt.year==2023]
   elif lab=='2024-2026':z=d[d.date.dt.year.between(2024,2026)]
   elif lab=='Before ETF':z=d[d.date<'2024-01-11']
   elif lab=='After ETF':z=d[d.date>='2024-01-11']
   elif lab in ['Mark price','Two-sided price','Last 15 minutes','Band 2%','Band 10%','Volume-weighted','Calm window','Previous-day IV']:
    v={'Mark price':'mark','Two-sided price':'bilateral','Last 15 minutes':'last15','Band 2%':'band02','Band 10%':'band10','Volume-weighted':'weighted','Calm window':'dropmove1','Previous-day IV':'placebo_lag1'}[lab];z=common_variant(c,v)
   elif lab in ['Spot RV','Deribit RV','One-minute RV','Subsampled RV','Realized kernel','Settlement RV']:
    v={'Spot RV':'binance_spot_rv5','Deribit RV':'deribit_rv5','One-minute RV':'binance_perp_rv1','Subsampled RV':'binance_perp_subsample','Realized kernel':'binance_perp_kernel','Settlement RV':'binance_perp_payoff'}[lab];z=pred(c,'rv_'+v).dropna(subset=['y']+HIST+['DVOL_raw','IV_calibrated','IV_HAR'])
   else:continue
   for j,(a,m) in enumerate([('HAR','IV_calibrated'),('DVOL_raw','IV_calibrated'),('GB','IV_HAR')]):gain(5,i,2+j+(0 if c=='BTC' else 3),z.y,z[a],z[m],source=f'out/R5/chen/forecast_comparisons.csv / variant_forecasts.csv / measurement_forecasts.csv {c},row={lab},base={a},model={m}')
  for j,m in enumerate(['IV_raw','IV_HAR']):
   X=np.log(d[[best,m]]);X.columns=['history','IV'];fit=sm.OLS(np.log(d.y),sm.add_constant(X)).fit(cov_type='HAC',cov_kwds={'maxlags':7})
   for r,v in [(3,'history'),(5,'IV'),(7,'const')]:stat(6,r,2+j+(0 if c=='BTC' else 2),dict(mean=fit.params[v],t=fit.params[v]/fit.bse[v],p=fit.pvalues[v]),source=f'out/R5/chen/encompassing.csv {c},model={m},form=log,variable={v}')
  for hidx,h in enumerate([1,2,3,7,14,30]):
   frames={k:pred(c,f'matrix_h{h}_k{k}_w500_log') for k in [1,2,3,7,14,30,'DVOL']};z=next(iter(frames.values()))[['date','y']].copy()
   for k,p in frames.items():z=z.merge(p[['date','HAR','IV_calibrated','IV_HAR']].rename(columns={m:f'{m}_{k}' for m in ['HAR','IV_calibrated','IV_HAR']}),on='date',validate='one_to_one')
   z=z.dropna();bb=7 if c=='BTC' else 8
   for j,k in enumerate(frames):
    co=compare(z.y,z[f'HAR_{k}'],z[f'IV_calibrated_{k}'],h);put(bb,2+hidx,2+j,co['gain'],f'out/R5/chen/tenor_matrix.csv {c},h={h},k={k},model=IV_calibrated',p=co['p'])
   put(bb,2+hidx,9,len(z),'out/R5/chen/tenor_matrix.csv common7')
 print('Forecast tables recalculated',flush=True)

def tables_premia():
 for c,offset in [('BTC',0),('ETH',4)]:
  d=premium(c);a=d[(d.variant=='trade_ATM')&(d.tenor==1)].dropna(subset=['vrp']);vals={'IV':np.sqrt(a.iv_annual)*100,'RV':np.sqrt(a.rv_annual.clip(lower=0))*100,'VP':a.volpts}
  for j,(v,x) in enumerate(vals.items()):
   values=[x.count(),x.mean(),x.std(),x.min(),x.quantile(.25),x.quantile(.5),x.quantile(.75),x.max()]
   for col,val in enumerate(values,2):put(2,2+j+offset,col,val,f'out/R5/chen/premium_panel_{c}.parquet variant=trade_ATM,tenor=1 variable={v}; preserved R4 term_panel')
  for variant,bb,start in [('trade_ATM',9,2 if c=='BTC' else 16),('mark',10 if c=='BTC' else 11,2),('model_free_grid_deducted',12,2 if c=='BTC' else 16)]:
   for i,h in enumerate([1,2,3,7,14,30]):
    z=d[(d.variant==variant)&(d.tenor==h)].sort_values('date').dropna(subset=['atm_w','rv_total']);z=z[z.rv_total>0];lag=max(7,2*h);r=start+2*i;src=f'out/R5/chen/term_premium.csv {c},variant={variant},tenor={h},split=all'
    stat(bb,r,2,ci(z.volpts,lag),source=src+',metric=volpts');put(bb,r,3,z.volpts.median(),src);put(bb,r,4,(z.volpts.dropna()>0).mean()*100,src);stat(bb,r,5,ci(z.log_ratio,lag),source=src+',metric=log_ratio');put(bb,r,6,z.log_ratio.count(),src)
  for i,(v,method) in enumerate([('trade_ATM','mean'),('trade_ATM','trimRV1'),('trade_ATM','trimRV5'),('trade_ATM','winsor1'),('mark','mean'),('mark','trimRV5'),('model_free_grid_deducted','mean'),('model_free_grid_deducted','trimRV1')]):
   z=d[(d.variant==v)&(d.tenor==1)].sort_values('date').dropna(subset=['atm_w','rv_total']);z=z[z.rv_total>0];x=z.vrp
   if method=='trimRV1':x=x[z.rv_total<=z.rv_total.quantile(.99)]
   elif method=='trimRV5':x=x[z.rv_total<=z.rv_total.quantile(.95)]
   elif method=='winsor1':x=x.clip(x.quantile(.01),x.quantile(.99))
   cc=2 if c=='BTC' else 6;co=ci(x);stat(13,2+2*i,cc,co,1e4,f'out/R5/chen/tail_statistics.csv {c},variant={v},method={method}');put(13,2+2*i,cc+2,co['n'],'out/R5/chen/tail_statistics.csv')
  # Side premia require both sides on the same interpolated days.
  z=d[(d.variant=='trade_ATM')&(d.tenor==1)].copy();z['buy_vrp']=z.buy_w-z.rv_total;z['sell_vrp']=z.sell_w-z.rv_total;z['side_gap']=z.buy_w-z.sell_w;z=z.dropna(subset=['buy_vrp','sell_vrp','rv_total'])
  for i,m in enumerate(['buy_vrp','sell_vrp','side_gap']):co=ci(z[m]);cc=2 if c=='BTC' else 4;stat(22,2+2*i,cc,co,1e4,f'out/R4/chen/price_direction_bounds.csv {c},tenor=1,metric={m}');put(22,2+2*i,cc+1,len(z),'out/R4/chen/price_direction_bounds.csv')
 print('Daily premia tables recalculated',flush=True)

def decomposition():
 allrows=[]
 for c in ['BTC','ETH']:
  d=pd.read_parquet(DATA/f'decomposition_{c}.parquet');diag=pd.read_parquet(DATA/f'numerical_grid_diagnostics_{c}.parquet');diag['date']=pd.to_datetime(diag.date);d=d.merge(diag[['date','tenor','corrected_mf_w']],on=['date','tenor'],how='inner',validate='many_to_one');d['total_I']=d.corrected_mf_w
  for variant,bb in [('mark_R5total',14),('trade_R5total',15)]:
   for i,h in enumerate([1,2,3,7,14,30]):
    z=d[(d.variant==variant)&(d.tenor==h)].dropna(subset=['total_I','atm_I','rv','bv']);rr=(2 if c=='BTC' else 16)+2*i
    z=z.copy();z['center']=z.atm_I-z.bv;z['tail']=z.total_I-z.atm_I-(z.rv-z.bv).clip(lower=0);z['total']=z.total_I-z.rv
    for j,m in enumerate(['center','tail','total']):stat(bb,rr,2+j,ci(z[m],max(7,2*h)),1e4,f'out/R9/claude_grid_corrected_decomposition.csv cur={c},var={variant},cont=bv,tenor={h},group=all,comp={m}')
    put(bb,rr,5,len(z),'out/R9/claude_grid_corrected_decomposition.csv')
  base=d[(d.variant=='mark_R5total')&(d.tenor==1)].dropna(subset=['total_I','atm_I','rv','bv']);groups=[('all',base,'bv'),('nonjump',base[~base.jump1.astype(bool)],'bv'),('jump',base[base.jump1.astype(bool)],'bv'),('macro',base[base.macro],'bv'),('weekend',base[base.weekend],'bv'),('rest',base[base.rest],'bv'),('all',base,'trv')]
  for i,(group,z,est) in enumerate(groups):
   z=z.copy();z['center']=z.atm_I-z[est];z['tail']=z.total_I-z.atm_I-(z.rv-z[est]).clip(lower=0);z['total']=z.total_I-z.rv
   if c=='BTC':bb,rr=(16,2) if i==0 else (17,(i-1)*2)
   else:bb,rr=17,14+2*i
   for j,m in enumerate(['center','tail','total']):stat(bb,rr,2+j,ci(z[m],7),1e4,f'out/R9/claude_grid_corrected_decomposition.csv cur={c},var=mark_R5total,cont={est},tenor=1,group={group},comp={m}')
   put(bb,rr,5,len(z),'out/R9/claude_grid_corrected_decomposition.csv')
  allrows.append(d)
 pd.concat(allrows).to_parquet(OUT/'decomposition_daily.parquet',index=False,compression='zstd')
 print('Grid corrected decomposition recalculated',flush=True)

@lru_cache(maxsize=2)
def bootstrap_weights(n):
 rng=np.random.default_rng(SEED);idx=np.empty((4999,n),np.int32);idx[:,0]=rng.integers(n,size=4999)
 for j in range(1,n):idx[:,j]=np.where(rng.random(4999)<1/7,rng.integers(n,size=4999),(idx[:,j-1]+1)%n)
 return np.bincount((idx+np.arange(4999,dtype=np.int64)[:,None]*n).ravel(),minlength=4999*n).reshape(4999,n).astype(float)
def boot_influence(inf):
 s=inf.reindex(pd.date_range(inf.index.min(),inf.index.max()));x=s.to_numpy();ok=np.isfinite(x);w=bootstrap_weights(len(x));v=(w@np.where(ok,x,0))/(w@ok.astype(float));return np.quantile(v,[.025,.975])
def be_job(job):
 key,gross,turn,ratio=job;influence=(gross-ratio*turn)/turn.mean();lo,hi=boot_influence(influence);return key,dict(be=ratio,lo=ratio+lo,hi=ratio+hi)
def strategy(c):
 p=pd.read_parquet(DATA/f'straddle_daily_{c}.parquet');bad=pd.read_parquet(DATA/f'strategy_availability_{c}.parquet');bad['date']=pd.to_datetime(bad.date);bad=bad.fillna('');records={};bejobs=[]
 for (v,b,side),g in p.groupby(['variant','bps','side']):
  g=g.sort_values('date').set_index('date',drop=False);rr=g.iloc[0];idx=pd.date_range(g.date.min(),g.date.max());a=bad[bad.entry_mode==rr.entry_mode]
  for col,val in [('tool',rr.tool),('vol',rr.hedge_vol),('frequency',rr.frequency)]:a=a[a[col].eq(val)|a[col].eq('')]
  idx=idx.difference(pd.DatetimeIndex(a.date));usd=g.usd_10000.reindex(idx).fillna(0);co=ci(usd);co.update(premium=g.premium.sum()/len(idx),gamma=g.gamma.sum()/len(idx),funding=g.funding.sum()/len(idx),worst=g.usd_10000.min(),n_calendar=len(idx),trade_n=len(g));records[(v,int(b),side)]=co
  if b==0:
   turn=g.hedge_turnover+g.option_turnover;ratio=g.usd_10000.mean()/turn.mean();bejobs.append(((v,int(b),side),g.usd_10000,turn,ratio))
 with ProcessPoolExecutor(max_workers=int(os.environ.get('REPLICATION_WORKERS','4'))) as pool:
  for key,result in pool.map(be_job,bejobs):records[key].update(result)
 return p,records

def tables_trading():
 mk=pd.read_parquet(DATA/'atm_trade_daily_by_side.parquet')
 for c in ['BTC','ETH']:
  p,records=strategy(c);pd.DataFrame([dict(variant=k[0],bps=k[1],side=k[2],eligible_calendar_days=v['n_calendar'],traded_days=v['trade_n']) for k,v in records.items()]).to_csv(OUT/f'strategy_sample_counts_{c}.csv',index=False);cc=2 if c=='BTC' else 4
  for i,(v,b) in enumerate([('last',0),('last',2),('last',5),('last',10),('own',0),('own',2),('mark',0),('mark',2)]):
   for j,side in enumerate(['sell','buy']):v0=side if v=='own' else v;co=records[(v0+'_binance_entry_IV_hourly',b,side)];stat(18,2+2*i,cc+j,co,source=f'out/R5/chen/strategy_summary.csv {c},variant={v0}_binance_entry_IV_hourly,bps={b},side={side},split=all,spread=none')
  co=records[('last_binance_entry_IV_hourly',0,'sell')]
  for r,k in [(1,'premium'),(2,'gamma'),(3,'funding'),(4,'mean'),(5,'worst')]:put(19,r,2 if c=='BTC' else 3,co[k],'out/R5/chen/strategy_summary.csv last_binance_entry_IV_hourly,bps=0,side=sell')
  for i,(v,side) in enumerate([('last','sell'),('sell','sell'),('buy','buy'),('mark','sell')]):
   co=records[(v+'_binance_entry_IV_hourly',0,side)];put(20,2+i,cc,co['be'],f'out/R5/chen/breakeven_cost.csv {c},variant={v}_binance_entry_IV_hourly,side={side}');put(20,2+i,cc+1,[co['lo'],co['hi']],'out/R5/chen/breakeven_cost.csv 4999 stationary draws, seed=20261007')
  for i,side in enumerate(['all','buy','sell']):
   z=mk[(mk.currency==c)&(mk.side==side)].sort_values('date');co=ci(z.difference_mean);stat(21,2+2*i,cc,co,source=f'out/R4/chen/trade_mark_comparison.csv currency={c},split={side}');put(21,2+2*i,cc+1,z.trade_n.sum(),'out/R4/chen/trade_mark_comparison.csv')
  quality=pd.read_parquet(DATA/f'trade_quality_{c}.parquet');quality['date']=pd.to_datetime(quality.date);d=premium(c);v=d[(d.tenor==1)&d.variant.isin(['trade_ATM','mark'])].pivot(index='date',columns='variant',values='volpts');a=quality.merge(v,left_on='date',right_index=True,how='left');s=p[(p.variant=='last_binance_entry_IV_hourly')&(p.bps==0)&(p.side=='sell')].set_index('date');a['seller']=a.date.map(s.usd_10000)
  for i,y in enumerate(range(2020,2027)):
   z=a[a.date.dt.year==y];g=s[s.index.year==y];idle=pd.date_range(s.index.min(),s.index.max());bad=pd.read_parquet(DATA/f'strategy_availability_{c}.parquet').fillna('');bad['date']=pd.to_datetime(bad.date);bb=bad[bad.entry_mode=='last']
   for col,val in [('tool','binance'),('vol','entry_IV'),('frequency','hourly')]:bb=bb[bb[col].eq(val)|bb[col].eq('')]
   idle=idle.difference(pd.DatetimeIndex(bb.date));idle=idle[idle.year==y];seller=g.usd_10000.reindex(idle).fillna(0).mean();rr=(2 if c=='BTC' else 11)+i
   for j,val in enumerate([z.sell_n.sum()/z.trade_n.sum()*100,z.trade_mark_mean.mean(),z.trade_ATM.mean(),z.mark.mean(),seller],2):put(23,rr,j,val,f'out/R5/chen/annual_overview.csv currency={c},year={y}')
 print('Trading and order flow tables recalculated',flush=True)

def tables_risk():
 from risk import coverage_test
 for c in ['BTC','ETH']:
  if os.environ.get('REPLICATION_MODE','quick')=='full':p=pd.read_csv(WORK/f'risk_quantiles_{c}_w500_log_hour8_h1.csv')
  else:p=pd.read_parquet(DATA/f'risk_inputs_{c}.parquet')
  p['date']=pd.to_datetime(p.date);p=p[p.date.isin(basepred(c).date)]
  models=['HAR','GB','DVOL_raw','IV_calibrated','IV_HAR']
  for side,bb in [('long',24),('short',25)]:
   prefix=side+'_99';frames={m:p[p.model==m].set_index('date') for m in models};dates=sorted(set.intersection(*[set(x.index[x[prefix+'_VaR'].notna()]) for x in frames.values()]))
   for i,(m,z) in enumerate(frames.items()):
    z=z.loc[dates].sort_index();loss=(-z.r if side=='long' else z.r).to_numpy();q=z[prefix+'_VaR'].to_numpy();es=z[prefix+'_ES'].to_numpy();ex=loss>q;co=coverage_test(ex,z.index,.99);ec=ci(ex*(loss-es));rat=loss/q;K=int(round(len(q)*.01));a=np.sort(rat);mult=(a[-K-1]+a[-K])/2;qq=q*mult;margin=1-np.exp(-qq) if side=='long' else np.exp(qq)-1;rr=(2 if c=='BTC' else 9)+i
    for j,val in enumerate([co['exceedance']*100,co['uc_p'],co['cc_p'],ec['p'],margin.mean()*100],2):put(bb,rr,j,val,f'out/R5/chen/risk_backtests.csv / risk_capital_efficiency.csv {c},model={m},side={side},alpha=0.99')
 print('VaR and ES tables recalculated',flush=True)

def serializable(v):return v.item() if isinstance(v,np.generic) else v

def main():
 for fn in [tables_forecasts,tables_premia,decomposition,tables_trading,tables_risk]:
  fn()
 rows=[]
 for (b,r,c),v in CELLS.items():rows.append(dict(block=b,row=r,column=c,**{k:serializable(x) for k,x in v.items()}))
 (OUT/'recalculated_cells.json').write_text(json.dumps(rows,indent=2,allow_nan=True));print('Computed cells:',len(rows),flush=True)
if __name__=='__main__':main()
