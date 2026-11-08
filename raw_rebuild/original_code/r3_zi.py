#!/usr/bin/env python3
from r3_common import *
import itertools
from scipy.special import ndtr

def features(asset,measure,form,iv):
 d=daily(asset,measure);idx=d.index;y=d[measure];x1=har(y,'own',form)
 x2=x1.copy()
 for c in ['BTC','ETH']:
  r=daily(c,measure)[measure].reindex(idx);x2=x2.join(har(r,c,form))
 x3=x2.copy();x4=x3.copy();xr=x2.copy()
 for c in ['BTC','ETH']:
  v=iv[c].reindex(idx)**2/365;rv=daily(c,measure)[measure].reindex(idx).rolling(30).mean()
  x3[c+'_iv']=np.log(v.clip(lower=EPS)) if form=='log' else v
  x4[c+'_iv']=x3[c+'_iv'];x4[c+'_vrp']=v-rv;xr[c+'_vrp']=v-rv
 return d,y,{'M1':x1,'M2':x2,'M3':x3,'M4':x4,'M2_VRP':xr}

def hedge(asset,path):
 h=hourly(asset);price=h.close;fund=pd.read_parquet(ROOT/f'data_clean/funding_settled_{asset}.parquet');stamp=pd.to_datetime(fund.calc_time);marks=pd.to_numeric(fund.markPrice,errors='coerce').to_numpy();proxy=grid(asset).close.reindex(stamp.dt.floor('5min')).to_numpy();marks=np.where(np.isfinite(marks),marks,proxy)
 # Exact boundary settlements use the position held before the boundary; delayed settlements use the updated position.
 cash=pd.Series(fund.last_funding_rate.to_numpy()*marks,index=stamp.dt.ceil('h')).groupby(level=0).agg(lambda x:x.sum(min_count=len(x)));counts=stamp.dt.ceil('h').value_counts();fundcash=pd.Series(0.,index=price.index);idx=fundcash.index.intersection(cash.index);fundcash.loc[idx]=cash.reindex(idx);fundcounts=counts.reindex(price.index).fillna(0);csv([dict(asset=asset,settlements=len(fund),nonexact_hour_settlements=int((stamp!=stamp.dt.floor('h')).sum()),missing_notional=int((~np.isfinite(marks)).sum()),contract='actual_settlement_timestamp_and_notional')],f'zi/funding_audit_{asset}.csv')
 results=[];detail=[]
 for horizon in [7,30]:
  g=path[(path.horizon==horizon)&(path.spec=='rv5_log_w1000_lag0')].copy();length=horizon*24
  origins=pd.DatetimeIndex(g.date)+pd.Timedelta(days=1);dates=origins.to_numpy('datetime64[ns]');positions=price.index.get_indexer(origins)
  valid=(positions>=0)&(positions+length<len(price));g=g.loc[valid].copy();positions=positions[valid];origins=origins[valid]
  if not len(g):continue
  matrix=price.to_numpy()[positions[:,None]+np.arange(length+1)];f=fundcash.to_numpy()[positions[:,None]+np.arange(1,length+1)];valid=np.isfinite(matrix).all(1)&np.isfinite(f).all(1);g=g.loc[valid];matrix=matrix[valid];f=f[valid];positions=positions[valid];origins=origins[valid];fcounts=fundcounts.to_numpy()[positions[:,None]+np.arange(1,length+1)].sum(1)
  K=matrix[:,0];returns=matrix[:,1:]-matrix[:,:-1]
  for model in [c for c in g.columns if c.startswith('M')]:
   vv=g[model].to_numpy(float);valid=np.isfinite(vv)&(vv>0)
   if not valid.any():continue
   s=matrix[valid];k=K[valid];v=vv[valid];w=v*horizon
   delta=2*ndtr((np.log(s[:,:-1]/k[:,None])+.5*v[:,None]*np.arange(length,0,-1)/24)/np.sqrt(v[:,None]*np.arange(length,0,-1)/24))-1
   premium=k*(4*ndtr(np.sqrt(w)/2)-2);hedging=(delta*(s[:,1:]-s[:,:-1])).sum(1);payoff=np.abs(s[:,-1]-k);fundcost=(delta*f[valid]).sum(1)
   turnover=(np.abs(delta[:,0])*k+np.sum(np.abs(np.diff(delta,axis=1))*s[:,1:-1],axis=1)+np.abs(delta[:,-1])*s[:,-1])/k
   gross=(premium+hedging-payoff-fundcost)/k;annvol=np.sqrt(v*365)
   for bps in [0,2,5,10]:
    pnl=gross-turnover*bps/10000;mc=mean_ci(pnl,2*horizon)
    # Premium bump: keep the registered hedge path, solve the required premium.
    cost=(payoff-hedging+fundcost)/k+turnover*bps/10000
    target=np.maximum(cost,0);root=2*stats.norm.ppf(np.clip((target+2)/4,.50000001,.99999999))/np.sqrt(horizon/365)
    bump=(root-annvol)*100;bc=mean_ci(bump,2*horizon)
    results.append(dict(asset=asset,horizon=horizon,model=model,cost_bps=bps,n=len(pnl),mean_pnl_unit=mc['mean'],ci_low_unit=mc['ci_low'],ci_high_unit=mc['ci_high'],mean_usd_10000=mc['mean']*10000,ci_low_usd=mc['ci_low']*10000,ci_high_usd=mc['ci_high']*10000,sd_usd=np.std(pnl,ddof=1)*10000,q01_usd=np.quantile(pnl,.01)*10000,quote_bump_volpts=bc['mean'],quote_bump_low=bc['ci_low'],quote_bump_high=bc['ci_high'],negative_hedge_cost=int((cost<0).sum()),funding_contract='actual_settlement_timestamp_and_notional'))
    if bps==5:detail.append(pd.DataFrame(dict(asset=asset,date=origins[valid],horizon=horizon,model=model,pnl=pnl,quote_bump_volpts=bump,vol=annvol,turnover=turnover,funding_unit=fundcost/k,funding_settlements=fcounts[valid],funding_contract='actual_settlement_timestamp_and_notional')))
 csv(results,f'zi/hedge_{asset}.csv')
 if detail:pd.concat(detail).to_parquet(OUT/f'zi/hedge_path_{asset}.parquet',index=False)
 log(asset+' 跨式对冲完成')

def run_asset(asset,iv):
 allpaths=[];comparisons=[];inside=[];gates=[];ratios=[]
 specs=list(itertools.product(['rv5','rv_sub','rv_robust'],['log','level'],[750,1000,1500]))+[('rv5','log',1000)]
 for specnum,(measure,form,window) in enumerate(specs):
  lag=int(specnum==18);spec=f'{measure}_{form}_w{window}_lag{lag}';d,y,xs=features(asset,measure,form,iv)
  if lag:xs={m:x.shift(lag) for m,x in xs.items()}
  for horizon in [1,7,30]:
   target=forward_mean(y,horizon);p=pd.DataFrame(dict(date=d.index,asset=asset,horizon=horizon,spec=spec,target=target.to_numpy()))
   maxlabels=[]
   for model,x in xs.items():
    pr,last=rolling(x,target,horizon,window,form,start=1000);p[model]=pr.to_numpy();maxlabels.append(last)
    if spec=='rv5_log_w1000_lag0':
     z=pd.concat([x,target.rename('target')],axis=1).dropna();yy=np.log(z.target) if form=='log' else z.target
     xm=sm.add_constant(z[x.columns],has_constant='add');fit=sm.OLS(yy,xm).fit(cov_type='HAC',cov_kwds={'maxlags':2*horizon})
     for name in fit.params.index:
      inside.append(dict(asset=asset,horizon=horizon,model=model,variable=name,n=len(z),r2=fit.rsquared,coef=fit.params[name],se=fit.bse[name],p=fit.pvalues[name],ci_low=fit.conf_int().loc[name,0],ci_high=fit.conf_int().loc[name,1]))
   btc=daily('BTC',measure)[measure].reindex(d.index);vbtc=iv['BTC'].reindex(d.index)**2/365
   for rw in [30,90,180]:
    ratio=y.rolling(rw).mean()/btc.rolling(rw).mean();raw=vbtc*ratio
    matured=forward_mean(btc,30).shift(30)/vbtc.shift(30);correction=matured.rolling(rw).mean()
    p[f'M5_{rw}']=raw.shift(lag).to_numpy();p[f'M6_{rw}']=(raw*correction).shift(lag).to_numpy()
   models=[c for c in p if c.startswith('M')];common=np.isfinite(p[['target','M1','M2','M3','M4']].to_numpy()).all(1);p=p.loc[common].copy()
   if spec=='rv5_log_w1000_lag0':gates.append(dict(asset=asset,horizon=horizon,n_eval=len(p),min_required=700,passed=len(p)>=700))
   if len(p)<700 and spec=='rv5_log_w1000_lag0':log(f'{asset}/{horizon}数据门未过 {len(p)}');continue
   if len(p)<30:comparisons.append(dict(asset=asset,horizon=horizon,spec=spec,model='all',n=len(p),status='insufficient'));continue
   for model in models:
    if model=='M1':continue
    for subsample,gg in [('all',p)]+[(str(yr),s) for yr,s in p.groupby(p.date.dt.year)] if spec=='rv5_log_w1000_lag0' else [('all',p)]:
     r=improvement(gg.target,gg.M1,gg[model],horizon);r.update(asset=asset,horizon=horizon,spec=spec,model=model,subsample=subsample);comparisons.append(r)
   p['forecast_public_at']=p.date+pd.Timedelta(days=1);p['latest_label_end']=np.max(np.stack(maxlabels),axis=0)[common];p['feature_lag']=lag
   allpaths.append(p)
  log(f'子 {asset} {spec} 三期限完成')
 # Ratio stability, measured without outcome selection.
 d=daily(asset);r=d.rv5;btc=daily('BTC').rv5.reindex(d.index)
 for rw in [30,90,180]:
  a=np.sqrt(r.rolling(rw).mean()/btc.rolling(rw).mean());z=pd.concat([a,a.shift()],axis=1).dropna();rho=np.corrcoef(z.iloc[:,0],z.iloc[:,1])[0,1]
  threshold=d.daily_return.rolling(1000,min_periods=365).quantile(.01).shift();crash=d.daily_return<threshold
  v=a.diff();cr=v[crash].dropna();ci=mean_ci(cr,7)
  ratios.append(dict(asset=asset,window=rw,n=a.notna().sum(),mean=a.mean(),median=a.median(),q05=a.quantile(.05),q95=a.quantile(.95),ar1=rho,halflife=-np.log(2)/np.log(rho) if 0<rho<1 else np.nan,crash_n=len(cr),crash_jump=ci['mean'],crash_low=ci['ci_low'],crash_high=ci['ci_high']))
  csv(pd.DataFrame({'date':a.index,'ratio':a}),f'zi/ratio_{asset}_{rw}.csv')
 csv(gates,f'zi/gates_{asset}.csv');csv(comparisons,f'zi/forecast_{asset}.csv');csv(inside,f'zi/insample_{asset}.csv');csv(ratios,f'zi/stability_{asset}.csv')
 if allpaths:
  p=pd.concat(allpaths,ignore_index=True);p.to_parquet(OUT/f'zi/predictions_{asset}.parquet',index=False)
  if all(g['passed'] for g in gates):hedge(asset,p)
 return gates

def main():
 freeze();iv={c:iv_daily(c) for c in ['BTC','ETH']};coverage=[dict(currency=c,days=int(v.notna().sum()),first=str(v.index.min()),last=str(v.index.max()),passed=int(v.notna().sum())>=1500) for c,v in iv.items()];csv(coverage,'zi/iv_coverage.csv')
 if not all(c['passed'] for c in coverage):log('子 DVOL数据门未过，停止');return
 for asset in ASSETS[2:]:
  if (OUT/f'zi/hedge_{asset}.csv').exists():
   if 'funding_contract' not in pd.read_csv(OUT/f'zi/hedge_{asset}.csv',nrows=0):hedge(asset,pd.read_parquet(OUT/f'zi/predictions_{asset}.parquet'))
   continue
  run_asset(asset,iv)
 for stem in ['gates','forecast','insample','stability','hedge']:
  files=[OUT/f'zi/{stem}_{a}.csv' for a in ASSETS[2:]];csv(pd.concat([pd.read_csv(f) for f in files],ignore_index=True) if files else pd.DataFrame(),f'zi/{stem}.csv')
 p=pd.read_csv(OUT/'zi/forecast.csv');p=p[p.subsample.astype(str)=='all'];r=[]
 for (a,h,m),g in p.groupby(['asset','horizon','model']):r.append(dict(asset=a,horizon=h,model=m,denominator=len(g),positive=int((g.qlike_improvement_pct>0).sum()),negative=int((g.qlike_improvement_pct<0).sum()),significant_positive=int(((g.qlike_improvement_pct>0)&(g.dm_qlike_p<.05)).sum())))
 csv(r,'zi/robustness.csv');plt=plot_init();main=p[(p.spec=='rv5_log_w1000_lag0')&(p.horizon==7)];fig,ax=plt.subplots(figsize=(10,5))
 for i,m in enumerate(['M3','M5_90','M6_90']):
  g=main[main.model==m].set_index('asset').reindex(ASSETS[2:]);ax.errorbar(np.arange(8)+i*.2,g.qlike_improvement_pct,yerr=[g.qlike_improvement_pct-g.qlike_ci_low,g.qlike_ci_high-g.qlike_improvement_pct],fmt='o',label=m)
 ax.set_xticks(np.arange(8)+.2,ASSETS[2:]);ax.axhline(0,color='k',lw=.8);ax.set_ylabel('QLIKE improvement over own HAR (%)');ax.legend();fig.tight_layout();fig.savefig(OUT/'zi/forecast_7d.png',dpi=300);csv(main,'zi/forecast_7d_figure.csv');plt.close(fig)
 log('主题子完成')
if __name__=='__main__':main()
