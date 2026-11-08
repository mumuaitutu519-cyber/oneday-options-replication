from r5_common import *
from r4_chen_tables import stats_row

def interval_both(x,block=7):
 n=int(np.isfinite(np.asarray(x,float)).sum())
 if n<10:return dict(boot_low=np.nan,boot_high=np.nan,median_boot_low=np.nan,median_boot_high=np.nan,bootstrap_n=n)
 mean,median=stationary_statistics(x,block);a,b=np.nanquantile(mean,[.025,.975]);c,d=np.nanquantile(median,[.025,.975]);return dict(boot_low=a,boot_high=b,median_boot_low=c,median_boot_high=d,bootstrap_n=n)

def load_panel(c):
 base=pd.read_parquet(R4/f'term_panel_{c}.parquet');base=base[base.window_hour==8].copy();base['variant']='trade_ATM';parts=[base];mark=pd.read_parquet(CHEN/f'term_panel_{c}.parquet');parts.append(mark[mark.variant.isin(['mark','bilateral'])].copy());mf=base.copy();mf['variant']='model_free';mf['atm_w']=mf.mf_w;parts.append(mf)
 diag=pd.read_csv(CHEN/f'numerical_grid_diagnostics_{c}.csv',parse_dates=['date']);z=base.merge(diag[['date','tenor','corrected_mf_w','multiplicative_mf_w']],on=['date','tenor'],how='inner',validate='one_to_one')
 for variant,col in [('model_free_grid_deducted','corrected_mf_w'),('model_free_grid_ratio','multiplicative_mf_w')]:a=z.copy();a['variant']=variant;a['atm_w']=a[col];parts.append(a)
 p=pd.concat(parts,ignore_index=True);p['iv_annual']=p.atm_w*365/p.tenor;p['rv_annual']=p.rv_total*365/p.tenor;p['vrp']=p.atm_w-p.rv_total;p['vrp_annual']=p.vrp*365/p.tenor;p['volpts']=100*(np.sqrt(p.iv_annual.where(p.iv_annual>0))-np.sqrt(p.rv_annual.where(p.rv_annual>0)));p['log_ratio']=np.log(p.atm_w.where(p.atm_w>0)/p.rv_total.where(p.rv_total>0));p.to_parquet(CHEN/f'premium_panel_{c}.parquet',index=False);return p

def main():
 moments=[];tail=[];paths=[];pairs=[];annual=[];regimes=[];gridrows=[];availability=[]
 for c in ['BTC','ETH']:
  p=load_panel(c)
  for (variant,h),g in p.groupby(['variant','tenor']):
   g=g.sort_values('date').set_index('date',drop=False);lag=max(7,2*h);g=g.dropna(subset=['rv_total','atm_w']);g=g[g.rv_total>0]
   if not len(g):continue
   history=g.set_index('date').iv_annual.reindex(pd.date_range(g.date.min(),g.date.max()));threshold=history.shift(1).rolling(500,min_periods=120);lo=threshold.quantile(1/3);hi=threshold.quantile(2/3);g['iv_tercile_past']=np.where(g.iv_annual<=lo.reindex(g.date).to_numpy(),'low',np.where(g.iv_annual<=hi.reindex(g.date).to_numpy(),'middle','high'));missing=lo.reindex(g.date).isna().to_numpy();g.loc[missing,'iv_tercile_past']='missing_history'
   splits=[('all',g),('ETF_pre',g[g.date<'2024-01-11']),('ETF_post',g[g.date>='2024-01-11'])]+[(str(y),z) for y,z in g.groupby(g.date.dt.year)]+[('IV_'+label,z) for label,z in g.groupby('iv_tercile_past') if label!='missing_history']
   for split,z in splits:
    if len(z)<10:continue
    for metric in ['vrp','vrp_annual','volpts','log_ratio']:
     rr=stats_row(z,metric,lag);v=z[metric].dropna();bs=interval_both(z[metric]);moments.append(dict(currency=c,tenor=h,split=split,variant=variant,**rr,q25=v.quantile(.25),q50=v.quantile(.5),q75=v.quantile(.75),**bs,stationary_block=7,hac_lag=lag,nonpositive_IV=int((z.atm_w<=0).sum())))
    if split=='all' and h>3:
     for metric in ['vrp','volpts','log_ratio']:moments.append(dict(currency=c,tenor=h,split='overlap_block',variant=variant,**stats_row(z,metric,lag),**interval_both(z[metric],block=2*h),stationary_block=2*h,hac_lag=lag,nonpositive_IV=int((z.atm_w<=0).sum())))
   for method,x in [('mean',g.vrp),('trimRV1',g.loc[g.rv_total<=g.rv_total.quantile(.99),'vrp']),('trimRV5',g.loc[g.rv_total<=g.rv_total.quantile(.95),'vrp']),('winsor1',g.vrp.clip(g.vrp.quantile(.01),g.vrp.quantile(.99)))]:tail.append(dict(currency=c,tenor=h,split='all',variant=variant,method=method,median=g.vrp.median(),positive_share=(g.vrp>0).mean(),**ci(x,lag),**interval_both(x)))
   zz=g.sort_values('rv_total',ascending=False).copy();zz['rank']=np.arange(1,len(zz)+1);zz['day_share']=zz['rank']/len(zz);zz['cum_vrp']=zz.vrp.cumsum();total=zz.vrp.sum();zz['total_vrp']=total;zz['cum_fraction_of_net']=zz.cum_vrp/total;paths.append(zz[['currency','date','tenor','variant','rank','day_share','rv_total','vrp','cum_vrp','total_vrp','cum_fraction_of_net']]);top5=g.loc[g.rv_total>g.rv_total.quantile(.95),'vrp'].sum();top1=g.loc[g.rv_total>g.rv_total.quantile(.99),'vrp'].sum();exhaust=int(zz.loc[zz.cum_vrp<=total,'rank'].min()) if total<0 and (zz.cum_vrp<=total).any() else np.nan
   tail.append(dict(currency=c,tenor=h,split='all',variant=variant,method='tail_contribution',n=len(g),mean=g.vrp.mean(),net_total=total,highest_RV_days_cover_net_negative=exhaust,top1_RV_sum=top1,top5_RV_sum=top5,top5_fraction_of_net=top5/total if total else np.nan,top1_fraction_of_net=top1/total if total else np.nan));availability.append(dict(currency=c,tenor=h,variant=variant,days=len(g),nonpositive_iv=int((g.atm_w<=0).sum()),start=g.date.min(),end=g.date.max()))
   if h==1:regimes.append(g[['currency','date','variant','iv_annual','iv_tercile_past']])
  for variant,g in p.groupby('variant'):
   t=g[g.tenor.isin([1,30])].pivot(index='date',columns='tenor',values=['vrp_annual','volpts']).dropna()
   if 1 not in t.columns.get_level_values(1) or 30 not in t.columns.get_level_values(1):continue
   for metric in ['vrp_annual','volpts']:pairs.append(dict(currency=c,variant=variant,metric=metric,comparison='1_minus_30',**ci((t[metric][1]-t[metric][30]).to_numpy(),60),**interval_both(t[metric][1]-t[metric][30])))
  quality=pd.read_csv(CHEN/f'trade_quality_{c}.csv',parse_dates=['date']);t=p[(p.tenor==1)&p.variant.isin(['trade_ATM','mark'])];pivot=t.pivot(index='date',columns='variant',values='volpts');d=quality.merge(pivot,left_on='date',right_index=True,how='left');stratpath=CHEN/f'strategy_path_{c}.parquet'
  if stratpath.exists():
   s=pd.read_parquet(stratpath);s=s[(s.variant=='last_binance_entry_IV_hourly')&(s.side=='sell')&(s.bps==0)&(s.spread=='none')].set_index('date').usd_10000;d['zero_cost_seller']=d.date.map(s)
  else:d['zero_cost_seller']=np.nan
  for year,z in d.groupby(d.date.dt.year):annual.append(dict(currency=c,year=int(year),variant='annual_trade_summary',calendar_n=len(z),trade_n=int(z.trade_n.sum()),sell_share=z.sell_n.sum()/z.trade_n.sum() if z.trade_n.sum() else np.nan,trade_mark_mean=(z.trade_mark_mean*z.mark_valid_n).sum()/z.mark_valid_n.sum() if z.mark_valid_n.sum() else np.nan,trade_mark_daily_mean=z.trade_mark_mean.mean(),atm_mean=z.trade_ATM.mean(),atm_median=z.trade_ATM.median(),mark_mean=z.mark.mean(),zero_cost_seller=z.zero_cost_seller.mean(),seller_trade_n=int(z.zero_cost_seller.notna().sum())))
  diag=pd.read_csv(CHEN/f'numerical_grid_diagnostics_{c}.csv')
  for h,z in diag.groupby('tenor'):gridrows.append(dict(currency=c,tenor=h,variant='flat_grid_all',n=len(z),flat_grid_ratio_mean=z.flat_grid_ratio.mean(),median=z.flat_grid_ratio.median(),q05=z.flat_grid_ratio.quantile(.05),q95=z.flat_grid_ratio.quantile(.95),min=z.flat_grid_ratio.min(),max=z.flat_grid_ratio.max(),actual_ratio_mean=z.actual_ratio.mean(),deducted_ratio_mean=(z.corrected_mf_w/z.atm_w).mean(),nonpositive_corrected=int(z.nonpositive.sum())))
  log(f'R5 {c} premium robust inference complete')
 csv(moments,'term_premium.csv');csv(tail,'tail_statistics.csv');csv(pd.concat(paths,ignore_index=True),'tail_contribution_path.csv');csv(pairs,'term_difference.csv');csv(annual,'annual_overview.csv');csv(pd.concat(regimes,ignore_index=True),'IV_tercile_audit.csv');csv(gridrows,'numerical_grid_summary.csv');csv(availability,'premium_coverage.csv')
if __name__=='__main__':main()
