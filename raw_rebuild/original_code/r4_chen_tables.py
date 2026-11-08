from r4_common import *

def stats_row(z,metric,lag):
 x=z[metric].to_numpy();co=ci(x,lag);v=x[np.isfinite(x)];return dict(metric=metric,**co,median=float(np.median(v)) if len(v) else np.nan,positive_share=float((v>0).mean()) if len(v) else np.nan,q01=float(np.quantile(v,.01)) if len(v) else np.nan,q05=float(np.quantile(v,.05)) if len(v) else np.nan,q95=float(np.quantile(v,.95)) if len(v) else np.nan,q99=float(np.quantile(v,.99)) if len(v) else np.nan)

def main():
 moments=[];ratios=[];corrections=[];dvolrows=[];groups=[];tails=[];cdf=[];coverage=[];marks=[];sides=[];pairs=[];currency_parts=[];dataquality=[];slope=[]
 events=pd.read_csv(ROOT/'out/R3/shared/events.csv');ev=pd.to_datetime(events.datetime_utc,utc=True).dt.tz_localize(None).to_numpy();clock=Clock(assets=['BTC','ETH'])
 for c in ['BTC','ETH']:
  p=pd.read_parquet(OUT/f'chen/term_panel_{c}.parquet');p=p[p.window_hour==8].copy();p['mf_ratio']=p.mf_w/p.atm_w;p['mf_vrp']=p.mf_w-p.rv_total;p['mf_volpts']=100*(np.sqrt(p.mf_w*365/p.tenor)-np.sqrt(p.rv_annual));p['buy_vrp']=p.buy_w-p.rv_total;p['sell_vrp']=p.sell_w-p.rv_total;p['side_gap']=p.buy_w-p.sell_w
  for h,g in p.groupby('tenor'):
   lag=max(7,2*h)
   for split,z in [('all',g)]+[(str(y),z) for y,z in g.groupby(g.date.dt.year)]:
    for metric in ['vrp','vrp_annual','volpts']:moments.append(dict(currency=c,tenor=h,split=split,**stats_row(z,metric,lag)))
    zr=z[np.isfinite(z.mf_ratio)];coverage.append(dict(currency=c,tenor=h,split=split,atm_days=len(z),mf_days=len(zr),mf_share=len(zr)/len(z),tail_extended_days=int(zr.tail_extended.sum()),mean_tail_share=zr.tail_share.mean()))
    if len(zr):ratios.append(dict(currency=c,tenor=h,split=split,**stats_row(zr,'mf_ratio',lag)))
   ratio=g.mf_ratio.mean();gg=g.copy();gg['mean_ratio_vrp']=gg.atm_w*ratio-gg.rv_total;gg['mean_ratio_volpts']=100*(np.sqrt(gg.atm_w*ratio*365/h)-np.sqrt(gg.rv_annual))
   for metric in ['mf_vrp','mf_volpts','mean_ratio_vrp','mean_ratio_volpts']:
    corrections.append(dict(currency=c,tenor=h,method='daily' if metric.startswith('mf') else 'tenor_mean_ratio',ratio=ratio,**stats_row(gg,metric,lag)))
   z=g.dropna(subset=['buy_vrp','sell_vrp','rv_total'])
   for metric in ['buy_vrp','sell_vrp','side_gap']:sides.append(dict(currency=c,tenor=h,**stats_row(z,metric,lag)))
   z=g.dropna(subset=['rv_total']).copy();cut1=z.rv_total.quantile(.99);cut5=z.rv_total.quantile(.95);win=z.vrp.clip(z.vrp.quantile(.01),z.vrp.quantile(.99))
   for split,s in [('all',z)]+[(str(y),s) for y,s in z.groupby(z.date.dt.year)]:
    for method,x in [('mean',s.vrp),('trimRV1',s.loc[s.rv_total<=s.rv_total.quantile(.99),'vrp']),('trimRV5',s.loc[s.rv_total<=s.rv_total.quantile(.95),'vrp']),('winsor1',s.vrp.clip(s.vrp.quantile(.01),s.vrp.quantile(.99)))]:tails.append(dict(currency=c,tenor=h,split=split,method=method,median=s.vrp.median(),positive_share=(s.vrp>0).mean(),**ci(x.to_numpy(),lag)))
   zz=z.sort_values('rv_total',ascending=False).copy();zz['rank']=np.arange(1,len(zz)+1);zz['day_share']=zz['rank']/len(zz);zz['cum_vrp']=zz.vrp.cumsum();zz['total_vrp']=zz.vrp.sum();zz['cum_fraction_of_net']=zz.cum_vrp/zz.total_vrp;cdf.append(zz[['currency','date','tenor','rank','day_share','rv_total','vrp','cum_vrp','total_vrp','cum_fraction_of_net']]);exhaust=int(zz.loc[zz.cum_vrp<=zz.total_vrp,'rank'].min()) if zz.total_vrp.iloc[0]<0 and (zz.cum_vrp<=zz.total_vrp).any() else np.nan
   tails.append(dict(currency=c,tenor=h,split='all',method='tail_contribution',n=len(zz),mean=zz.vrp.mean(),net_total=zz.vrp.sum(),highest_RV_days_cover_net_negative=exhaust,top1_RV_sum=zz.loc[zz.rv_total>cut1,'vrp'].sum(),top5_RV_sum=zz.loc[zz.rv_total>cut5,'vrp'].sum()))
   if h==1:
    z=z.sort_values('date').copy();past=g.sort_values('date').set_index('date').iv_annual.reindex(pd.date_range(g.date.min(),g.date.max())).shift(1);z['previous_IV']=past.reindex(z.date).to_numpy();q=past.quantile([1/3,2/3]).to_numpy();z['iv_tercile']=pd.cut(z.previous_IV,[-np.inf,*q,np.inf],labels=['low','middle','high']);z['iv_tercile_past']=pd.cut(z.previous_IV,[-np.inf,*q,np.inf],labels=['low','middle','high']).astype(str)
    for j,row in z.iterrows():
     hist=past.loc[(past.index<row.date)&(past.index>=row.date-pd.Timedelta(days=1000))].dropna()
     if len(hist)>=100:
      qq=hist.quantile([1/3,2/3]);z.loc[j,'iv_tercile_past']='low' if row.previous_IV<=qq.iloc[0] else 'middle' if row.previous_IV<=qq.iloc[1] else 'high'
     else:z.loc[j,'iv_tercile_past']='missing_history'
    z['macro']=np.searchsorted(ev,z.target_end.to_numpy(),side='right')>np.searchsorted(ev,z.observed_at.to_numpy(),side='right');z['weekend']=z.date.dt.dayofweek>=5;a=grid(c);z['prior_return']=[a.logret.iloc[a.index.searchsorted(w-pd.Timedelta(days=1),side='right'):a.index.searchsorted(w,side='right')].sum(min_count=288) for w in z.observed_at];z['return_sign']=np.where(z.prior_return>=0,'nonnegative','negative');share=clock.integral(c,'v3_w1000_trim0_hol0_lag0',z.date.to_numpy('datetime64[D]'),z.observed_at.astype('int64').to_numpy()/1e6,z.target_end.astype('int64').to_numpy()/1e6);z['clock_vrp']=z.vrp/share;z['clock_volpts']=100*(np.sqrt(z.atm_w*365/share)-np.sqrt(z.rv_total*365/share))
    for grouping in ['macro','weekend','iv_tercile','iv_tercile_past','return_sign']:
     for label,s in z.groupby(grouping,observed=True):
      for metric in ['vrp','volpts','clock_vrp','clock_volpts']:groups.append(dict(currency=c,grouping=grouping,group=str(label),**stats_row(s,metric,7)))
    z['tail_group']=np.where(z.rv_total>cut5,'RV_top5','other');z['IV_volpts']=100*np.sqrt(z.iv_annual);z['RV_volpts']=100*np.sqrt(z.rv_annual)
    for label,s in z.groupby('tail_group'):
     for metric in ['IV_volpts','RV_volpts','volpts']:currency_parts.append(dict(currency=c,group=label,weight=len(s)/len(z),**ci(s[metric].to_numpy(),7)))
    currency_parts.append(dict(currency=c,group='all',metric='volpts',weight=1,**ci(z.volpts.to_numpy(),7)))
  t=p[p.tenor.isin([1,30])].pivot(index='date',columns='tenor',values=['vrp_annual','volpts']);t=t.dropna()
  for metric in ['vrp_annual','volpts']:pairs.append(dict(currency=c,metric=metric,comparison='1_minus_30',**ci((t[metric][1]-t[metric][30]).to_numpy(),60)))
  dv=pd.read_parquet(ROOT/f'data_clean/dvol_hourly_{c}.parquet').sort_values('public_at');times=pd.to_datetime(dv.public_at).dt.tz_localize(None).to_numpy();g=p[p.tenor==30].copy();pos=np.searchsorted(times,g.observed_at.to_numpy(),side='right')-1;good=(pos>=0);safe=np.maximum(pos,0);good&=(g.observed_at.to_numpy()-times[safe])<=np.timedelta64(1,'h');g['DVOL_variance']=np.where(good,(dv.close.to_numpy()[safe]/100)**2,np.nan);g['difference_annual']=g.DVOL_variance-g.iv_annual;g['difference_volpts']=100*(np.sqrt(g.DVOL_variance)-np.sqrt(g.iv_annual));g['DVOL_level']=pd.qcut(g.DVOL_variance,3,labels=['low','middle','high'],duplicates='drop')
  csv(g,f'chen/atm_dvol_daily_{c}.csv')
  for label,z in [('all',g)]+[(str(v),z) for v,z in g.groupby('DVOL_level',observed=True)]:
   for metric in ['difference_annual','difference_volpts']:dvolrows.append(dict(currency=c,volatility_group=label,**stats_row(z,metric,60)))
  quality=pd.read_csv(OUT/f'chen/data_daily_{c}.csv',parse_dates=['date'])
  for year,z in [('all',quality)]+[(str(y),s) for y,s in quality.groupby(quality.date.dt.year)]:
   v=z.atm_trades;dataquality.append(dict(currency=c,year=year,calendar_days=len(z),available_days=int((v>=3).sum()),atm_trade_n=int(v.sum()),n_mean=v.mean(),n_median=v.median(),n_p01=v.quantile(.01),n_p05=v.quantile(.05),n_p95=v.quantile(.95),n_p99=v.quantile(.99),buy_share=z.buy_n.sum()/v.sum() if v.sum() else np.nan,sell_share=z.sell_n.sum()/v.sum() if v.sum() else np.nan))
  mk=pd.read_parquet(OUT/f'chen/trade_mark_{c}.parquet');mk=mk[mk.mark_valid].copy()
  for split,z in [('all',mk),('buy',mk[mk.direction=='buy']),('sell',mk[mk.direction=='sell'])]+[(str(y),s) for y,s in mk.groupby(mk.date.dt.year)]:
   dailymean=z.groupby('date').difference_volpts.mean();vals=z.difference_volpts
   marks.append(dict(currency=c,split=split,trade_n=len(z),daily_n=len(dailymean),median=vals.median(),q01=vals.quantile(.01),q05=vals.quantile(.05),q95=vals.quantile(.95),q99=vals.quantile(.99),trade_mean=vals.mean(),**ci(dailymean.to_numpy(),7)))
  predpath=OUT/f'chen/predictions_{c}_w1000_log_hour8_h7.parquet'
  if predpath.exists():
   d=pd.read_parquet(predpath);q7=p[p.tenor==7].set_index('date');q1=pd.read_parquet(OUT/f'chen/expiry_quotes_{c}.parquet');q1=q1[(q1.window_hour==8)&(q1.expiry==q1.date+pd.Timedelta(days=1,hours=8))].set_index('date');d['slope']=q1.atm_w.reindex(d.date).to_numpy()*365/(23/24)-q7.iv_annual.reindex(d.date).to_numpy();one=pd.read_parquet(OUT/f'chen/predictions_{c}_w1000_log_hour8_h1.parquet').set_index('date');d['change']=d.y/(7-1/24)-one.y.reindex(d.date).to_numpy()/(23/24);z=d[['change','slope','rv24','rv7','rv30']].dropna()
   if len(z)>=30:
    fit=sm.OLS(z.change,sm.add_constant(z.drop(columns='change'))).fit(cov_type='HAC',cov_kwds={'maxlags':14})
    for v in fit.params.index:slope.append(dict(currency=c,variable=v,n=len(z),coef=fit.params[v],se=fit.bse[v],p=fit.pvalues[v],ci_low=fit.conf_int().loc[v,0],ci_high=fit.conf_int().loc[v,1]))
 common_parts=[]
 aa={}
 for c in ['BTC','ETH']:
  z=pd.read_parquet(OUT/f'chen/term_panel_{c}.parquet');aa[c]=z[(z.window_hour==8)&(z.tenor==1)].set_index('date')[['iv_annual','rv_annual']].dropna()
 common=aa['BTC'].join(aa['ETH'],lsuffix='_BTC',rsuffix='_ETH',how='inner')
 common['IV_gap']=100*(np.sqrt(common.iv_annual_ETH)-np.sqrt(common.iv_annual_BTC));common['RV_gap']=100*(np.sqrt(common.rv_annual_ETH)-np.sqrt(common.rv_annual_BTC));common['premium_gap']=common.IV_gap-common.RV_gap;tail=(common.rv_annual_ETH>common.rv_annual_ETH.quantile(.95))|(common.rv_annual_BTC>common.rv_annual_BTC.quantile(.95));common['group']=np.where(tail,'RV_top5_union','other')
 for group,z in [('all',common)]+list(common.groupby('group')):
  for metric in ['IV_gap','RV_gap','premium_gap']:common_parts.append(dict(group=group,metric=metric,weight=len(z)/len(common),contribution=z[metric].mean()*len(z)/len(common),**ci(z[metric].to_numpy(),7)))
 csv(common_parts,'chen/currency_common_decomposition.csv')
 for stem,rows in [('term_premium',moments),('model_free_ratios',ratios),('smile_corrections',corrections),('model_free_coverage',coverage),('atm_dvol_comparison',dvolrows),('negative_premium_groups',groups),('tail_statistics',tails),('price_direction_bounds',sides),('term_difference',pairs),('currency_decomposition',currency_parts),('data_quality',dataquality),('trade_mark_comparison',marks),('slope_predictability',slope)]:csv(rows,'chen/'+stem+'.csv')
 csv(pd.concat(cdf,ignore_index=True),'chen/tail_contribution_path.csv');log('辰 统一期限、微笑偏差、尾部来源与成交口径表完成')
if __name__=='__main__':main()
