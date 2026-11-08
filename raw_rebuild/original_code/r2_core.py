#!/usr/bin/env python3
from r2_common import *
from r2_clock import Clock,config
import argparse,itertools
import statsmodels.api as sm
from scipy.stats import t as student
KLIST=[1,2,4,8,12,24,48,72]
def integrate(clock,currency,cfg,origins,start,end):
 a=np.full(len(start),np.nan)
 for j in range(0,len(a),12000):a[j:j+12000]=clock.integral(currency,cfg,origins[j:j+12000],start[j:j+12000],end[j:j+12000])
 return a
def cases():
 result=[]
 for v,window,trim,holiday in itertools.product([1,2,3,4],[365,1000],[False,True],[False,True]):
  result.append(dict(case=config(v,window,trim,holiday),config=config(v,window,trim,holiday),band=.05,mintrades=3,include_block=False,subset='all'))
 for label,changes in [('band02',{'band':.02}),('band10',{'band':.10}),('min1',{'mintrades':1}),('min5',{'mintrades':5}),('blocks',{'include_block':True}),('exclude_top1',{'subset':'exclude_top1'}),('etf_pre',{'subset':'etf_pre'}),('etf_post',{'subset':'etf_post'}),('lag30',{'config':config(lag=30)})]:
  result.append(dict(case=label,config=PRIMARY,band=.05,mintrades=3,include_block=False,subset='all',**{})|changes)
 return result
def make_pairs(buckets,case):
 b=buckets[(buckets.band==case['band'])&(buckets.include_block==case['include_block'])&(buckets.n>=case['mintrades'])].copy()
 if case['subset']=='etf_pre':b=b[b.date<'2024-01-11']
 elif case['subset']=='etf_post':b=b[b.date>='2024-01-11']
 if case['subset']=='exclude_top1':
  d=pd.read_parquet(ROOT/'data_clean/market_daily.parquet');d=d[(d.market=='futures/um')&(d.asset==b.currency.iloc[0])&d.valid_day];cut=d.rv5.quantile(.99);bad=set(d.loc[d.rv5>cut,'date']);b=b[~b.date.isin(bad)]
 cols=['expiration_timestamp','hour_bucket','timestamp','n','w','F','remaining_hours','tenor','date'];b=b[cols];parts=[]
 for k in KLIST:
  following=b[['expiration_timestamp','hour_bucket','timestamp','w','F']].rename(columns={'timestamp':'timestamp2','w':'w2','F':'F2'});following['hour_bucket']-=pd.Timedelta(hours=k)
  z=b.merge(following,on=['expiration_timestamp','hour_bucket'],how='inner',validate='one_to_one');z['k']=k;parts.append(z)
 p=pd.concat(parts,ignore_index=True);p=p[(p.timestamp2>p.timestamp)&(p.timestamp2<p.expiration_timestamp)].copy();p['actual']=1-p.w2/p.w;p['clock0']=(p.timestamp2-p.timestamp)/(p.expiration_timestamp-p.timestamp);p['abs_return']=abs(np.log(p.F2/p.F));return p
def two_way_fit(z,controls=False):
 columns=['clock0','clock1']+(['abs_return','dvol_change'] if controls else [])
 z=z.dropna(subset=['actual']+columns);n=len(z);out=dict(n=n,dates=z.date.nunique(),expiries=z.expiration_timestamp.nunique(),status='estimated')
 if n<100 or out['dates']<20 or out['expiries']<10:out['status']='insufficient_sample';return out
 X=sm.add_constant(z[columns]).to_numpy();y=z.actual.to_numpy()
 if np.linalg.matrix_rank(X)<X.shape[1]:out['status']='rank_deficient';return out
 groups=np.column_stack([pd.factorize(z.expiration_timestamp)[0],pd.factorize(z.date)[0]])
 fit=sm.OLS(y,X).fit(cov_type='cluster',cov_kwds={'groups':groups});df=min(out['dates'],out['expiries'])-1
 for j,label in enumerate(['intercept']+columns):
  se=float(fit.bse[j]);co=float(fit.params[j]);stat=co/se if se>0 else np.nan;out.update({label+'_coef':co,label+'_se':se,label+'_p':float(2*student.sf(abs(stat),df)),label+'_t':stat})
 out['clock1_equal_one_p']=float(2*student.sf(abs((fit.params[2]-1)/fit.bse[2]),df));out['r2']=fit.rsquared;return out
def errors(z):
 out=dict(n=len(z),dates=z.date.nunique(),status='estimated' if len(z)>=100 else 'insufficient_sample')
 if not len(z):return out
 e0=abs(z.actual-z.clock0);e1=abs(z.actual-z.clock1)
 out.update(mae_clock0=e0.mean(),mae_clock1=e1.mean(),mae_improvement_pct=100*(1-e1.mean()/e0.mean()),mse_clock0=np.mean(e0**2),mse_clock1=np.mean(e1**2))
 daily=pd.DataFrame({'date':z.date,'difference':e0-e1}).groupby('date').difference.mean().sort_index();h=max(1,int(np.ceil(z.k.max()/24)));out['dm_stat'],out['dm_p']=dm(daily,max(7,2*h),horizon=h)
 for col in ['clock0','clock1']:
  if len(z)>=100 and z[col].std()>1e-12:
   fit=sm.OLS(z.actual,sm.add_constant(z[[col]])).fit();out[col+'_fit_r2']=fit.rsquared;out[col+'_fit_mae']=np.mean(abs(fit.resid));out[col+'_fit_slope']=fit.params.iloc[1]
 return out
def dvol_control(p,currency):
 h=pd.read_parquet(ROOT/f'data_clean/dvol_hourly_{currency}.parquet').sort_values('public_at');h['public_ms']=pd.to_datetime(h.public_at,utc=True).astype('int64')//10**6
 for label,col in [('dvol1','timestamp'),('dvol2','timestamp2')]:
  idx=np.searchsorted(h.public_ms.to_numpy(),p[col].to_numpy(),side='right')-1;good=idx>=0;idx=np.maximum(idx,0);p[label]=np.where(good,h.close.to_numpy()[idx],np.nan)
 p['dvol_change']=np.log(p.dvol2/p.dvol1)
 # Historical, origin-specific median threshold: previous 1000 days of
 # published hourly absolute DVOL changes, without the current interval.
 idx=np.searchsorted(h.public_ms.to_numpy(),p.timestamp.to_numpy(),side='right')-1;good=idx>=0;idx=np.maximum(idx,0);threshold=np.full(len(p),np.nan)
 for k in KLIST:
  actual_gap=h.public_ms-h.public_ms.shift(k);change=abs(np.log(h.close).diff(k)).where(actual_gap==k*3600000);historical_median=change.rolling(24000,min_periods=1000).median().shift(1).to_numpy();select=p.k.to_numpy()==k;threshold[select]=historical_median[idx[select]]
 p['dvol_low_threshold']=threshold;p.loc[~good,'dvol_low_threshold']=np.nan
 return p
def patterns(b,currency,case,clock):
 q=b[(b.band==case['band'])&(b.include_block==case['include_block'])&(b.n>=case['mintrades'])].copy()
 if case['subset']=='etf_pre':q=q[q.date<'2024-01-11']
 elif case['subset']=='etf_post':q=q[q.date>='2024-01-11']
 elif case['subset']=='exclude_top1':
  d=pd.read_parquet(ROOT/'data_clean/market_daily.parquet');d=d[(d.market=='futures/um')&(d.asset==currency)&d.valid_day];q=q[~q.date.isin(d.loc[d.rv5>d.rv5.quantile(.99),'date'])]
 orig=q.date.to_numpy('datetime64[D]');C=integrate(clock,currency,case['config'],orig,q.timestamp.to_numpy(),q.expiration_timestamp.to_numpy());q['log0']=.5*np.log(q.w/((q.expiration_timestamp-q.timestamp)/(365*86400000)));q['log1']=.5*np.log(365*q.w/C);q=q.sort_values(['expiration_timestamp','hour_bucket'])
 prev=q.groupby('expiration_timestamp').hour_bucket.shift();same=(q.hour_bucket-prev)==pd.Timedelta(hours=1)
 q['change0']=q.groupby('expiration_timestamp').log0.diff();q['change1']=q.groupby('expiration_timestamp').log1.diff();q=q[same].dropna(subset=['change0','change1']);q['slot']=q.hour_bucket.dt.dayofweek*24+q.hour_bucket.dt.hour;rows=[]
 for tenorname,z in [('all',q)]+[(t,z) for t,z in q.groupby('tenor',observed=True)]:
  result=dict(currency=currency,case=case['case'],tenor=tenorname,n=len(z),status='estimated' if len(z)>=100 else 'insufficient_sample')
  for var in ['change0','change1']:
   if len(z):
    fitted=z.groupby('slot')[var].transform('mean');sst=np.sum((z[var]-z[var].mean())**2);result[var+'_r2']=1-np.sum((z[var]-fitted)**2)/sst if sst>0 else np.nan
  if len(z):result['pattern_reduction_pct']=100*(1-result['change1_r2']/result['change0_r2']) if result['change0_r2']>0 else np.nan
  rows.append(result)
 return rows
def weekend_path(p,currency):
 z=p[(p.hour_bucket.dt.dayofweek==4)&(p.hour_bucket.dt.hour==16)&(p.remaining_hours>=56)&(p.k<=56)].copy()
 z['actual_remaining_ratio']=1-z.actual;z['clock_remaining_ratio']=1-z.clock0;z['variance_remaining_ratio']=1-z.clock1
 if len(z):return z.groupby('k')[['actual_remaining_ratio','clock_remaining_ratio','variance_remaining_ratio']].mean().reset_index().assign(currency=currency,n=z.groupby('k').size().to_numpy())
 return pd.DataFrame(columns=['k','currency','n','actual_remaining_ratio','clock_remaining_ratio','variance_remaining_ratio'])
def full_weekend_path(b,currency,case,clock):
 q=b[(b.band==case['band'])&(b.include_block==case['include_block'])&(b.n>=case['mintrades'])].copy();left=q[(q.hour_bucket.dt.dayofweek==4)&(q.hour_bucket.dt.hour==16)&(q.remaining_hours>=56)].copy();parts=[]
 for k in range(1,57):
  following=q[['expiration_timestamp','hour_bucket','timestamp','w']].rename(columns={'timestamp':'timestamp2','w':'w2'});following['hour_bucket']-=pd.Timedelta(hours=k);z=left.merge(following,on=['expiration_timestamp','hour_bucket'],validate='one_to_one');z=z[(z.timestamp2>z.timestamp)&(z.timestamp2<z.expiration_timestamp)].copy()
  if not len(z):continue
  z['k']=k;z['actual']=1-z.w2/z.w;z['clock0']=(z.timestamp2-z.timestamp)/(z.expiration_timestamp-z.timestamp);o=z.date.to_numpy('datetime64[D]');num=integrate(clock,currency,PRIMARY,o,z.timestamp.to_numpy(),z.timestamp2.to_numpy());den=integrate(clock,currency,PRIMARY,o,z.timestamp.to_numpy(),z.expiration_timestamp.to_numpy());z['clock1']=num/den;parts.append(z.dropna(subset=['clock1']))
 return weekend_path(pd.concat(parts,ignore_index=True),currency) if parts else pd.DataFrame(columns=['k','currency','n','actual_remaining_ratio','clock_remaining_ratio','variance_remaining_ratio'])
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--primary-only',action='store_true');args=ap.parse_args();clock=Clock();reg=cases();csv(reg,'core_specifications.csv');fits=[];loss=[];patterns_all=[];weekends=[]
 for currency in ['BTC','ETH']:
  b=pd.read_parquet(OUT/f'buckets_{currency}.parquet')
  shared_case=dict(band=.05,mintrades=3,include_block=False,subset='all');shared_pairs=dvol_control(make_pairs(b,shared_case),currency)
  for case in reg:
   if args.primary_only and case['case']!=PRIMARY:continue
   cfg=case['config']
   if (currency,cfg) not in clock.lookup:
    for t in TENORS:loss.append(dict(currency=currency,case=case['case'],tenor=t,k='all',status='event_clock_unavailable' if cfg.startswith('v4_') else 'clock_unavailable'))
    continue
   same=all(case[key]==value for key,value in shared_case.items());p=shared_pairs.copy() if same else dvol_control(make_pairs(b,case),currency)
   orig=p.date.to_numpy('datetime64[D]');cons=integrate(clock,currency,cfg,orig,p.timestamp.to_numpy(),p.timestamp2.to_numpy());total=integrate(clock,currency,cfg,orig,p.timestamp.to_numpy(),p.expiration_timestamp.to_numpy());p['clock1']=cons/total;p=p.dropna(subset=['clock1'])
   primary=case['case']==PRIMARY
   if primary or case['case']=='lag30':p.to_parquet(OUT/f'core_pairs_{currency}_{case["case"]}.parquet',index=False)
   for t,z in [('all',p)]+[(t,p[p.tenor==t]) for t in TENORS]:
    for k,sample in [('all',z)]+[(str(k),z[z.k==k]) for k in KLIST]:
     keys=dict(currency=currency,case=case['case'],config=cfg,tenor=t,k=k)
     fits.append(keys|two_way_fit(sample));loss.append(keys|errors(sample))
     if primary:
      fits.append(keys|dict(controls='return_dvol')|two_way_fit(sample,True));low=sample[abs(sample.dvol_change)<sample.dvol_low_threshold];fits.append(keys|dict(controls='low_dvol')|two_way_fit(low,True));loss.append(keys|dict(controls='low_dvol')|errors(low))
   patterns_all.extend(patterns(b,currency,case,clock))
   if primary:weekends.append(full_weekend_path(b,currency,case,clock))
   log(f'核心检验 {currency}/{case["case"]}：{len(p):,} 个事前时钟配对')
 f=csv(fits,'core_race.csv');l=csv(loss,'core_errors.csv');pat=csv(patterns_all,'seasonality_removal.csv');csv(pd.concat(weekends) if weekends else [],'weekend_path.csv')
 if 'controls' not in f:f['controls']=np.nan
 if 'controls' not in l:l['controls']=np.nan
 mainf=f[(f.case==PRIMARY)&(f.tenor=='all')&(f.k=='all')&f.controls.isna()];mainl=l[(l.case==PRIMARY)&(l.k=='all')&l.controls.isna()];mainpat=pat[(pat.case==PRIMARY)&(pat.tenor=='all')]
 rows=[]
 for currency in ['BTC','ETH']:
  z=mainf[mainf.currency==currency];p=mainpat[mainpat.currency==currency];s=mainl[(mainl.currency==currency)&mainl.tenor.isin(TENORS)]
  rows.append(dict(currency=currency,positive_clock_coef_pass=len(z)==1 and z.iloc[0].get('clock1_coef',np.nan)>0 and z.iloc[0].get('clock1_p',np.nan)<.05,significant_better_tenors=int(((s.mae_improvement_pct>0)&(s.dm_p<.05)).sum()),pattern_reduction_pct=p.iloc[0].get('pattern_reduction_pct',np.nan) if len(p) else np.nan))
 r=csv(rows,'gate2_components.csv');common=set.intersection(*[set(mainl.loc[(mainl.currency==c)&mainl.tenor.isin(TENORS)&(mainl.mae_improvement_pct>0)&(mainl.dm_p<.05),'tenor']) for c in ['BTC','ETH']]);passed=bool(r.positive_clock_coef_pass.all() and len(common)>=2 and (r.pattern_reduction_pct>=100/3).all());csv([dict(gate=2,passed=passed,common_better_tenors=len(common),adjustment='not_used_no_demonstrated_construction_defect')],'gate2.csv')
 robust=l[l.tenor.isin(TENORS[:2])&(l.k=='all')&(l.case!='lag30')&l.controls.isna()].copy();robust['direction']=robust.mae_improvement_pct>0;robust['significant']=robust.direction&(robust.dm_p<.05);csv(robust,'core_robustness.csv');log(f'核心闸门2={passed}，继续定价与盯市模块。')
if __name__=='__main__':main()
