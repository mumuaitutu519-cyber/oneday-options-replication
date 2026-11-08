#!/usr/bin/env python3
from r2_common import *
import argparse,itertools,requests,re
from pandas.tseries.holiday import USFederalHolidayCalendar
from scipy.stats import norm
def holidays():
 h=USFederalHolidayCalendar().holidays('2015-01-01','2027-12-31')
 return h[~((h.month==6)&(h.year<2021))]
def official_calendars(offline=False):
 folder=OUT/'raw/calendars';folder.mkdir(parents=True,exist_ok=True);receipts=[]
 urls={'bls2024':'https://www.bls.gov/schedule/2024/home.htm','fomc':'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm','cme_hours':'https://www.cmegroup.com/education/courses/introduction-to-bitcoin/what-are-bitcoin-futures','cme_247':'https://cmegroupclientsite.atlassian.net/wiki/spaces/EPICSANDBOX/pages/1283194884/Cryptocurrency+Futures+and+Options+Migration+to+24-7+Trading'}
 for label,url in urls.items():
  p=folder/(label+'.html')
  if not p.exists() and not offline:
   try:
    r=requests.get(url,timeout=35);p.write_bytes(r.content);status=r.status_code
   except Exception as e:receipts.append(dict(label=label,url=url,status='failed',reason=repr(e)));continue
  else:status='cached' if p.exists() else 'missing'
  receipts.append(dict(label=label,url=url,status=status,sha256=sha(p) if p.exists() else '',path=str(p.relative_to(ROOT))))
 csv(receipts,'official_calendar_sources.csv')
 from r2_events import prepare_events
 prepare_events(offline)
def prepare():
 d=pd.read_parquet(ROOT/'data_clean/market_daily.parquet');d=d[d.market=='futures/um'].copy();d.date=pd.to_datetime(d.date)
 columns=['hour_'+str(i) for i in range(24)]
 a=d.melt(id_vars=['date','asset','valid_day','rv5','funding_window','expiry_window'],value_vars=columns,var_name='hour',value_name='rv')
 a['hour']=a.hour.str.replace('hour_','').astype(int);a['datetime']=a.date+pd.to_timedelta(a.hour,unit='h');a['dow']=a.datetime.dt.dayofweek;a['slot']=a.dow*24+a.hour
 a['valid']=a.valid_day.astype(bool)&np.isfinite(a.rv)&(a.rv>=0)
 a=a.sort_values(['asset','datetime']).reset_index(drop=True);a.to_parquet(OUT/'hours.parquet',index=False)
 return d,a
def profiles(d):
 result=[];hs=set(holidays());cols=['hour_'+str(i) for i in range(24)]
 for asset,g in d.groupby('asset'):
  g=g.sort_values('date').set_index('date');idx=pd.date_range(g.index.min(),g.index.max()+pd.Timedelta(days=1),freq='D');g=g.reindex(idx)
  values=g[cols].to_numpy(float);values[~g.valid_day.fillna(False).to_numpy(bool)]=np.nan
  dow=idx.dayofweek.to_numpy();ish=np.array([x in hs for x in idx]);ordinary=np.tile(np.arange(168),1)
  for window,trim,holiday,version in itertools.product([365,1000],[False,True],[False,True],[1,2,3]):
   cfg=config(version,window,trim,holiday)
   week=(dow>=5)|(holiday&ish)
   category=(week.astype(int)[:,None]+np.zeros((len(idx),24),int) if version==1 else week.astype(int)[:,None]*24+np.arange(24) if version==2 else np.where(holiday&ish,6,dow)[:,None]*24+np.arange(24))
   ncat={1:2,2:48,3:168}[version]
   futurecat=(ordinary//24>=5).astype(int) if version==1 else (ordinary//24>=5).astype(int)*24+ordinary%24 if version==2 else ordinary
   sums=np.zeros((len(idx),ncat));counts=np.zeros_like(sums)
   if not trim:
    for row in range(len(idx)):
     good=np.isfinite(values[row]);np.add.at(sums[row],category[row,good],values[row,good]);np.add.at(counts[row],category[row,good],1)
    cs=np.vstack([np.zeros((1,ncat)),np.cumsum(sums,axis=0)]);cc=np.vstack([np.zeros((1,ncat)),np.cumsum(counts,axis=0)])
   for i in range(window,len(idx)):
    if trim:
     vv=values[i-window:i];cat=category[i-window:i];good=np.isfinite(vv)
     if not good.any():continue
     cutoff=np.quantile(vv[good],.99);good&=vv<=cutoff
     ss=np.bincount(cat[good],weights=vv[good],minlength=ncat);nn=np.bincount(cat[good],minlength=ncat)
    else:ss=cs[i]-cs[i-window];nn=cc[i]-cc[i-window]
    if (nn<20).any():continue
    means=ss/nn;shape=means[futurecat];den=shape.sum()
    if not den>0:continue
    w=7*shape/den
    result.append(dict(asset=asset,date=idx[i],config=cfg,version=version,window=window,trim=trim,holiday=holiday,lag=0,training_start=idx[i-window],training_end=idx[i]-pd.Timedelta(days=1),training_hours=int(nn.sum()),hourly_level=den/168,training_weight_sum=float(nn@w) if version==3 else np.nan,**{f'w{j}':w[j] for j in range(168)}))
   log(f'时钟权重 {asset}/{cfg} 完成')
 base=[r for r in result if r['config']==PRIMARY]
 for r in base:
  shifted=dict(r);shifted['date']=r['date']+pd.Timedelta(days=30);shifted['lag']=30;shifted['config']=config(lag=30)
  if shifted['date']<=d.date.max()+pd.Timedelta(days=1):result.append(shifted)
 frame=pd.DataFrame(result);frame=event_profiles(frame,d);frame.to_parquet(OUT/'clock_profiles.parquet',index=False)
 csv(frame.groupby(['asset','config']).agg(n=('date','size'),first=('date','min'),last=('date','max'),training_hours_min=('training_hours','min')).reset_index(),'clock_profile_coverage.csv')
 return frame
def event_profiles(frame,d):
 events=pd.read_csv(OUT/'scheduled_events.csv',parse_dates=['timestamp','publication'])
 if not len(events):return frame
 extra=[];hs=set(holidays());wc=[f'w{j}' for j in range(168)]
 for (asset,cfg),base in frame[(frame.version==3)&(frame.lag==0)].groupby(['asset','config']):
  source=d[(d.asset==asset)&d.valid_day].set_index('date');values=source[[f'hour_{i}' for i in range(24)]]
  event_rv=np.array([values.loc[t.normalize(),f'hour_{t.hour}'] if t.normalize() in values.index else np.nan for t in events.timestamp]);times=events.event_ms.to_numpy();pub=events.publication_ms.to_numpy();out=[]
  for r in base.itertuples(index=False):
   origin=r.date.value//10**6;start=r.training_start.value//10**6;ok=(times>=start)&(times<origin)&(pub<origin)&np.isfinite(event_rv)
   if r.trim:
    vv=values.loc[(values.index>=r.training_start)&(values.index<r.date)].to_numpy();cut=np.nanquantile(vv,.99);ok&=event_rv<=cut
   ii=np.flatnonzero(ok)
   if len(ii)<3:continue
   w=np.array([getattr(r,c) for c in wc]);slot=np.array([(6 if r.holiday and events.timestamp.iloc[j].normalize() in hs else events.timestamp.iloc[j].dayofweek)*24+events.timestamp.iloc[j].hour for j in ii]);units=w[slot]
   expected=(24*r.hourly_level*units).sum()
   if not expected>0:continue
   multiplier=event_rv[ii].sum()/expected;normalizer=1+(multiplier-1)*units.sum()/r.training_weight_sum
   row=r._asdict();row.update(config=cfg.replace('v3_','v4_',1),version=4,event_multiplier=multiplier,event_normalizer=normalizer,event_training_n=len(ii));out.append(row)
  extra.extend(out);log(f'事前FOMC事件时钟 {asset}/{cfg}：{len(out)} 日')
 return pd.concat([frame,pd.DataFrame(extra)],ignore_index=True)
class Clock:
 def __init__(self,path=None):
  self.frame=pd.read_parquet(path or OUT/'clock_profiles.parquet');self.lookup={};self.event_lookup={}
  for (asset,cfg),g in self.frame.groupby(['asset','config']):
   g=g.sort_values('date');self.lookup[asset,cfg]=(g.date.to_numpy('datetime64[D]').astype('int64'),g[[f'w{j}' for j in range(168)]].to_numpy(float))
   if cfg.startswith('v4_'):self.event_lookup[asset,cfg]=(g.event_multiplier.to_numpy(float),g.event_normalizer.to_numpy(float))
  self.hs=holidays().to_numpy('datetime64[D]').astype('int64')
  ep=OUT/'scheduled_events.csv';self.events=pd.read_csv(ep) if ep.exists() else pd.DataFrame()
 def weights(self,asset,cfg,origins):
  origins=np.asarray(origins,dtype='datetime64[D]').astype('int64');n=len(origins)
  if cfg=='v0':return np.full((n,168),1/24),np.ones(n,bool)
  item=self.lookup.get((asset,cfg))
  if item is None:return np.full((n,168),np.nan),np.zeros(n,bool)
  dates,w=item;pos=np.searchsorted(dates,origins);valid=pos<len(dates);safe=np.minimum(pos,len(dates)-1);valid&=dates[safe]==origins
  result=w[safe].copy();result[~valid]=np.nan;return result,valid
 def integral(self,asset,cfg,origins,start,end):
  start=np.asarray(start,float);end=np.asarray(end,float);origins=np.asarray(origins,dtype='datetime64[D]');w,good=self.weights(asset,cfg,origins)
  # Unix epoch Thursday; shift to Monday 1969-12-29 to define weekly slots.
  def primitive(t):
   hour=t/3600000+72;whole=np.floor(hour).astype('int64');slot=whole%168;weeks=whole//168;frac=hour-whole
   c=np.cumsum(w,axis=1);prev=np.column_stack([np.zeros(len(w)),c[:,:-1]])
   return weeks*w.sum(1)+prev[np.arange(len(w)),slot]+frac*w[np.arange(len(w)),slot]
  answer=primitive(end)-primitive(start)
  if '_hol1_' in cfg:
   # On a federal holiday replace that weekday's shape by Sunday's shape.
   lo=np.minimum(start,end);hi=np.maximum(start,end);sign=np.sign(end-start)
   for hd in self.hs:
    hstart=hd*86400000.;overlap=(hi>hstart)&(lo<hstart+86400000)
    ii=np.flatnonzero(overlap)
    if not len(ii):continue
    day=int((hd+3)%7)
    for hour in range(24):
     duration=np.maximum(0,np.minimum(hi[ii],hstart+(hour+1)*3600000)-np.maximum(lo[ii],hstart+hour*3600000))/3600000*sign[ii]
     answer[ii]+=duration*(w[ii,6*24+hour]-w[ii,day*24+hour])
  if cfg.startswith('v4_') and (asset,cfg) in self.event_lookup:
   dates,_=self.lookup[asset,cfg];pos=np.searchsorted(dates,origins.astype('int64'));pos=np.minimum(pos,len(dates)-1);mult,norm=self.event_lookup[asset,cfg];mult=mult[pos];norm=norm[pos];origin_ms=origins.astype('int64')*86400000.
   for e in self.events.itertuples(index=False):
    lo=np.minimum(start,end);hi=np.maximum(start,end);duration=np.maximum(0,np.minimum(hi,e.event_ms+3600000)-np.maximum(lo,e.event_ms))/3600000;duration*=np.sign(end-start)*(e.publication_ms<origin_ms)
    day=e.event_ms//86400000;hour=e.event_ms//3600000%24;slot=int((day+3)%7)*24+int(hour)
    if '_hol1_' in cfg and day in self.hs:slot=144+int(hour)
    answer+=duration*w[:,slot]*(mult-1)
   answer/=norm
  answer[~good]=np.nan;return answer
def facts(d,a):
 good=a[a.valid].copy();good['year']=good.date.dt.year
 shape=good.groupby(['asset','year','slot','dow','hour']).rv.agg(['mean','count']).reset_index();csv(shape,'clock_shape_year.csv')
 csv(good.groupby(['asset','slot','dow','hour']).rv.agg(['mean','count']).reset_index(),'clock_shape.csv')
 rows=[];session=[]
 import statsmodels.api as sm
 for asset,g in d.groupby('asset'):
  g=g[g.valid_day].sort_values('date');g['year']=g.date.dt.year
  for period,z in [('all',g)]+[(str(y),z) for y,z in g.groupby('year')]+[('etf_pre',g[g.date<'2024-01-11']),('etf_post',g[g.date>='2024-01-11'])]:
   weekend=z.date.dt.dayofweek>=5;mw=z.loc[weekend,'rv5'].mean();md=z.loc[~weekend,'rv5'].mean()
   X=np.column_stack([np.ones(len(z)),weekend.astype(float)]);fit=sm.OLS(z.rv5,X).fit(cov_type='HAC',cov_kwds={'maxlags':7})
   rows.append(dict(asset=asset,period=period,n=len(z),weekend_n=int(weekend.sum()),weekday_n=int((~weekend).sum()),weekend_mean=mw,weekday_mean=md,weekend_ratio=mw/md,weekend_coefficient=fit.params.iloc[1],weekend_p=fit.pvalues.iloc[1]))
   total=z[['asia','europe','america']].sum().sum()
   session.append(dict(asset=asset,period=period,n=len(z),asia_share=z.asia.sum()/total,europe_share=z.europe.sum()/total,america_share=z.america.sum()/total))
 csv(rows,'weekend_facts.csv');csv(session,'session_facts.csv')
 t=pd.to_datetime(good.datetime,utc=True).dt.tz_convert('America/New_York');h=t.dt.hour;dow=t.dt.dayofweek
 good['legacy_closed']=((dow==4)&(h>=17))|(dow==5)|((dow==6)&(h<18))
 # CME officially migrated to 24/7 on 2026-05-29; full-weekend closure facts
 # are therefore separately reported for the historical rule and pre-change.
 rows=[]
 for asset,g in good.groupby('asset'):
  for name,z in [('fixed_legacy_window_all',g),('actual_full_weekend_closure_pre_2026_05_29',g[g.datetime<'2026-05-29 21:00:00'])]:
   cl=z.legacy_closed;rows.append(dict(asset=asset,definition=name,n=len(z),closed_hours=int(cl.sum()),closed_mean=z.loc[cl,'rv'].mean(),other_mean=z.loc[~cl,'rv'].mean(),ratio=z.loc[cl,'rv'].mean()/z.loc[~cl,'rv'].mean()))
 csv(rows,'cme_closure_facts.csv')
 window=[]
 for asset,g in d[d.valid_day].groupby('asset'):
  for field,hours in [('funding_window',3.),('expiry_window',2.)]:
   window.append(dict(asset=asset,window=field,n=len(g),share=g[field].sum()/g.rv5.sum(),hours=hours,intensity_ratio=(g[field].sum()/hours)/((g.rv5.sum()-g[field].sum())/(24-hours))))
 csv(window,'settlement_expiry_facts.csv')
 contrasts=[]
 for asset,g in good.groupby('asset'):
  z=g.copy();z['post']=(z.date>='2024-01-11').astype(float);z['america']=(z.hour>=16).astype(float);z['weekend']=(z.dow>=5).astype(float)
  for name,x in [('america',z.america),('weekend',z.weekend)]:
   X=np.column_stack([np.ones(len(z)),x,z.post,x*z.post]);fit=sm.OLS(z.rv,X).fit(cov_type='cluster',cov_kwds={'groups':z.date.astype('int64')})
   contrasts.append(dict(asset=asset,test=name,n=len(z),post_interaction=fit.params.iloc[3],p=fit.pvalues.iloc[3],comparison='BTC_ETH_against_other_assets_descriptive'))
 csv(contrasts,'etf_clock_contrasts.csv')
 z=d[d.valid_day].copy();z['post']=(z.date>='2024-01-11').astype(float);z['major']=z.asset.isin(['BTC','ETH']).astype(float);z['weekend']=(z.date.dt.dayofweek>=5).astype(float);fixed=pd.get_dummies(z.asset,drop_first=True,dtype=float);rows=[]
 for label,y,columns in [('america_share',z.america/z.rv5,['post','post_major']),('weekend_discount_log_rv',np.log(z.rv5),['post','weekend','post_major','post_weekend','weekend_major','triple'])]:
  extra=pd.DataFrame({'post':z.post,'weekend':z.weekend,'post_major':z.post*z.major,'post_weekend':z.post*z.weekend,'weekend_major':z.weekend*z.major,'triple':z.post*z.weekend*z.major},index=z.index);X=sm.add_constant(pd.concat([fixed,extra[columns]],axis=1));fit=sm.OLS(y,X).fit(cov_type='cluster',cov_kwds={'groups':z.date.astype('int64')})
  for col in columns:rows.append(dict(outcome=label,term=col,coefficient=fit.params[col],se=fit.bse[col],p=fit.pvalues[col],n=len(z),dates=z.date.nunique(),comparison='BTC_ETH_minus_other_eight_assets_descriptive'))
 csv(rows,'etf_group_comparison.csv')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--offline',action='store_true');ap.add_argument('--facts-only',action='store_true');args=ap.parse_args()
 source_freeze();official_calendars(args.offline);d,a=prepare();facts(d,a)
 if not args.facts_only:profiles(d)
 log('时钟事实和事前权重构造完成。')
if __name__=='__main__':main()
