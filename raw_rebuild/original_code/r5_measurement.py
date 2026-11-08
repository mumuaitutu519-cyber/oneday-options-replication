from r5_common import *
from r5_forecast import features,predict_panel,HIST
from r4_common import grid
import gzip,re

def prices(c,market):
 p=CHEN/f'prices_{market}_{c}.parquet'
 if p.exists() and (market!='deribit' or (CHEN/f'deribit_native_{c}.json').exists()):return pd.read_parquet(p).close
 if market=='deribit':
  parts=[];receipts=[]
  for source in sorted((ROOT/'out/R2/raw/hedge').glob(f'perpetual5_{c}_*.json.gz')):
   match=re.fullmatch(r'perpetual5_[A-Z]+_(\d{4}-\d{2})_(\d+)\.json\.gz',source.name)
   if not match:continue
   month=pd.Timestamp(match[1]+'-01');begin=month.value//10**6+int(match[2])*4000*300000;finish=min(month+pd.offsets.MonthBegin(1),CUTOFF).value//10**6;end=min(begin+4000*300000,finish)
   with gzip.open(source,'rt') as f:raw=json.load(f)
   result=raw.get('result',{});d=pd.DataFrame({x:result.get(x,[]) for x in ['ticks','close','volume']});d=d[(d.ticks>=begin)&(d.ticks<end)&(d.close>0)&np.isfinite(d.close)]
   parts.append(d);receipts.append(dict(currency=c,path=str(source.relative_to(ROOT)),sha256=sha(source),returned=len(d),zero_volume=int(d.volume.eq(0).sum())))
  d=pd.concat(parts,ignore_index=True).sort_values('ticks');assert not d.ticks.duplicated().any(),(c,'duplicate native chart timestamps')
  idx=pd.to_datetime(d.ticks,unit='ms')+pd.Timedelta(minutes=5);s=pd.Series(d.close.to_numpy(),index=idx);freq='5min';csv(receipts,f'deribit_native_sources_{c}.csv');(CHEN/f'deribit_native_{c}.json').write_text(json.dumps(dict(currency=c,returned_bars=len(d),zero_volume_bars=int(d.volume.eq(0).sum()),source='existing_raw_chart_responses',native_minutes=5)))
  log(f'R5 {c} Deribit raw archive: {len(d)} returned bars, {int(d.volume.eq(0).sum())} zero-volume bars with real positive close')
 else:
  folder=ROOT/f'data_clean/minute/{"futures/um" if market=="binance_perp" else "spot"}/{c}';parts=[]
  for f in sorted(folder.glob('*.parquet')):
   if f.name.startswith('._') or f.name[:7]<'2019-12':continue
   parts.append(pd.read_parquet(f,columns=['timestamp','close']))
  d=pd.concat(parts,ignore_index=True);d=d[(pd.to_datetime(d.timestamp)>='2019-12-01')&(pd.to_datetime(d.timestamp)<CUTOFF)];idx=pd.to_datetime(d.timestamp)+pd.Timedelta(minutes=1);s=pd.Series(d.close.to_numpy(),index=idx);freq='1min'
 s=s.sort_index();duplicates=s.index.duplicated(keep=False)
 if duplicates.any():
  assert s[duplicates].groupby(level=0).nunique().max()==1,(c,market,'conflicting duplicate');s=s[~s.index.duplicated()]
 s=s.reindex(pd.date_range(s.index.min(),s.index.max(),freq=freq));pd.DataFrame({'close':s}).to_parquet(p);return s

class Measure:
 def __init__(self,s,measure,step):self.s=s;self.p=np.log(s.to_numpy(float));self.first=s.index[0].value;self.dt=int(step*60*1e9);self.kind=measure;self.step=step
 def path(self,start,end,stride=1,phase=0):
  i=(pd.Timestamp(start).value-self.first)//self.dt;j=(pd.Timestamp(end).value-self.first)//self.dt
  if i<0 or j>=len(self.p) or i>=j:return None
  idx=np.arange(i+phase,j+1,stride,dtype=int);idx=np.unique(np.r_[i,idx,j]);p=self.p[idx]
  if np.isnan(p).any():return None
  return p
 def calc(self,start,end,extras=False):
  stride=1 if self.kind in ['rv1','kernel'] else int((15 if self.kind=='rv15' else 5)/self.step)
  if self.kind=='subsample':paths=[self.path(start,end,5,j) for j in range(5)]
  else:paths=[self.path(start,end,stride)]
  if any(x is None for x in paths):return (np.nan,{}) if extras else np.nan
  vals=[];aug=[]
  for p in paths:
   r=np.diff(p);n=len(r);rv=r@r
   if self.kind=='kernel':
    # Endpoint jitter in price levels, before log returns.
    p=p.copy();k=min(5,max(1,len(p)//3));p[0]=special.logsumexp(p[:k])-np.log(k);p[-1]=special.logsumexp(p[-k:])-np.log(k);r=np.diff(p);g1=r[1:]@r[:-1];iq=n/3*np.sum(r**4);omega=max(-g1/n,0);H=int(np.clip(np.ceil(3.5134*(omega/max(np.sqrt(iq),EPS))**.4*n**.6),1,min(60,n-1)));x=np.arange(1,H+1)/(H+1);weights=np.where(x<=.5,1-6*x*x+6*x**3,2*(1-x)**3);rv=r@r+2*sum(w*(r[j:]@r[:-j]) for j,w in enumerate(weights,1))
   vals.append(rv)
   if extras:aug.append(dict(semi_up=np.sum(r[r>=0]**2),semi_down=np.sum(r[r<0]**2),jump=max(np.sum(r*r)-(np.pi/2)*np.sum(abs(r[1:]*r[:-1])),0),quarticity=n/3*np.sum(r**4)))
  v=float(np.mean(vals));e={k:float(np.mean([x[k] for x in aug])) for k in aug[0]} if aug else {};return (v,e) if extras else v

def build(c,market,kind):
 variant=f'{market}_{kind}';dest=CHEN/f'measurement_panel_{c}_{variant}.parquet'
 if dest.exists() and not FORCE and not (market=='deribit' and 'price_origin' not in pd.read_parquet(dest).columns):
  cached=pd.read_parquet(dest)
  if variant=='binance_perp_rv5':
   original=features(c)
   for col in ['y','rv1','rv8','rv24','rv7','rv30','semi_up','semi_down','jump','quarticity','harq','return','rv1_clock','rv8_clock','rv24_clock']:cached[col]=original[col]
   cached['rv_ratio']=cached.y/cached.baseline_y;cached.to_parquet(dest,index=False)
  return cached
 base=features(c).copy();s=prices(c,market);step=5 if market=='deribit' else 1
 if market=='deribit' and kind in ['rv1','subsample']:return None
 m=Measure(s,kind,step);delivery=pd.read_parquet(ROOT/f'out/R3/shared/delivery_{c}.parquet').set_index('expiration_timestamp').delivery_price
 for col in ['y','rv1','rv8','rv24','rv7','rv30','semi_up','semi_down','jump','quarticity','harq','return']:base[col]=np.nan
 for i,row in base.iterrows():
  origin=row.observed_at;end=row.label_end
  for hours,col in [(1,'rv1'),(8,'rv8')]:base.loc[i,col]=m.calc(origin-pd.Timedelta(hours=hours),origin)
  v,e=m.calc(origin-pd.Timedelta(days=1),origin,True);base.loc[i,'rv24']=v
  for col,value in e.items():base.loc[i,col]=value
  if end<CUTOFF:
   base.loc[i,'y']=m.calc(origin,end);p0=s.get(origin,np.nan);p1=s.get(end,np.nan);base.loc[i,'return']=np.log(p1/p0) if p0>0 and p1>0 else np.nan
  if kind=='payoff':
   finish=end-pd.Timedelta(minutes=30);p=s.get(finish,np.nan);D=delivery.get(int(end.value/1e6),np.nan);rv=m.calc(origin,finish) if end<CUTOFF else np.nan;base.loc[i,'y']=rv+np.log(D/p)**2 if np.isfinite(rv) and D>0 and p>0 else np.nan
  if (i+1)%600==0:log(f'R5 RV measure {c}/{variant}: {i+1} days')
 base['rv7']=base.rv24.rolling(7,min_periods=7).mean();base['rv30']=base.rv24.rolling(30,min_periods=30).mean();base['harq']=base.rv24*np.sqrt(base.quarticity.clip(lower=EPS))
 # Existing pre-origin clock weights transform the newly measured realized features.
 original=features(c)
 for n,col in [(1,'rv1'),(8,'rv8'),(24,'rv24')]:base[col+'_clock']=base[col]*original[col+'_clock']/original[col]
 base['variant']=variant;base['baseline_y']=original.y
 if market=='deribit':base['price_origin']='raw_returned_chart_bars'
 if variant=='binance_perp_rv5':
  base['independent_rv5']=base.y
  for col in ['y','rv1','rv8','rv24','rv7','rv30','semi_up','semi_down','jump','quarticity','harq','return','rv1_clock','rv8_clock','rv24_clock']:base[col]=original[col]
 base['rv_ratio']=base.y/base.baseline_y;base.to_parquet(dest,index=False);return base

def worker(c):
 results=[];quality=[];availability=[]
 for market in ['binance_perp','binance_spot','deribit']:
  for kind in ['rv5','rv1','subsample','rv15','kernel']+(['payoff'] if market=='binance_perp' else []):
   variant=market+'_'+kind;d=build(c,market,kind)
   if d is None:availability.append(dict(currency=c,variant=variant,status='native_5min_no_1min_observations'));continue
   if variant=='binance_perp_rv5':
    # Independently computed grid is saved; primary forecasts use the existing common sample.
    p=pd.read_parquet(CHEN/f'predictions_{c}_w500_log_hour8_h1.parquet') if (CHEN/f'predictions_{c}_w500_log_hour8_h1.parquet').exists() else predict_panel(d,c,'rv_'+variant,models=HIST+['DVOL_raw','IV_raw','IV_calibrated','IV_HAR'])
   else:p=predict_panel(d.copy(),c,'rv_'+variant,models=HIST+['DVOL_raw','IV_raw','IV_calibrated','IV_HAR'])
   z=p.dropna(subset=['y']+HIST+['DVOL_raw','IV_calibrated','IV_HAR']);best=min(HIST,key=lambda m:np.mean(qlike(z.y,z[m]))) if len(z) else 'GB'
   for base,model,claim in [('HAR','IV_calibrated','甲'),('DVOL_raw','IV_calibrated','乙'),('GB','IV_HAR','丙'),(best,'IV_HAR','丙_best')]:results.append(dict(currency=c,spec='rv_'+variant,variant=variant,split='all',base=base,model=model,strongest_historical=best,claim=claim,nested=False,**{k:(np.nan if k in ['cw_t','cw_p'] else v) for k,v in compare(z.y,z[base],z[model]).items()}))
   v=d.dropna(subset=['y','IV_raw']);prem=v.IV_raw-v.y;annual=365/(23/24);vol=pd.Series(100*(np.sqrt(v.IV_raw.to_numpy()*annual)-np.sqrt(v.y.to_numpy()*annual)),index=pd.DatetimeIndex(v.date));quality.append(dict(currency=c,variant=variant,tenor=1,split='all',metric='volpts',**ci(vol),median=vol.median(),positive_share=(vol>0).mean(),rv_ratio_mean=v.rv_ratio.mean(),rv_ratio_median=v.rv_ratio.median(),rv_var_mean=prem.mean(),target_nonpositive=int((d.y<=0).sum()),coverage_start=v.date.min(),coverage_end=v.date.max(),**bootstrap(vol)));availability.append(dict(currency=c,variant=variant,status='estimated',n=len(z)))
   csv(pd.DataFrame(results).drop_duplicates(['currency','spec','variant','split','base','model']),f'measurement_forecasts_{c}.csv');csv(quality,f'measurement_premium_{c}.csv');csv(availability,f'measurement_availability_{c}.csv')
 log(f'R5 {c} RV variants finished')
if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--currency',choices=['BTC','ETH']);args=ap.parse_args()
 for c in [args.currency] if args.currency else ['BTC','ETH']:worker(c)
 if all((CHEN/f'measurement_forecasts_{c}.csv').exists() for c in ['BTC','ETH']):
  for stem in ['measurement_forecasts','measurement_premium','measurement_availability']:csv(pd.concat([pd.read_csv(CHEN/f'{stem}_{c}.csv') for c in ['BTC','ETH']],ignore_index=True),stem+'.csv')
