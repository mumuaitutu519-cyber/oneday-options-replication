from r4_common import *
from sklearn.ensemble import HistGradientBoostingRegressor
from statsmodels.stats.multitest import multipletests
HIST=['HAR','HAR_S','HAR_J','HAR_Q','HAR_clock','HAR_DVOL','GB']
MODELS=HIST+['DVOL_raw','DVOL_calibrated','IV_raw','IV_calibrated','IV_HAR','IV_GB']
PARAMS=[dict(max_iter=n,max_leaf_nodes=k) for n in [40,80] for k in [7,15]]

def features(currency,hour):
 a=grid(currency);idx=pd.date_range(a.index.min().floor('D'),a.index.max().floor('D'));when=idx+pd.Timedelta(hours=hour+1);r=a.logret;v=a.rv
 def sample(s):return s.reindex(when).to_numpy()
 d=pd.DataFrame({'date':idx,'observed_at':when})
 for n,name in [(12,'rv1'),(96,'rv8'),(288,'rv24'),(2016,'rv7'),(8640,'rv30'),(2016,'rv168')]:d[name]=sample(v.rolling(n,min_periods=n).sum())/(n/288 if name in ['rv7','rv30'] else 1)
 d['semi_up']=sample(v.where(r>=0,0).rolling(288,min_periods=288).sum());d['semi_down']=sample(v.where(r<0,0).rolling(288,min_periods=288).sum());bp=(np.pi/2)*r.abs()*r.shift(1).abs();d['jump']=np.maximum(d.rv24-sample(bp.rolling(288,min_periods=288).sum()),0);rq=(288/3)*r**4;d['quarticity']=sample(rq.rolling(288,min_periods=288).sum());d['harq']=d.rv24*np.sqrt(d.quarticity.clip(lower=EPS));d['weekend']=(idx.dayofweek>=5).astype(float);d['prior_return']=sample(r.rolling(288,min_periods=288).sum())
 dv=pd.read_parquet(ROOT/f'data_clean/dvol_hourly_{currency}.parquet').sort_values('public_at');dt=pd.to_datetime(dv.public_at).dt.tz_localize(None).to_numpy();pos=np.searchsorted(dt,when.to_numpy(),side='right')-1;good=(pos>=0);safe=np.maximum(pos,0);good&=(when.to_numpy()-dt[safe])<=np.timedelta64(1,'h');d['dvol']=np.where(good,(dv.close.to_numpy()[safe]/100)**2,np.nan);d['dvol_public_at']=pd.to_datetime(np.where(good,dt[safe],np.datetime64('NaT')))
 clock=Clock(assets=[currency]);starts=when.view('int64')/1e6;end=(idx+pd.Timedelta(days=1,hours=8)).view('int64')/1e6;d['clock_future']=clock.integral(currency,'v3_w1000_trim0_hol0_lag0',idx.to_numpy('datetime64[D]'),starts,end)
 for n,col in [(1,'rv1'),(8,'rv8'),(24,'rv24')]:
  share=clock.integral(currency,'v3_w1000_trim0_hol0_lag0',idx.to_numpy('datetime64[D]'),starts-n*3600000,starts);d[col+'_clock']=d[col]/share
 return d

def target_panel(currency,hour,h):
 d=features(currency,hour);a=grid(currency);q=pd.read_parquet(OUT/f'chen/expiry_quotes_{currency}.parquet');q=q[q.window_hour==hour].copy();q['contract_day']=q.date+pd.Timedelta(days=1,hours=8);q=q[q.expiry==q.contract_day].set_index('date');d['IV_raw']=q.atm_w.reindex(d.date).to_numpy();d['last_option_trade']=q.last_trade.reindex(d.date).to_numpy();d['label_end']=d.date+pd.Timedelta(days=h,hours=8);d['y']=[rv_interval(a,w,e) if e<CUTOFF else np.nan for w,e in zip(d.observed_at,d.label_end)];d['IV_raw']*= (h-(hour+1-8)/24)/(1-(hour+1-8)/24);d['DVOL_raw']=d.dvol*(h-(hour+1-8)/24)/365
 return d

def matrices(d,form):
 cols=['rv24','rv7','rv30','rv1','rv8'];base=d[cols].copy()
 if form=='log':base=np.log(base.clip(lower=EPS))
 sets={'HAR':base}
 for m,extras in [('HAR_S',['semi_up','semi_down']),('HAR_J',['jump']),('HAR_Q',['quarticity','harq']),('HAR_clock',['clock_future','rv1_clock','rv8_clock','rv24_clock','weekend']),('HAR_DVOL',['dvol'])]:
  e=d[extras].copy()
  if form=='log':
   for col in extras:
    if col!='weekend':e[col]=np.log(e[col].clip(lower=EPS))
  sets[m]=pd.concat([base,e],axis=1)
 allcols=list(dict.fromkeys(sum([list(x.columns) for x in sets.values()],[])));full=pd.concat([base,d[[x for x in allcols if x not in base]]],axis=1)
 if form=='log':
  for col in full:
   if col not in base and col!='weekend':full[col]=np.log(full[col].clip(lower=EPS))
 sets['GB']=full;sets['IV_calibrated']=np.log(d[['IV_raw']].clip(lower=EPS)) if form=='log' else d[['IV_raw']];sets['DVOL_calibrated']=np.log(d[['DVOL_raw']].clip(lower=EPS)) if form=='log' else d[['DVOL_raw']]
 sets['IV_HAR']=pd.concat([base,sets['IV_calibrated']],axis=1)
 return sets

def ols_predict(X,y,tr,t,form):
 xx=X[tr];xt=X[t];mu=xx.mean(0);scale=xx.std(0);scale[scale<1e-15]=1;xx=np.column_stack([np.ones(len(tr)),(xx-mu)/scale]);xt=np.r_[1.,(xt-mu)/scale];target=np.log(np.maximum(y[tr],EPS)) if form=='log' else y[tr];beta=np.linalg.lstsq(xx,target,rcond=1e-10)[0];p=xt@beta
 if form=='log':p=np.exp(np.clip(p,-50,30))*np.mean(np.exp(np.clip(target-xx@beta,-50,30)))
 return max(float(p),EPS)

def run_spec(currency,hour,h,window,form):
 spec=f'w{window}_{form}_hour{hour}_h{h}';dest=OUT/f'chen/predictions_{currency}_{spec}.parquet'
 if not FORCE and dest.exists():return
 d=target_panel(currency,hour,h);sets=matrices(d,form);y=d.y.to_numpy();end=d.label_end.to_numpy();origin=d.observed_at.to_numpy();pred={m:np.full(len(d),np.nan) for m in MODELS};pred['IV_raw']=d.IV_raw.to_numpy().copy();pred['DVOL_raw']=d.DVOL_raw.to_numpy().copy();audits=[];tunes=[];selected=None;lastmonth=None
 for m,Xdf in sets.items():
  X=Xdf.to_numpy(float);ok=np.isfinite(X).all(1)&np.isfinite(y)&(y>0)
  for t in range(1000,len(d)):
   if not np.isfinite(X[t]).all():continue
   cut=np.searchsorted(end,origin[t],side='right');begin=0 if window=='expanding' else max(0,t-int(window));tr=np.flatnonzero(ok[begin:cut])+begin
   if len(tr)<200:continue
   if m=='GB':
    month=d.date.iloc[t].strftime('%Y-%m')
    if selected is None or month!=lastmonth:
     nval=max(30,int(len(tr)*.2));it=tr[:-nval];iv=tr[-nval:];it=it[end[it]<=origin[iv[0]]]
     if len(it)<100:continue
     scores=[];ty=np.log(np.maximum(y[it],EPS)) if form=='log' else y[it]
     for param in PARAMS:
      model=HistGradientBoostingRegressor(**param,learning_rate=.08,min_samples_leaf=30,l2_regularization=1,max_bins=64,early_stopping=False,random_state=SEED).fit(X[it],ty);v=model.predict(X[iv])
      if form=='log':v=np.exp(np.clip(v,-50,30))*np.mean(np.exp(np.clip(ty-model.predict(X[it]),-50,30)))
      scores.append(float(np.mean(qlike(y[iv],np.maximum(v,EPS)))))
     selected=PARAMS[int(np.argmin(scores))];lastmonth=month;tunes.append(dict(currency=currency,spec=spec,date=d.date.iloc[t],selected=json.dumps(selected),validation_start=d.observed_at.iloc[iv[0]],validation_last_label=d.label_end.iloc[iv[-1]],train_last_label=d.label_end.iloc[it[-1]],scores=json.dumps(scores)))
    ty=np.log(np.maximum(y[tr],EPS)) if form=='log' else y[tr];model=HistGradientBoostingRegressor(**selected,learning_rate=.08,min_samples_leaf=30,l2_regularization=1,max_bins=64,early_stopping=False,random_state=SEED).fit(X[tr],ty);value=float(model.predict(X[[t]])[0])
    if form=='log':value=np.exp(np.clip(value,-50,30))*np.mean(np.exp(np.clip(ty-model.predict(X[tr]),-50,30)))
    pred[m][t]=max(value,EPS)
   else:pred[m][t]=ols_predict(X,y,tr,t,form)
   audits.append(dict(currency=currency,spec=spec,model=m,date=d.date.iloc[t],origin=d.observed_at.iloc[t],train_n=len(tr),last_training_label=d.label_end.iloc[tr[-1]],last_training_origin=d.observed_at.iloc[tr[-1]],feature_cutoff=d.observed_at.iloc[t],dvol_public_at=d.dvol_public_at.iloc[t]))
  log(f'辰 预测 {currency}/{spec}/{m} 有效{np.isfinite(pred[m]).sum()}日')
 # Combine contemporaneous IV and previously issued out-of-sample GB forecasts.
 X=np.column_stack([pred['IV_raw'],pred['GB']]);X=np.log(np.maximum(X,EPS)) if form=='log' else X;ok=np.isfinite(X).all(1)&np.isfinite(y)&(y>0)
 for t in range(1000,len(d)):
  if not np.isfinite(X[t]).all():continue
  cut=np.searchsorted(end,origin[t],side='right');begin=0 if window=='expanding' else max(0,t-int(window));tr=np.flatnonzero(ok[begin:cut])+begin
  if len(tr)<200:continue
  pred['IV_GB'][t]=ols_predict(X,y,tr,t,form);audits.append(dict(currency=currency,spec=spec,model='IV_GB',date=d.date.iloc[t],origin=d.observed_at.iloc[t],train_n=len(tr),last_training_label=d.label_end.iloc[tr[-1]],last_training_origin=d.observed_at.iloc[tr[-1]],feature_cutoff=d.observed_at.iloc[t],dvol_public_at=d.dvol_public_at.iloc[t]))
 for m in MODELS:d[m]=pred[m];d.loc[d.index<1000,m]=np.nan
 d.to_parquet(dest,index=False);csv(audits,f'chen/forecast_audit_{currency}_{spec}.csv');csv(tunes,f'chen/hyperparameters_{currency}_{spec}.csv')

def worker(currency):
 for w in [750,1000,1500,'expanding']:
  for form in ['log','level']:run_spec(currency,8,1,w,form)
 for hour in [9,12,16]:run_spec(currency,hour,1,1000,'log')
 for h in [2,3,7]:run_spec(currency,8,h,1000,'log')

def summarize():
 rows=[];enc=[];mcrows=[];bestrows=[];rob=[]
 for currency in ['BTC','ETH']:
  for path in sorted((OUT/'chen').glob(f'predictions_{currency}_*.parquet')):
   spec=path.stem.replace('predictions_'+currency+'_','');d=pd.read_parquet(path);h=int(spec.rsplit('_h',1)[1]);common=d.dropna(subset=['y']+MODELS).copy()
   periods=[('all',common),('trimRV1',common[common.y<=common.y.quantile(.99)])]+[(str(y),z) for y,z in common.groupby(common.date.dt.year)]
   for split,z in periods:
    losses={m:float(np.mean(qlike(z.y,z[m]))) for m in HIST} if len(z) else {m:np.inf for m in HIST};best=min(losses,key=losses.get)
    for base in ['HAR','HAR_DVOL','DVOL_raw','DVOL_calibrated',best]:
     for model in MODELS:
      if model==base:continue
      rr=loss_compare(z.y,z[base],z[model],h);rr['nested']=base=='HAR' and model in ['HAR_S','HAR_J','HAR_Q','HAR_clock','HAR_DVOL','IV_HAR']
      if not rr['nested']:rr['cw_t']=np.nan;rr['cw_p']=np.nan
      rows.append(dict(currency=currency,spec=spec,split=split,base=base,model=model,strongest_historical=best,**rr))
    if len(z)>=30:
     candidates=[]
     for model in ['IV_raw','IV_calibrated']:
      r=loss_compare(z.y,z[best],z[model],h);ps=[loss_compare(z.y,z[b],z[model],h)['qlike_dm_p'] for b in HIST];adjust=multipletests(ps,method='holm')[1][HIST.index(best)];candidates.append((r['qlike_gain'],model,r,adjust));rob.append(dict(currency=currency,spec=spec,split=split,model=model,base=best,**r,holm_p=adjust))
     _,model,r,adjust=max(candidates,key=lambda x:x[0]);bestrows.append(dict(currency=currency,spec=spec,split=split,best_IV=model,best_history=best,**r,holm_p=adjust))
   if len(common)>=30:
    best=min(HIST,key=lambda m:np.mean(qlike(common.y,common[m])))
    for model in ['IV_raw','IV_calibrated','IV_HAR','IV_GB']:
     X=common[[best,model]];X.columns=['history','IV'];fit=sm.OLS(common.y,sm.add_constant(X)).fit(cov_type='HAC',cov_kwds={'maxlags':max(7,2*h)})
     for v in fit.params.index:enc.append(dict(currency=currency,spec=spec,model=model,variable=v,n=len(common),coef=fit.params[v],se=fit.bse[v],p=fit.pvalues[v],ci_low=fit.conf_int().loc[v,0],ci_high=fit.conf_int().loc[v,1]))
    for loss in ['qlike','mse']:
     L=np.column_stack([qlike(common.y,common[m]) if loss=='qlike' else (common.y-common[m])**2 for m in MODELS]);active,history=mcs(L,h,reps=REPS)
     for j,m in enumerate(MODELS):mcrows.append(dict(currency=currency,spec=spec,loss=loss,model=m,n=len(common),mean_loss=float(L[:,j].mean()),in_mcs=j in active,confidence=.9,history=json.dumps(history)))
 csv(pd.DataFrame(rows).drop_duplicates(['currency','spec','split','base','model']),'chen/forecast_comparisons.csv');csv(bestrows,'chen/forecast_best.csv');csv(rob,'chen/forecast_robustness_cells.csv');csv(enc,'chen/encompassing.csv');csv(mcrows,'chen/model_confidence_set.csv')
 summary=[]
 for c,g in pd.DataFrame(rob).groupby('currency'):
  z=g[g.spec.str.endswith('_h1')];summary.append(dict(currency=c,denominator=len(z),positive=int((z.qlike_gain>0).sum()),significant_positive=int(((z.qlike_gain>0)&(z.qlike_dm_p<.05)).sum()),direction_share=(z.qlike_gain>0).mean(),significant_share=((z.qlike_gain>0)&(z.qlike_dm_p<.05)).mean()))
 csv(summary,'chen/forecast_robustness.csv');log('辰 全部预测比较、置信集与包络完成')
def main():
 from concurrent.futures import ProcessPoolExecutor
 with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(worker,['BTC','ETH']))
 summarize()
if __name__=='__main__':main()
