from r5_common import *
from r4_forecast import features as old_features,matrices as old_matrices,PARAMS
from r4_common import grid,rv_interval
from r3_clock import Clock
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LassoLars
from statsmodels.stats.multitest import multipletests
HIST=['HAR','HAR_S','HAR_J','HAR_Q','HAR_clock','HAR_DVOL','GB','LASSO','RF']
CORE=HIST+['DVOL_raw','IV_raw','IV_calibrated','IV_HAR']
MODELS=CORE+['DVOL_calibrated','IV_GB']

def features(c):
 p=CHEN/f'features_{c}.parquet'
 if p.exists() and not FORCE:return pd.read_parquet(p)
 d=old_features(c,8);clk=Clock(assets=[c]);origin=d.observed_at.astype('int64').to_numpy()/1e6;days=d.date.to_numpy('datetime64[D]');cfg='v3_w365_trim0_hol0_lag0'
 d['clock_future']=clk.integral(c,cfg,days,origin,origin+23*3600000)
 for n,col in [(1,'rv1'),(8,'rv8'),(24,'rv24')]:d[col+'_clock']=d[col]/clk.integral(c,cfg,days,origin-n*3600000,origin)
 a=grid(c);q=pd.read_parquet(R4/f'expiry_quotes_{c}.parquet');q=q[(q.window_hour==8)&(q.expiry==q.date+pd.Timedelta(days=1,hours=8))].set_index('date')
 d['IV_raw']=q.atm_w.reindex(d.date).to_numpy();d['last_option_trade']=q.last_trade.reindex(d.date).to_numpy();d['label_end']=d.date+pd.Timedelta(days=1,hours=8);d['y']=[rv_interval(a,w,e) if e<CUTOFF else np.nan for w,e in zip(d.observed_at,d.label_end)];d['DVOL_raw']=d.dvol*23/24/365
 start=a.close.reindex(d.observed_at).to_numpy();finish=a.close.reindex(d.label_end).to_numpy();d['return']=np.log(finish/start);d.to_parquet(p,index=False);return d

def fit_predict(X,y,tr,t,form,kind='OLS',param=None,extra=None):
 ty=np.log(np.maximum(y[tr],EPS)) if form=='log' else y[tr];xx=X[tr];xt=X[[t]];evalx=None if extra is None else X[extra]
 if kind in ['OLS','LASSO']:
  mu=xx.mean(0);scale=xx.std(0);scale[scale<1e-15]=1;xx=(xx-mu)/scale;xt=(xt-mu)/scale
  if evalx is not None:evalx=(evalx-mu)/scale
 if kind=='OLS':
  xx=np.column_stack([np.ones(len(tr)),xx]);xt=np.column_stack([np.ones(len(xt)),xt]);beta=np.linalg.lstsq(xx,ty,rcond=1e-10)[0];fitted=xx@beta;p=xt@beta;past=None if evalx is None else np.column_stack([np.ones(len(evalx)),evalx])@beta
 else:
  if kind=='GB':m=HistGradientBoostingRegressor(**param,learning_rate=.08,min_samples_leaf=30,l2_regularization=1,max_bins=64,early_stopping=False,random_state=SEED)
  elif kind=='RF':m=RandomForestRegressor(n_estimators=64,**param,max_features=.7,random_state=SEED,n_jobs=1)
  else:m=LassoLars(**param)
  m.fit(xx,ty);fitted=m.predict(xx);p=m.predict(xt);past=None if evalx is None else m.predict(evalx)
 if form=='log':
  smear=np.mean(np.exp(np.clip(ty-fitted,-50,30)));p=np.exp(np.clip(p,-50,30))*smear
  if past is not None:past=np.exp(np.clip(past,-50,30))*smear
 return np.maximum(p,EPS),None if past is None else np.maximum(past,EPS)

def tuning(X,y,tr,end,origin,form,kind):
 nv=max(30,int(len(tr)*.2));iv=tr[-nv:];it=tr[:-nv];it=it[end[it]<=origin[iv[0]]]
 if len(it)<100:return None,None
 params=PARAMS if kind=='GB' else ([dict(alpha=x) for x in [1e-4,1e-3,.01,.1]] if kind=='LASSO' else [dict(max_depth=x,min_samples_leaf=y) for x in [4,8] for y in [10,30]])
 scores=[]
 # Validation prediction uses one fit at its first origin, with the complete held-out block.
 for pp in params:
  _,f=fit_predict(X,y,it,iv[0],form,kind,pp,extra=iv);scores.append(float(np.mean(qlike(y[iv],f))))
 return params[int(np.argmin(scores))],dict(validation_start=pd.Timestamp(origin[iv[0]]),validation_last_label=pd.Timestamp(end[iv[-1]]),train_last_label=pd.Timestamp(end[it[-1]]),scores=json.dumps(scores))

def predict_panel(d,c,spec,window=500,form='log',models=None,risk=False,minimum=120):
 dest=CHEN/f'predictions_{c}_{spec}.parquet'
 if dest.exists() and not FORCE:
  cached=pd.read_parquet(dest)
  if len(cached)==len(d) and np.allclose(cached.y,d.y,equal_nan=True,rtol=1e-12,atol=1e-15):return cached
 models=models or MODELS;sets=old_matrices(d,form);sets['LASSO']=sets['GB'];sets['RF']=sets['GB'];y=d.y.to_numpy();end=d.label_end.to_numpy();origin=d.observed_at.to_numpy();n=len(d);start=365 if window=='expanding' else int(window)
 pred={m:np.full(n,np.nan) for m in models};aud=[];tunes=[];riskrows=[]
 for m in ['DVOL_raw','IV_raw']:
  if m in pred:pred[m]=d[m].to_numpy().copy();pred[m][:start]=np.nan
 # Shared historic residual dates have all information required by the five risk models.
 riskmodels=['HAR','GB','DVOL_raw','IV_calibrated','IV_HAR'];riskok=np.isfinite(y)&(y>0)
 if risk:
  riskok&=np.isfinite(d['return'])
  for m in riskmodels:
   if m=='DVOL_raw':riskok&=np.isfinite(d.DVOL_raw)
   else:riskok&=np.isfinite(sets[m].to_numpy()).all(1)
 for m in models:
  if m not in sets:continue
  X=sets[m].to_numpy(float);ok=np.isfinite(X).all(1)&np.isfinite(y)&(y>0);selected=None;lastmonth=None;kind=m if m in ['GB','RF','LASSO'] else 'OLS'
  for t in range(start,n):
   if not np.isfinite(X[t]).all():continue
   cut=min(t,np.searchsorted(end,origin[t],side='right'));begin=0 if window=='expanding' else max(0,t-int(window));tr=np.flatnonzero(ok[begin:cut])+begin
   if len(tr)<minimum:continue
   if kind!='OLS':
    month=d.date.iloc[t].strftime('%Y-%m')
    if selected is None or month!=lastmonth:
     selected,z=tuning(X,y,tr,end,origin,form,kind)
     if selected is None:continue
     lastmonth=month;tunes.append(dict(currency=c,spec=spec,variant=spec,model=m,date=d.date.iloc[t],selected=json.dumps(selected),**z))
   rr=None
   if risk and m in riskmodels:rr=np.flatnonzero(riskok[max(0,t-500):cut])+max(0,t-500)
   p,past=fit_predict(X,y,tr,t,form,kind,selected,extra=rr);pred[m][t]=p[0]
   aud.append(dict(currency=c,spec=spec,variant=spec,model=m,date=d.date.iloc[t],origin=d.observed_at.iloc[t],train_n=len(tr),last_training_label=d.label_end.iloc[tr[-1]],last_training_origin=d.observed_at.iloc[tr[-1]],feature_cutoff=d.observed_at.iloc[t],dvol_public_at=d.dvol_public_at.iloc[t]))
   if rr is not None and len(rr)>=120:
    residual=d['return'].to_numpy()[rr]/np.sqrt(past)
    riskrows.append(risk_quantiles(c,spec,m,d,t,residual,p[0],rr))
  log(f'R5 {c}/{spec}/{m}: {np.isfinite(pred[m]).sum()} forecasts')
 if 'IV_GB' in models:
  X=np.column_stack([pred['IV_raw'],pred['GB']]);X=np.log(np.maximum(X,EPS)) if form=='log' else X;ok=np.isfinite(X).all(1)&np.isfinite(y)&(y>0)
  for t in range(start,n):
   if not np.isfinite(X[t]).all():continue
   cut=min(t,np.searchsorted(end,origin[t],side='right'));begin=0 if window=='expanding' else max(0,t-int(window));tr=np.flatnonzero(ok[begin:cut])+begin
   if len(tr)<minimum:continue
   pred['IV_GB'][t]=fit_predict(X,y,tr,t,form)[0][0];aud.append(dict(currency=c,spec=spec,variant=spec,model='IV_GB',date=d.date.iloc[t],origin=d.observed_at.iloc[t],train_n=len(tr),last_training_label=d.label_end.iloc[tr[-1]],last_training_origin=d.observed_at.iloc[tr[-1]],feature_cutoff=d.observed_at.iloc[t],dvol_public_at=d.dvol_public_at.iloc[t]))
 if risk:
  for t in range(start,n):
   cut=min(t,np.searchsorted(end,origin[t],side='right'));rr=np.flatnonzero(riskok[max(0,t-500):cut])+max(0,t-500)
   if len(rr)>=120 and np.isfinite(pred['DVOL_raw'][t]):riskrows.append(risk_quantiles(c,spec,'DVOL_raw',d,t,d['return'].to_numpy()[rr]/np.sqrt(d.DVOL_raw.to_numpy()[rr]),pred['DVOL_raw'][t],rr))
 for m in models:d[m]=pred[m]
 d.to_parquet(dest,index=False);csv(aud,f'forecast_audit_{c}_{spec}.csv');csv(tunes,f'hyperparameters_{c}_{spec}.csv')
 if risk:csv(riskrows,f'risk_quantiles_{c}_{spec}.csv')
 return d

def risk_quantiles(c,spec,m,d,t,z,var,rr):
 row=dict(currency=c,spec=spec,variant=spec,model=m,date=d.date.iloc[t],origin=d.observed_at.iloc[t],label_end=d.label_end.iloc[t],r=d['return'].iloc[t],sigma=np.sqrt(var),residual_n=len(z),first_residual_label=d.label_end.iloc[rr[0]],last_residual_label=d.label_end.iloc[rr[-1]])
 for side in ['long','short']:
  loss=-z if side=='long' else z
  for alpha in [.95,.99]:
   q=np.quantile(loss,alpha);tail=loss[loss>=q];key=side+'_'+str(int(alpha*100));row[key+'_VaR']=np.sqrt(var)*q;row[key+'_ES']=np.sqrt(var)*tail.mean();row[key+'_tail_n']=len(tail)
 return row

def worker(c):
 for w in [500,365,'expanding']:
  predict_panel(features(c).copy(),c,f'w{w}_log_hour8_h1',w,'log',risk=w==500)

def summarize():
 rows=[];coverage=[];mc=[];enc=[];bestrows=[];rob=[];cal=[];paired=[]
 for c in ['BTC','ETH']:
  for w in [365,500,'expanding']:
   spec=f'w{w}_log_hour8_h1';d=pd.read_parquet(CHEN/f'predictions_{c}_{spec}.parquet');common=d.dropna(subset=['y']+CORE).copy()
   coverage.append(dict(currency=c,spec=spec,variant=spec,n=len(common),start=common.date.min(),end=common.date.max(),core_models=','.join(CORE),ivgb_pair_n=len(d.dropna(subset=['y','GB','IV_GB']))))
   for split,z in periods(common):
    losses={m:np.mean(qlike(z.y,z[m])) if len(z) else np.inf for m in HIST};best=min(losses,key=losses.get)
    for base in list(dict.fromkeys(['HAR','HAR_DVOL','DVOL_raw','GB',best])):
     for m in CORE:
      if m==base:continue
      rr=compare(z.y,z[base],z[m]);rr['nested']=base=='HAR' and m in ['HAR_S','HAR_J','HAR_Q','HAR_clock','HAR_DVOL','IV_HAR']
      if not rr['nested']:rr['cw_t']=np.nan;rr['cw_p']=np.nan
      rows.append(dict(currency=c,spec=spec,variant=spec,split=split,base=base,model=m,strongest_historical=best,**rr))
    for m in ['IV_raw','IV_calibrated','IV_HAR']:
     r=compare(z.y,z[best],z[m]);r['cw_t']=np.nan;r['cw_p']=np.nan;ps=[compare(z.y,z[b],z[m]).get('qlike_dm_p',np.nan) for b in HIST];holm=multipletests(np.nan_to_num(ps,nan=1),method='holm')[1][HIST.index(best)]
     rob.append(dict(currency=c,spec=spec,variant=spec,split=split,model=m,base=best,holm_p=holm,**r))
    if split=='all':
     candidates=[(compare(z.y,z[best],z[m]).get('qlike_gain',-np.inf),m) for m in ['IV_raw','IV_calibrated']];_,winner=max(candidates);br=compare(z.y,z[best],z[winner]);br['cw_t']=np.nan;br['cw_p']=np.nan
     ps=[compare(z.y,z[b],z[winner]).get('qlike_dm_p',np.nan) for b in HIST];hp=multipletests(np.nan_to_num(ps,nan=1),method='holm')[1][HIST.index(best)]
     bestrows.append(dict(currency=c,spec=spec,variant=spec,split=split,best_IV=winner,best_history=best,best_history_loss=losses[best],holm_p=hp,**br))
   own=d.dropna(subset=['y','GB','IV_GB']);rr=compare(own.y,own.GB,own.IV_GB);rr['cw_t']=np.nan;rr['cw_p']=np.nan
   rows.append(dict(currency=c,spec=spec,variant=spec,split='IVGB_own',base='GB',model='IV_GB',strongest_historical='',nested=False,**rr))
   if len(common)>=30:
    for loss in ['qlike','mse']:
     L=np.column_stack([qlike(common.y,common[m]) if loss=='qlike' else (common.y-common[m])**2 for m in CORE])
     for conf in [.90,.75]:
      active,history=mcs(L,1,alpha=1-conf,reps=1999)
      for j,m in enumerate(CORE):mc.append(dict(currency=c,spec=spec,variant=spec,loss=loss,model=m,n=len(common),mean_loss=L[:,j].mean(),in_mcs=j in active,confidence=conf,history=json.dumps(history)))
    best=min(HIST,key=lambda m:np.mean(qlike(common.y,common[m])))
    for m in ['IV_raw','IV_calibrated','IV_HAR','DVOL_raw','HAR','GB']:
     for form in ['level','log']:
      x=common[m].to_numpy();y=common.y.to_numpy()
      if form=='log':x=np.log(x);y=np.log(y)
      fit=sm.OLS(y,sm.add_constant(x)).fit(cov_type='HAC',cov_kwds={'maxlags':7});wald=fit.wald_test((np.eye(2),np.array([0,1.])),scalar=True)
      cal.append(dict(currency=c,spec=spec,variant=spec,model=m,form=form,n=len(y),intercept=fit.params[0],slope=fit.params[1],r_squared=fit.rsquared,intercept_p=fit.pvalues[0],slope_p=fit.pvalues[1],joint_wald=float(wald.statistic),joint_p=float(wald.pvalue)))
      X=common[[best,m]].copy();X.columns=['history','IV'];yy=common.y
      if form=='log':X=np.log(X);yy=np.log(yy)
      fit=sm.OLS(yy,sm.add_constant(X)).fit(cov_type='HAC',cov_kwds={'maxlags':7})
      joint=fit.wald_test(np.array([[0,1,0],[0,0,1]]),scalar=True)
      for v in fit.params.index:enc.append(dict(currency=c,spec=spec,variant=spec,model=m,form=form,variable=v,n=len(common),joint_slopes_p=float(joint.pvalue),coef=fit.params[v],se=fit.bse[v],p=fit.pvalues[v],ci_low=fit.conf_int().loc[v,0],ci_high=fit.conf_int().loc[v,1]))
   if w==500:
    old=pd.read_parquet(R4/f'predictions_{c}_w1000_log_hour8_h1.parquet');oldmodels=['HAR','HAR_S','HAR_J','HAR_Q','HAR_clock','HAR_DVOL','GB','DVOL_raw','DVOL_calibrated','IV_raw','IV_calibrated','IV_HAR','IV_GB'];oc=old.dropna(subset=['y']+oldmodels);pc=common.merge(oc[['date']+oldmodels],on='date',suffixes=('','_R4'))
    for m in ['IV_calibrated','IV_HAR']:
     for base in ['HAR','GB','DVOL_raw']:
      for variant,z,ss in [('R4_original',oc,'_R4'),('R5_on_R4_dates',pc,''),('R4_on_paired_dates',pc,'_R4')]:
       a=z[base] if variant=='R4_original' else z[base+ss];b=z[m] if variant=='R4_original' else z[m+ss];paired.append(dict(currency=c,variant=variant,model=m,base=base,start=z.date.min(),end=z.date.max(),**compare(z.y,a,b)))
 csv(rows,'forecast_comparisons.csv');csv(coverage,'forecast_coverage.csv');csv(mc,'model_confidence_set.csv');csv(enc,'encompassing.csv');csv(cal,'unbiasedness.csv');csv(bestrows,'forecast_best.csv');csv(rob,'forecast_robustness_cells.csv');csv(paired,'forecast_R4_paired.csv')
 log('R5 extended forecasts summarized')

def main():
 from concurrent.futures import ProcessPoolExecutor
 with ProcessPoolExecutor(max_workers=2) as p:list(p.map(worker,['BTC','ETH']))
 summarize()
if __name__=='__main__':main()
