from model_engine import *
MODELS=HIST[:7]+['DVOL_raw','DVOL_calibrated','IV_raw','IV_calibrated','IV_HAR','IV_GB']
def run_spec(currency,hour,h,window,form):
 spec=f'w{window}_{form}_hour{hour}_h{h}';dest=WORK/f'predictions_{currency}_{spec}.parquet'
 if not FORCE and dest.exists():return
 d=load_input(f'{currency}_{spec}');sets=matrices(d,form);y=d.y.to_numpy();end=d.label_end.to_numpy();origin=d.observed_at.to_numpy();pred={m:np.full(len(d),np.nan) for m in MODELS};pred['IV_raw']=d.IV_raw.to_numpy().copy();pred['DVOL_raw']=d.DVOL_raw.to_numpy().copy();audits=[];tunes=[];selected=None;lastmonth=None
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
