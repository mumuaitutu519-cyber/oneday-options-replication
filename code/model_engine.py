from common import *
from sklearn.ensemble import HistGradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import LassoLars
PARAMS=[dict(max_iter=n,max_leaf_nodes=k) for n in [40,80] for k in [7,15]]
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

old_matrices=matrices
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
 dest=WORK/f'predictions_{c}_{spec}.parquet'
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

def ols_predict(X,y,tr,t,form):
 xx=X[tr];xt=X[t];mu=xx.mean(0);scale=xx.std(0);scale[scale<1e-15]=1;xx=np.column_stack([np.ones(len(tr)),(xx-mu)/scale]);xt=np.r_[1.,(xt-mu)/scale];target=np.log(np.maximum(y[tr],EPS)) if form=='log' else y[tr];beta=np.linalg.lstsq(xx,target,rcond=1e-10)[0];p=xt@beta
 if form=='log':p=np.exp(np.clip(p,-50,30))*np.mean(np.exp(np.clip(target-xx@beta,-50,30)))
 return max(float(p),EPS)
