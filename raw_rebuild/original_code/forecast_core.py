"""Calendar-aligned, purged rolling forecasts and dependence-robust inference."""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.linear_model import LassoLars
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

SEED=20261007
EPS=1e-12

def forward_mean(x,h):
    # At the end of t: target comprises t+1,...,t+h. Never compress the calendar.
    return x.shift(-1).rolling(h,min_periods=h).mean().shift(-(h-1))

def har(x,prefix,form='log'):
    z=np.log(x.clip(lower=EPS)) if form=='log' else x
    return pd.DataFrame({prefix+'_d':z,prefix+'_w':z.rolling(7,min_periods=7).mean(),
                         prefix+'_m':z.rolling(30,min_periods=30).mean()})

def design(series):
    x=np.asarray(series,dtype=float)
    return np.column_stack([np.ones(len(x)),x])

def hac_mean(x,lag):
    x=np.asarray(x,dtype=float);x=x[np.isfinite(x)];n=len(x)
    if n<max(10,lag+2):return np.nan,np.nan,np.nan
    v=x-x.mean();lr=float(v@v/n)
    for j in range(1,min(lag,n-2)+1):
        lr+=2*(1-j/(lag+1))*float(v[j:]@v[:-j]/n)
    se=np.sqrt(max(lr,0)/n)
    t=float(x.mean()/se) if se>EPS else (0.0 if abs(x.mean())<EPS else np.sign(x.mean())*np.inf)
    return float(x.mean()),float(se),t

def dm_test(l0,l1,h):
    x=np.asarray(l0)-np.asarray(l1);x=x[np.isfinite(x)]
    m,se,t=hac_mean(x,2*h);n=len(x)
    if not np.isfinite(m):return np.nan,np.nan
    cor=np.sqrt(max((n+1-2*h+h*(h-1)/n)/n,0))
    t=t*cor;p=2*stats.t.sf(abs(t),n-1)
    return float(t),float(p)

def cw_test(y,f0,f1,h):
    # Single-sided alternative: extended forecast improves baseline MSE.
    d=(y-f0)**2-(y-f1)**2+(f0-f1)**2
    _,_,t=hac_mean(d,2*h)
    return float(t),float(stats.norm.sf(t))

def qlike(y,f):
    r=np.maximum(y,EPS)/np.maximum(f,EPS)
    return r-np.log(r)-1

def block_indices(n,reps,block,seed=SEED):
    g=np.random.default_rng(seed);blocks=int(np.ceil(n/block))
    starts=g.integers(0,n,size=(reps,blocks))
    return ((starts[:,:,None]+np.arange(block)[None,None,:])%n).reshape(reps,-1)[:,:n]

def block_ci(x,h,reps=1999):
    x=np.asarray(x);x=x[np.isfinite(x)]
    if len(x)<max(30,2*h+2):return np.nan,np.nan
    b=block_indices(len(x),reps,max(2*h,7))
    return tuple(np.quantile(x[b].mean(axis=1),[.025,.975]).astype(float))

def mcs(loss,h,alpha=.10,reps=1999):
    """Tmax MCS, common circular-block draws, sequential worst-model removal."""
    a=np.asarray(loss,dtype=float);a=a[np.isfinite(a).all(axis=1)]
    if len(a)<max(30,2*h+2):return [],[]
    idx=block_indices(len(a),reps,max(2*h,7));active=list(range(a.shape[1]));history=[]
    bm=np.stack([a[j].mean(axis=0) for j in idx])
    while len(active)>1:
        l=a[:,active];d=l.mean(axis=0)-l.mean();
        boot=bm[:,active]-bm[:,active].mean(axis=1,keepdims=True)
        se=np.maximum(boot.std(axis=0,ddof=1),EPS)
        obs=np.max(d/se);sim=np.max((boot-d)/se,axis=1)
        p=float((1+(sim>=obs).sum())/(reps+1));worst=int(np.argmax(d/se))
        history.append({'models':','.join(map(str,active)),'p':p,'remove':active[worst] if p<alpha else None})
        if p>=alpha:break
        active.pop(worst)
    return active,history

def joint_fit(x,y,h,extra_cols,form='log'):
    z=pd.concat([x,y.rename('target')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)<max(100,len(x.columns)*5):return None
    yy=np.log(z.target.clip(lower=EPS)) if form=='log' else z.target
    xx=pd.DataFrame(StandardScaler().fit_transform(z[x.columns]),index=z.index,columns=x.columns)
    xx=sm.add_constant(xx,has_constant='add')
    m=sm.OLS(yy,xx).fit(cov_type='HAC',cov_kwds={'maxlags':2*h})
    cols=[c for c in extra_cols if c in x.columns]
    if not cols:return m,1.0,[]
    r=np.zeros((len(cols),len(m.params)))
    for i,c in enumerate(cols):r[i,list(m.params.index).index(c)]=1
    w=m.wald_test(r,scalar=True)
    rows=[{'variable':c,'coefficient':float(m.params[c]),'se':float(m.bse[c]),'p':float(m.pvalues[c])} for c in m.params.index]
    return m,float(w.pvalue),rows

def _transformed_train(x,train,predict,pca_cols=None,pca_only=False,orthogonal=False):
    # Fit scaling and PCA only on training rows, never on the evaluation period.
    scaler=StandardScaler().fit(x[train]);a=scaler.transform(x[train]);b=scaler.transform(x[[predict]])
    if orthogonal:
        # Perpetual innovation relative to spot HAR space, fitted within this window.
        spot=design(a[:,:3]);projection=np.linalg.lstsq(spot,a[:,3:6],rcond=None)[0]
        a[:,3:6]=a[:,3:6]-spot@projection
        b[:,3:6]=b[:,3:6]-design(b[:,:3])@projection
    if pca_cols:
        pc=PCA(n_components=1,random_state=SEED).fit(a[:,pca_cols])
        weights=pc.components_[0]
        sign=1 if weights[0]>=0 else -1
        pa=sign*pc.transform(a[:,pca_cols])[:,0];pb=sign*pc.transform(b[:,pca_cols])[:,0]
        if pca_only:
            keep=[j for j in range(a.shape[1]) if j not in pca_cols]
            a=np.column_stack([a[:,keep],pa]);b=np.column_stack([b[:,keep],pb])
        else:
            a=np.column_stack([a,pa]);b=np.column_stack([b,pb])
    return a,b

def rolling_predict(x,y,h,kind='ols',window=1000,scheme='rolling',form='log',start=1000,pca_cols=None,pca_only=False,orthogonal=False):
    """Refit daily. Training labels end no later than origin t+1, including CV purge."""
    a=np.asarray(x,dtype=float);yy=np.asarray(y,dtype=float)
    ok=np.isfinite(a).all(axis=1)&np.isfinite(yy)
    target=np.log(np.maximum(yy,EPS)) if form=='log' else yy
    pred=np.full(len(yy),np.nan);selected=[];counts={'nonpositive':0,'origin_count':0,'train_n_min':None,'train_n_max':None,'train_n_sum':0}
    for t in range(start,len(yy)):
        # An unknown future target may never decide whether a forecast is issued.
        # Future-target completeness enters evaluation only, not information at t.
        if not np.isfinite(a[t]).all():continue
        last=t-h # y[last] is fully observed by end-of-day t.
        first=max(0,last-window+1) if scheme=='rolling' else 0
        tr=np.flatnonzero(ok[first:last+1])+first
        # The registered window is elapsed calendar days. Missing observations
        # reduce effective N; a second unregistered 90% threshold must not stop it.
        if len(tr)<max(100,a.shape[1]*5):continue
        counts['train_n_min']=len(tr) if counts['train_n_min'] is None else min(counts['train_n_min'],len(tr))
        counts['train_n_max']=len(tr) if counts['train_n_max'] is None else max(counts['train_n_max'],len(tr))
        counts['train_n_sum']+=len(tr)
        xx,xt=_transformed_train(a,tr,t,pca_cols,pca_only,orthogonal)
        train_y=target[tr]
        if kind=='ols':
            bx=design(xx);beta=np.linalg.lstsq(bx,train_y,rcond=None)[0]
            fitted=bx@beta;fp=float(design(xt)[0]@beta)
        else:
            # Three chronological validation folds. CV labels purge the horizon.
            cv=TimeSeriesSplit(n_splits=3,gap=h)
            if kind=='lasso':grid=[{'alpha':v} for v in [.001,.01,.1]]
            elif kind=='gb':grid=[{'learning_rate':lr,'max_depth':d} for lr in [.03,.10] for d in [1,2]]
            else:raise ValueError(kind)
            scores=[]
            for params in grid:
                fold=[]
                for it,iv in cv.split(xx):
                    # Outer scaling is refit inside CV; it cannot see validation rows.
                    sc=StandardScaler().fit(a[tr[it]])
                    cx=sc.transform(a[tr[it]]);vx=sc.transform(a[tr[iv]])
                    if pca_cols:
                        pc=PCA(n_components=1,random_state=SEED).fit(cx[:,pca_cols])
                        sg=1 if pc.components_[0,0]>=0 else -1
                        cx=np.column_stack([cx,sg*pc.transform(cx[:,pca_cols])[:,0]])
                        vx=np.column_stack([vx,sg*pc.transform(vx[:,pca_cols])[:,0]])
                    est=LassoLars(**params,max_iter=1000) if kind=='lasso' else HistGradientBoostingRegressor(**params,max_iter=50,max_bins=64,early_stopping=False,random_state=SEED)
                    est.fit(cx,train_y[it]);fold.append(np.mean((train_y[iv]-est.predict(vx))**2))
                scores.append(np.mean(fold))
            params=grid[int(np.argmin(scores))]
            est=LassoLars(**params,max_iter=1000) if kind=='lasso' else HistGradientBoostingRegressor(**params,max_iter=50,max_bins=64,early_stopping=False,random_state=SEED)
            est.fit(xx,train_y);fp=float(est.predict(xt)[0]);fitted=est.predict(xx)
            if kind=='lasso':selected.append({'date':str(x.index[t]),'variables':','.join(list(x.columns)[i] if i<len(x.columns) else 'crowding_pc1' for i,v in enumerate(est.coef_) if abs(v)>1e-8),'alpha':params['alpha']})
        if form=='log':
            # Duan retransformation; smearing is estimated in this same training window.
            smearing=float(np.mean(np.exp(np.clip(train_y-fitted,-700,700))))
            fp=float(np.exp(np.clip(fp,-700,700))*smearing)
        if fp<=EPS:counts['nonpositive']+=1
        pred[t]=max(fp,EPS);counts['origin_count']+=1
    return pd.Series(pred,index=x.index),selected,counts

def compare_paths(y,f0,f1,h,nested=False):
    z=pd.concat([y.rename('y'),f0.rename('baseline'),f1.rename('extended')],axis=1).dropna()
    if len(z)<max(30,2*h+2):return {'n':len(z),'status':'insufficient_oos'}
    yy=z.y.to_numpy();a=z.baseline.to_numpy();b=z.extended.to_numpy()
    l0=qlike(yy,a);l1=qlike(yy,b);m0=(yy-a)**2;m1=(yy-b)**2
    dq,pq=dm_test(l0,l1,h);dm,pm=dm_test(m0,m1,h);tc,pc=cw_test(yy,a,b,h)
    enc=sm.OLS(yy,sm.add_constant(z[['baseline','extended']],has_constant='add')).fit(cov_type='HAC',cov_kwds={'maxlags':2*h})
    return {'n':len(z),'status':'estimated','qlike_baseline':float(l0.mean()),'qlike_model':float(l1.mean()),
            'qlike_improvement_pct':100*(1-float(l1.mean()/l0.mean())),
            'mse_baseline':float(m0.mean()),'mse_model':float(m1.mean()),'oos_r2':1-float(m1.mean()/m0.mean()),
            'dm_qlike_t':dq,'dm_qlike_p':pq,'dm_mse_t':dm,'dm_mse_p':pm,
            'cw_t':tc if nested else np.nan,'cw_p':pc if nested else np.nan,
            'encompassing_model_coefficient':float(enc.params['extended']),'encompassing_model_p':float(enc.pvalues['extended'])}

def var_tests(realized,pred,alpha=.01):
    z=pd.concat([realized.rename('r'),pred.rename('p')],axis=1).dropna()
    if len(z)<30:return {'n':len(z),'status':'insufficient_oos'}
    # RV measures log-return variance; compare simple returns to the expm1 quantile.
    threshold=np.expm1(stats.norm.ppf(alpha)*np.sqrt(np.maximum(z.p.to_numpy(),EPS)))
    hits=(z.r.to_numpy()<threshold).astype(int)
    n=len(hits);k=int(hits.sum());ph=np.clip(k/n,EPS,1-EPS)
    ll0=k*np.log(alpha)+(n-k)*np.log1p(-alpha);ll1=k*np.log(ph)+(n-k)*np.log1p(-ph)
    luc=max(0,2*(ll1-ll0));n00=np.sum((hits[:-1]==0)&(hits[1:]==0));n01=np.sum((hits[:-1]==0)&(hits[1:]==1))
    n10=np.sum((hits[:-1]==1)&(hits[1:]==0));n11=np.sum((hits[:-1]==1)&(hits[1:]==1))
    p0=np.clip(n01/max(n00+n01,1),EPS,1-EPS);p1=np.clip(n11/max(n10+n11,1),EPS,1-EPS)
    pp=np.clip((n01+n11)/max(n00+n01+n10+n11,1),EPS,1-EPS)
    lmark=n00*np.log1p(-p0)+n01*np.log(p0)+n10*np.log1p(-p1)+n11*np.log(p1)
    lindep=(n00+n10)*np.log1p(-pp)+(n01+n11)*np.log(pp)
    li=max(0,2*(lmark-lindep));lc=luc+li
    return {'n':n,'exceptions':k,'exception_rate':k/n,'kupiec_lr':luc,'kupiec_p':float(stats.chi2.sf(luc,1)),
            'independence_lr':li,'independence_p':float(stats.chi2.sf(li,1)),'conditional_lr':lc,'conditional_p':float(stats.chi2.sf(lc,2)),
            'status':'estimated'}
