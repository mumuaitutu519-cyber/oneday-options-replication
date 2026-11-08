import os
import numpy as np,pandas as pd
from scipy import stats
SEED=20261007
EPS=1e-12
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
