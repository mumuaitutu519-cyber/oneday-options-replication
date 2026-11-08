import numpy as np
from scipy.special import ndtr
def black(F,K,w,kind):
 F=np.asarray(F,float);K=np.asarray(K,float);w=np.maximum(np.asarray(w,float),1e-24)
 s=np.sqrt(w);d1=(np.log(F/K)+w/2)/s;d2=d1-s
 return np.where(np.asarray(kind)=='call',F*ndtr(d1)-K*ndtr(d2),K*ndtr(-d2)-F*ndtr(-d1))

def implied_w(F,K,P,kind):
 F=np.asarray(F,float);K=np.asarray(K,float);P=np.asarray(P,float);kind=np.asarray(kind)
 intrinsic=np.where(kind=='call',np.maximum(F-K,0),np.maximum(K-F,0));upper=np.where(kind=='call',F,K)
 good=np.isfinite(F)&np.isfinite(K)&np.isfinite(P)&(F>0)&(K>0)&(P>intrinsic)&(P<upper)
 lo=np.zeros(len(F));hi=np.full(len(F),64.)
 for _ in range(60):
  mid=(lo+hi)/2;less=black(F,K,mid,kind)<P;lo=np.where(less,mid,lo);hi=np.where(less,hi,mid)
 w=(hi+lo)/2;w[~good]=np.nan
 return w,good
