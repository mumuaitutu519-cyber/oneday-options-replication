from common import *
def bern_ll(k,n,p):return special.xlogy(k,p)+special.xlogy(n-k,1-p)

def coverage_test(exceed,date,alpha):
 e=np.asarray(exceed,int);n=len(e);k=int(e.sum());p=k/n if n else np.nan;uc=2*(bern_ll(k,n,p)-bern_ll(k,n,1-alpha)) if n else np.nan
 consecutive=np.diff(pd.to_datetime(date).to_numpy('datetime64[D]')).astype(int)==1;before=e[:-1][consecutive];after=e[1:][consecutive];N=np.zeros((2,2),int)
 for a,b in zip(before,after):N[a,b]+=1
 n0=N[0].sum();n1=N[1].sum();total=N.sum();n11=N[1,1];n01=N[0,1];pool=(n01+n11)/total if total else np.nan
 ind=2*(bern_ll(n01,n0,n01/n0)+bern_ll(n11,n1,n11/n1)-bern_ll(n01+n11,total,pool)) if n0>0 and n1>0 else np.nan
 return dict(n=n,exceptions=k,exceedance=p,uc_lr=uc,uc_p=stats.chi2.sf(max(uc,0),1),ind_lr=ind,ind_p=stats.chi2.sf(max(ind,0),1) if np.isfinite(ind) else np.nan,cc_lr=uc+ind,cc_p=stats.chi2.sf(max(uc+ind,0),2) if np.isfinite(ind) else np.nan,transition_n=int(total),n00=int(N[0,0]),n01=int(N[0,1]),n10=int(N[1,0]),n11=int(N[1,1]),transition_status='estimated' if n0>0 and n1>0 else 'no_conditional_tail_transition')

