from r9_common import *
from scipy import optimize

ROWS={}
def add(name,r):ROWS.setdefault(name,[]).append(r)
def ts(g,col):return pd.Series(g[col].to_numpy(float),index=pd.DatetimeIndex(g.date))
def ratio_stats(g,numerator,denominator,h=1):
 g=g.dropna(subset=[numerator,denominator]);num=ts(g,numerator);den=ts(g,denominator);ratio=num.mean()/den.mean() if den.mean()!=0 else np.nan
 influence=(num-ratio*den)/den.mean();r=summary(influence,h,False);r.update(mean=ratio,median=np.nan,ci_low=ratio-1.96*r.get('se',np.nan),ci_high=ratio+1.96*r.get('se',np.nan),p=np.nan)
 dates=pd.date_range(num.index.min(),num.index.max());nn=num.reindex(dates).to_numpy();dd=den.reindex(dates).to_numpy();w=weights(len(dates));bn=w@np.nan_to_num(nn);bd=w@np.nan_to_num(dd);br=np.divide(bn,bd,out=np.full(REPS,np.nan),where=bd!=0);r['boot_low'],r['boot_high']=np.nanquantile(br,[.025,.975]);return r
def metrics(g,name,cols,boot=True,**labels):
 for metric,col,comp in cols:
  add(name,dict(**labels,component=comp,metric=metric,**summary(ts(g,col),int(labels.get('tenor',1)),boot)))

def native_terms(c,q,rv):
 rows=[]
 for (variant,date),z in q.groupby(['variant','date']):
  z=z.sort_values('expiry');origin=date+pd.Timedelta(hours=9)
  for h in TENORS:
   target=origin+pd.Timedelta(days=h);lo=z[z.expiry<=target];hi=z[z.expiry>=target]
   if lo.empty or hi.empty:continue
   l=lo.iloc[-1];u=hi.iloc[0]
   if l.expiry==u.expiry:
    above=z[z.expiry>target];below=z[z.expiry<target]
    if len(above):u=above.iloc[0]
    elif len(below):l=below.iloc[-1]
    else:continue
   w=(target-l.expiry)/(u.expiry-l.expiry);rr=dict(currency=c,date=date,variant=variant,tenor=h,observed_at=origin,target_end=target,window='fixed',last_trade=max(l.last_trade,u.last_trade))
   for src,dst in [('atm','atm_I'),('total','total_I'),('center','center_I')]:rr[dst]=(1-w)*l[src]+w*u[src]
   rows.append(rr)
 return pd.DataFrame(rows).merge(rv[(rv.variant=='rv5')&(rv.window=='fixed')].drop(columns=['variant','observed_at','target_end']),on=['currency','date','tenor','window'],how='left',validate='many_to_one')

def build_panel(c):
 p=pd.read_parquet(R5/f'premium_panel_{c}.parquet');rv=pd.read_parquet(O/f'rv_{c}.parquet');q=pd.read_parquet(O/f'quotes_{c}.parquet')
 qindex=q[q.variant.isin(['trade_native','mark_native'])].copy();qindex['variant']=qindex.variant.str.replace('_native','_index_corridor',regex=False)
 futures=pd.read_parquet(O/f'futures_corridor_{c}.parquet');q=q.merge(futures[['date','expiry','variant','center_futures','F_futures','futures_known_at']],on=['date','expiry','variant'],how='left',validate='one_to_one')
 primary=q.variant.isin(['trade_native','mark_native']);q.loc[primary,'center']=q.loc[primary,'center_futures'];q=pd.concat([q,qindex],ignore_index=True)
 native=native_terms(c,q,rv);parts=[native]
 total=p[p.variant=='model_free'][['currency','date','tenor','observed_at','target_end','mf_w','rv_total','last_trade']].rename(columns={'mf_w':'total_I','rv_total':'R5_rv'})
 for old,new in [('trade_ATM','trade_R5total'),('mark','mark_R5total')]:
  atm=p[p.variant==old][['date','tenor','atm_w']].rename(columns={'atm_w':'atm_I'});z=total.merge(atm,on=['date','tenor'],validate='one_to_one');z['variant']=new;z['window']='fixed';z=z.merge(rv[(rv.variant=='rv5')&(rv.window=='fixed')].drop(columns=['variant','observed_at','target_end']),on=['currency','date','tenor','window'],how='left',validate='one_to_one');z['center_I']=np.nan;parts.append(z)
 # Daily actual expiry (23h), reusing R5 feature raw IV and y; MF from R4 expiry buckets for exact comparability.
 features=pd.read_parquet(R5/f'features_{c}.parquet');refs=pd.read_parquet(R4/f'expiry_quotes_{c}.parquet');refs=refs[(refs.window_hour==8)&(refs.expiry==refs.date+pd.Timedelta(days=1,hours=8))][['date','mf_w']].rename(columns={'mf_w':'total_I'});f=features[['date','observed_at','label_end','IV_raw','y','last_option_trade']].rename(columns={'label_end':'target_end','IV_raw':'atm_I','y':'R5_rv','last_option_trade':'last_trade'});f=f.merge(refs,on='date',how='left',validate='one_to_one');f['currency']=c;f['tenor']=1;f['window']='expiry23';f['variant']='trade_R5total';f=f.merge(rv[(rv.variant=='rv5')&(rv.window=='expiry23')].drop(columns=['variant','observed_at','target_end']),on=['currency','date','tenor','window'],how='left',validate='one_to_one');f['center_I']=np.nan;parts.append(f)
 for v in q.variant.unique():
  z=q[(q.variant==v)&(q.expiry==q.date+pd.Timedelta(days=1,hours=8))].copy();z=z.rename(columns={'atm':'atm_I','total':'total_I','center':'center_I','expiry':'target_end'});z['tenor']=1;z['window']='expiry23';z=z.merge(rv[(rv.variant=='rv5')&(rv.window=='expiry23')].drop(columns=['variant','observed_at','target_end']),on=['currency','date','tenor','window'],how='left',validate='one_to_one');parts.append(z)
 panel=pd.concat(parts,ignore_index=True);panel['tail_I']=panel.total_I-panel.atm_I;panel['wing_I']=panel.total_I-panel.center_I
 panel=event_flags(panel);panel.to_parquet(O/f'panel_{c}.parquet',index=False)
 for (v,w,h),g in panel.groupby(['variant','window','tenor']):
  matched=g[['atm_I','total_I','rv','bv']].notna().all(axis=1)
  add('coverage.csv',dict(currency=c,variant=v,window=w,tenor=h,component='total',n=len(g),matched_n=int(matched.sum()),implied_n=int(g[['atm_I','total_I']].notna().all(axis=1).sum()),rv_n=int(g.rv.notna().sum()),start=g.date.min(),end=g.date.max(),matched_start=g.loc[matched,'date'].min(),matched_end=g.loc[matched,'date'].max()))
 return panel,rv

def premium(g,est='bv',corridor=False):
 g=g.copy();g['continuous']=g[est];g['jump']=(g.rv-g.continuous).clip(lower=0);g['atm']=g['center_I' if corridor else 'atm_I']-g.continuous;g["tail"]=g['wing_I' if corridor else 'tail_I']-g.jump;g['total']=g.total_I-g.rv;g['atm_raw']=g.atm_I-g.rv
 g['tail_share']=g["tail"]/g.total.replace(0,np.nan);g['tail_normalized']=g["tail"]/g['wing_I' if corridor else 'tail_I'].replace(0,np.nan);g['atm_log']=np.log(g['center_I' if corridor else 'atm_I'].where(lambda x:x>0)/g.continuous.where(lambda x:x>0));g['total_log']=np.log(g.total_I.where(lambda x:x>0)/g.rv.where(lambda x:x>0));return g

def premium_rows(g,name,variant,split='all',boot=True,extra=None):
 c=g.currency.iloc[0];h=int(g.tenor.iloc[0]);common=g.dropna(subset=['total_I','atm_I','continuous','rv','tail_I']);cols=[]
 for comp in ['atm','tail','total']:
  common[comp+'_annual']=common[comp]*365/h;cols.extend([('vrp',comp,comp),('vrp_annual',comp+'_annual',comp)])
 cols.extend([('log_ratio','atm_log','atm'),('log_ratio','total_log','total'),('normalized_premium','tail_normalized','tail'),('premium_share','tail_share','tail')]);labels=dict(currency=c,tenor=h,split=split,variant=variant,window=g.window.iloc[0],classification='事后' if name in ['jump_premium.csv'] else '配对实现分解',**(extra or {}));metrics(common,name,cols,boot,**labels)
 if len(common):
  add(name,dict(**labels,component='tail',metric='ratio_of_mean_premia',**ratio_stats(common,'tail','total',h),total_mean=common.total.mean(),tail_mean=common["tail"].mean(),negative_tail_I=int((common.tail_I<0).sum()),near_zero_total_count=int((common.total.abs()<1e-10).sum())))
 return common

def jump_statistics(c,rv):
 d=pd.concat([rv,pd.read_parquet(O/f'rv_daily23_{c}.parquet')],ignore_index=True)
 for (variant,window,h),g in d.groupby(['variant','window','tenor']):
  for split,z in [('all',g)]+[(str(y),zz) for y,zz in g.groupby(g.date.dt.year)]:
   for estimator in ['bv','trv','medrv']:
    s=z.copy();s['jump_var']=(s.rv-s[estimator]).clip(lower=0);s['share']=s.jump_var/s.rv;s['continuous_raw']=s[estimator+'_raw'];labels=dict(currency=c,tenor=h,variant=variant+'_'+estimator,window=window,split=split,classification='事后');metrics(s,'jump_statistics.csv',[('variance_share','share','jump'),('variance','jump_var','jump'),('variance',estimator,'continuous')],False,**labels)
    add('jump_statistics.csv',dict(**labels,component='jump',metric='ratio_of_sum_variances',n=len(s),mean=s.jump_var.sum()/s.rv.sum(),median=np.nan,continuous_clipped_count=int((s.continuous_raw>s.rv).sum())))
   for rule in ['jump1','jump5','jump_rule']:
    s=z.copy();s['indicator']=s[rule].astype(float);labels=dict(currency=c,tenor=h,variant=variant+'_'+rule,window=window,split=split,classification='事后');metrics(s,'jump_statistics.csv',[('day_frequency','indicator','jump')],False,**labels);metrics(s[s[rule].astype(bool)],'jump_statistics.csv',[('max_abs_5min_return' if variant=='rv5' else 'max_abs_sampled_return','max_abs_return','jump')],False,**labels)
 log(f'{c} 跳跃全样本/逐年统计完成')

def implied_stats(p):
 for (variant,window,h),g in p[p.window=='fixed'].groupby(['variant','window','tenor']):
  for mode in ['atm','corridor']:
   if mode=='atm' and variant.endswith('index_corridor'):continue
   z=g.dropna(subset=['total_I','atm_I','center_I'] if mode=='corridor' else ['total_I','atm_I']).copy()
   if z.empty:continue
   z['center']=z.center_I if mode=='corridor' else z.atm_I;z["tail"]=z.total_I-z.center;z['share']=z["tail"]/z.total_I
   metrics(z,'implied_decomposition.csv',[('implied_variance','center','atm'),('implied_variance','tail','tail'),('implied_variance','total_I','total'),('variance_share','share','tail')],True,currency=z.currency.iloc[0],tenor=h,variant=variant+'_'+mode,window=window,split='all',negative_tail_n=int((z["tail"]<0).sum()))
   add('implied_decomposition.csv',dict(currency=z.currency.iloc[0],tenor=h,variant=variant+'_'+mode,window=window,split='all',component='tail',metric='ratio_of_mean_variances',n=len(z),mean=z["tail"].mean()/z.total_I.mean()))

def term_stats(p):
 for (variant,window,h),g in p[(p.window=='fixed')&p.variant.isin(['trade_R5total','mark_R5total','trade_native','mark_native','trade_index_corridor','mark_index_corridor'])].groupby(['variant','window','tenor']):
  z=None
  if not variant.endswith('index_corridor'):z=premium_rows(premium(g),'term_premium.csv',variant+'_bv')
  if variant in ['trade_native','mark_native','trade_index_corridor','mark_index_corridor']:premium_rows(premium(g,corridor=True).dropna(subset=['center_I']),'term_premium.csv',variant+'_corridor_bv')
  if h==1 and z is not None:log(f'{g.currency.iloc[0]} {variant} 一天配对 n={len(z)} ATM-C={z.atm.mean():.9f} tail-J={z["tail"].mean():.9f} total={z.total.mean():.9f}')
 for variant,g in p[(p.window=='fixed')&p.variant.isin(['trade_R5total','mark_R5total','trade_native','mark_native'])].groupby('variant'):
  s=premium(g);one=s[s.tenor==1].set_index('date');long=s[s.tenor==30].set_index('date');z=one[['tail','total','tail_share']].join(long[['tail','total','tail_share']],lsuffix='_1',rsuffix='_30',how='inner').dropna();z['share_difference']=z.tail_share_1-z.tail_share_30;z['raw_tail_difference']=z.tail_1-z.tail_30;z['annual_tail_difference']=z.tail_1*365-z.tail_30*365/30;z['total_annual_difference']=z.total_1*365-z.total_30*365/30;z['date']=z.index
  metrics(z,'term_difference.csv',[(k,k,'tail' if k!='total_annual_difference' else 'total') for k in ['share_difference','raw_tail_difference','annual_tail_difference','total_annual_difference']],True,currency=g.currency.iloc[0],tenor=30,variant=variant+'_bv',split='common_1_30',comparison='1_minus_30')
  for h,col in [(1,'tail_1'),(30,'tail_30')]:
   tcol=f'total_{h}';ratio=z[col].mean()/z[tcol].mean();add('term_difference.csv',dict(currency=g.currency.iloc[0],tenor=h,variant=variant+'_bv',split='common_1_30',component='tail',metric='ratio_of_mean_premia',n=len(z),mean=ratio))
 # Exact R5 all-available total comparison, independent of matched ATM availability.
 r5=pd.read_csv(R5/'term_premium.csv');c=p.currency.iloc[0];src=pd.read_parquet(R5/f'premium_panel_{c}.parquet');src=src[src.variant=='model_free']
 for h,g in src.groupby('tenor'):
  z=g.dropna(subset=['atm_w','rv_total']);z=z[z.rv_total>0].copy();z['total']=z.atm_w-z.rv_total;z['volpts']=100*(np.sqrt(z.atm_w*365/h)-np.sqrt(z.rv_total*365/h));z['log_ratio']=np.log(z.atm_w/z.rv_total)
  for metric,col in [('vrp','total'),('volpts','volpts'),('log_ratio','log_ratio')]:
   now=summary(ts(z,col),h,False);old=r5[(r5.currency==c)&(r5.tenor==h)&(r5.variant=='model_free')&(r5.split=='all')&(r5.metric==metric)].iloc[0];add('R5_total_reconciliation.csv',dict(currency=c,tenor=h,component='total',variant='R5_all_available',metric=metric,n=now['n'],R5_n=int(old.n),mean=now['mean'],R5_mean=old['mean'],mean_error=now['mean']-old['mean'],median_error=now['median']-old['median'],status='PASS' if now['n']==int(old.n) and abs(now['mean']-old['mean'])<1e-12 else 'FAIL'))


def contrast(g,metric,a,b,h=1):
 # HAC ratio influence functions on complete calendar, with both means estimated together.
 dates=pd.date_range(g.date.min(),g.date.max());v=ts(g,metric).reindex(dates);ma=pd.Series(np.asarray(a,bool),index=pd.DatetimeIndex(g.date)).reindex(dates,fill_value=False)&v.notna();mb=pd.Series(np.asarray(b,bool),index=pd.DatetimeIndex(g.date)).reindex(dates,fill_value=False)&v.notna();na=ma.sum();nb=mb.sum()
 if min(na,nb)<3:return dict(n=int(na+nb),group_n=int(na),reference_n=int(nb),mean=np.nan,se=np.nan,p=np.nan,status='insufficient')
 aa=v[ma].mean();bb=v[mb].mean();score=ma.astype(float)*(v.fillna(aa)-aa)*len(dates)/na-mb.astype(float)*(v.fillna(bb)-bb)*len(dates)/nb;res=summary(score,h,False);m=aa-bb;se=res['se'];t=m/se if se else np.nan;res.update(mean=m,t=t,p=2*stats.norm.sf(abs(t)),p_greater=stats.norm.sf(t),p_less=stats.norm.cdf(t),group_n=int(na),reference_n=int(nb),ci_low=m-1.96*se,ci_high=m+1.96*se,status='estimated');return res

def probit_fit(x,y,dates):
 X=np.column_stack([np.ones(len(x)),x]);y=np.asarray(y,float)
 def fun(b):
  eta=X@b;ll=np.where(y>0,special.log_ndtr(eta),special.log_ndtr(-eta));return -ll.sum()
 def jac(b):
  eta=X@b;ratio=np.exp(stats.norm.logpdf(eta)-np.where(y>0,special.log_ndtr(eta),special.log_ndtr(-eta)));return -(X*((2*y-1)*ratio)[:,None]).sum(0)
 fit=optimize.minimize(fun,np.zeros(2),jac=jac,method='BFGS',options={'gtol':1e-7,'maxiter':300});b=fit.x;eta=X@b;phi=stats.norm.pdf(eta);P=special.ndtr(eta);fac=np.where(y>0,np.exp(stats.norm.logpdf(eta)-special.log_ndtr(eta)),np.exp(stats.norm.logpdf(eta)-special.log_ndtr(-eta)));score=X*((2*y-1)*fac)[:,None];w=fac*(fac+(2*y-1)*eta);H=X.T@(X*w[:,None]);idx=pd.DatetimeIndex(dates);sc=pd.DataFrame(score,index=idx).reindex(pd.date_range(idx.min(),idx.max()),fill_value=0).to_numpy();S=sc.T@sc
 for j in range(1,min(7,len(sc)-1)+1):G=sc[j:].T@sc[:-j];S+=(1-j/8)*(G+G.T)
 inv=np.linalg.pinv(H);cov=inv@S@inv;se=np.sqrt(np.maximum(np.diag(cov),0));return b,se,P,fit.success or np.linalg.norm(jac(b))<1e-5

def auc(y,p):
 y=np.asarray(y,bool);p=np.asarray(p,float);good=np.isfinite(p);y=y[good];p=p[good];n1=y.sum();n0=(~y).sum()
 return (stats.rankdata(p)[y].sum()-n1*(n1+1)/2)/(n1*n0) if n1*n0 else np.nan

def attribution(p):
 c=p.currency.iloc[0]
 for (variant,window),g in p[(p.tenor==1)&p.variant.isin(['trade_R5total','mark_native'])].groupby(['variant','window']):
  s=premium(g).dropna(subset=['atm_I','rv','bv']).sort_values('date')
  for rule in ['jump1','jump5','jump_rule']:
   for jumpflag,z in s.groupby(rule):
    labels=dict(currency=c,tenor=1,variant=variant+'_bv_'+rule,window=window,split='jump' if jumpflag else 'nonjump',classification='事后')
    metrics(z,'jump_premium.csv',[('vrp_ATM_minus_RV','atm_raw','atm'),('vrp','atm','continuous'),('vrp','tail','tail'),('vrp','total','total')],False,**labels)
   j=s[s[rule].astype(bool)]
   for category in ['CPI','NFP','FOMC','macro','weekend']:
    if len(j):add('jump_calendar.csv',dict(currency=c,tenor=1,variant=variant+'_'+rule,window=window,classification='事后',component='jump',category=category,n=len(j),mean=j[category].mean()))
  curve=s.sort_values('rv',ascending=False).copy();curve['component']='atm';curve['rank']=np.arange(1,len(curve)+1);curve['day_share']=curve['rank']/len(curve);curve['cum_atm_premium']=curve.atm_raw.cumsum();curve['net_atm_premium']=curve.atm_raw.sum();curve['top5']=curve.rv>curve.rv.quantile(.95);add('contribution_summary.csv',dict(currency=c,tenor=1,variant=variant,window=window,classification='事后',component='atm',n=len(s),top5_fraction_of_net=curve.loc[curve.top5,'atm_raw'].sum()/curve.atm_raw.sum(),top5_jump_share=curve.loc[curve.top5,'jump1'].mean(),jump_fraction_of_net=curve.loc[curve.jump1.astype(bool),'atm_raw'].sum()/curve.atm_raw.sum(),jump_mean=s.loc[s.jump1.astype(bool),'atm_raw'].mean(),nonjump_mean=s.loc[~s.jump1.astype(bool),'atm_raw'].mean(),nonjump_positive_share=(s.loc[~s.jump1.astype(bool),'atm_raw']>0).mean()));curve.to_csv(O/f'contribution_{c}_{variant}_{window}.csv',index=False)
  # Previous day origin, exact calendar matching. All pricing timestamps remain before that origin.
  a=s.set_index('date');past=a[['atm_I','tail_I','last_trade']].copy();past.index+=pd.Timedelta(days=1);past=past.rename(columns={'atm_I':'prior_atm','tail_I':'prior_tail','last_trade':'prior_last_trade'});t=a.join(past,how='left');t['date']=t.index
  for flag,z in t.groupby('jump1'):
   metrics(z,'prior_pricing.csv',[('prior_implied_variance','prior_atm','atm'),('prior_implied_variance','prior_tail','tail')],False,currency=c,tenor=1,variant=variant,window=window,split='jump' if flag else 'nonjump',classification='事后标签/前一日09:00信息')
  for k,comp in [('prior_atm','atm'),('prior_tail','tail')]:add('prior_pricing_tests.csv',dict(currency=c,tenor=1,variant=variant,window=window,component=comp,metric=k,classification='事后标签/前一日09:00信息',**contrast(t,k,t.jump1,~t.jump1.astype(bool))))
  cal=a.tail_I.reindex(pd.date_range(a.index.min(),a.index.max()));hist=cal.shift(1).rolling(500,min_periods=120);std=(cal-hist.mean())/hist.std();std.index+=pd.Timedelta(days=1);t['x']=std.reindex(t.index);z=t.dropna(subset=['x','jump1']);yy=z.jump1.astype(float).to_numpy();xx=z.x.to_numpy()
  if len(z)>30 and len(np.unique(yy))==2:
   b,se,prob,ok=probit_fit(xx,yy,z.date);z=z.copy();z['fitted']=prob;z['oos_prob']=np.nan;z['train_label_end']=pd.NaT
   for start in range(365,len(z),60):
    test=z.iloc[start:start+60];train=z.iloc[:start];train=train[train.target_end<=test.observed_at.iloc[0]-pd.Timedelta(days=1)]
    if train.jump1.nunique()<2:continue
    bb,_,_,converged=probit_fit(train.x,train.jump1,train.date)
    if converged:
     z.loc[test.index,'oos_prob']=special.ndtr(bb[0]+bb[1]*test.x);z.loc[test.index,'train_label_end']=train.target_end.max()
   good=z.oos_prob.notna();add('probit.csv',dict(currency=c,tenor=1,variant=variant,window=window,component='tail',classification='事后标签/前一日09:00信息',n=len(z),coef=b[1],se=se[1],t=b[1]/se[1],p=2*stats.norm.sf(abs(b[1]/se[1])),auc_in_sample=auc(yy,prob),auc_oos=auc(z.loc[good,'jump1'],z.loc[good,'oos_prob']),oos_n=int(good.sum()),converged=ok));z.to_csv(O/f'probit_path_{c}_{variant}_{window}.csv',index=False)
   add('lookahead_checks.csv',dict(currency=c,variant=variant,window=window,check='OOS_training_labels_complete_at_previous_origin',n=int(good.sum()),failures=int((z.loc[good,'train_label_end']>z.loc[good,'observed_at']-pd.Timedelta(days=1)).sum())))
  audit=t.dropna(subset=['prior_last_trade']);add('lookahead_checks.csv',dict(currency=c,variant=variant,window=window,check='previous_day_quote_pre_origin',n=len(audit),failures=int((audit.prior_last_trade>=(audit.observed_at-pd.Timedelta(days=1)).astype('int64')/1e6).sum())))
 log(f'{c} 事后跳跃归因与事前概率单位完成')


def calendar_stats(p):
 for (variant,h),g in p[(p.window=='fixed')&p.variant.isin(['trade_R5total','mark_R5total','trade_native','mark_native'])].groupby(['variant','tenor']):
  s=premium(g).sort_values('date');a=s.set_index('date').tail_I.copy();a.index+=pd.Timedelta(days=1);s['prior_tail_I']=a.reindex(s.date).to_numpy();s['pre_event_change']=s.tail_I-s.prior_tail_I
  for category in ['CPI','NFP','FOMC','weekend','rest','CPI_known','NFP_known','FOMC_known']:
   z=s[s[category]]
   labels=dict(currency=s.currency.iloc[0],tenor=h,variant=variant+'_bv',window='fixed',split=category,classification='窗口覆盖/事前支持子样本' if category.endswith('known') else '窗口覆盖')
   metrics(z,'calendar_premium.csv',[('vrp_ATM_minus_RV','atm_raw','atm'),('vrp','atm','continuous'),('vrp','tail','tail'),('vrp','total','total'),('implied_change_previous_day','pre_event_change','tail')],False,**labels)
   if category not in ['rest']:
    r=contrast(s,'tail',s[category],s.rest,h);add('calendar_tests.csv',dict(currency=s.currency.iloc[0],tenor=h,variant=variant+'_bv',component='tail',category=category,reference='rest',alternative='less' if category=='weekend' else 'greater',**r))
 log(f'{p.currency.iloc[0]} 公告与周末统计完成')

def strategies(c,rv):
 d=pd.read_parquet(R5/f'strategy_path_{c}.parquet');d=d[(d.variant=='last_binance_entry_IV_hourly')&(d.side=='sell')&(d.bps==0)&(d.spread=='none')].copy();d['observed_at']=d.entry;d['target_end']=d.max_expiry;d['prior_macro_R5']=d.macro;d['prior_weekend_R5']=d.weekend;d=event_flags(d);one=rv[(rv.variant=='rv5')&(rv.window=='expiry23')&(rv.tenor==1)][['date','jump1','jump5','jump_rule']];d=d.merge(one,on='date',how='left',validate='one_to_one');d['component']='total';d.to_csv(O/f'strategy_path_{c}.csv',index=False);worst=d.nsmallest(10,'usd_10000')
 availability=pd.read_csv(R5/f'strategy_availability_{c}.csv',parse_dates=['date']).fillna('');bad=availability[availability.entry_mode.eq('last')]
 for col,value in [('tool','binance'),('vol','entry_IV'),('frequency','hourly')]:bad=bad[bad[col].eq(value)|bad[col].eq('')]
 idx=pd.date_range(d.date.min(),d.date.max()).difference(pd.DatetimeIndex(bad.date));calendar=pd.DataFrame({'date':idx});calendar['observed_at']=calendar.date+pd.Timedelta(hours=9);calendar['target_end']=calendar.date+pd.Timedelta(days=1,hours=8);calendar=event_flags(calendar);calendar=calendar.merge(one,on='date',how='left',validate='one_to_one');values=d.set_index('date');calendar['usd_10000']=calendar.date.map(values.usd_10000).fillna(0);calendar.to_csv(O/f'strategy_calendar_{c}.csv',index=False)
 for category,mask in [('all',np.ones(len(calendar),bool)),('jump',calendar.jump1.eq(True)),('nonjump',calendar.jump1.eq(False))]+[(k,calendar[k]) for k in ['CPI','NFP','FOMC','weekend','rest']]:
  cal=calendar[mask];z=d[d.date.isin(cal.date)]
  if cal.empty:continue
  cs=cal.usd_10000.cumsum();draw=cs-cs.cummax().clip(lower=0)
  row=dict(currency=c,split=category,variant='last_binance_entry_IV_hourly',tenor=1,frequency='hourly',bps=0,spread='none',side='sell',selection='all',component='total',classification='事后' if category in ['jump','nonjump'] else '窗口覆盖',**summary(ts(cal,'usd_10000'),1,False),trade_n=len(z),trade_mean=z.usd_10000.mean(),trade_median=z.usd_10000.median(),trade_low=np.nan,trade_high=np.nan,max_actual_days=23/24,annual_sharpe=np.sqrt(365)*cal.usd_10000.mean()/cal.usd_10000.std(),worst_day=z.usd_10000.min(),skew=stats.skew(cal.usd_10000,bias=False),max_drawdown_usd=-draw.min(),worst10_jump_count=int(worst.jump1.eq(True).sum()),worst10_n=len(worst))
  trade=summary(ts(z,'usd_10000'),1,False);row['trade_low']=trade.get('ci_low',np.nan);row['trade_high']=trade.get('ci_high',np.nan);row['boot_low']=np.nan;row['boot_high']=np.nan
  for col in ['premium','gamma','funding','transaction','halfspread','bridge_residual']:row[col+'_mean']=z[col].sum(min_count=1)/len(cal)
  add('strategy_summary.csv',row)
 old=pd.read_csv(R5/'strategy_summary.csv');old=old[(old.currency==c)&(old.variant=='last_binance_entry_IV_hourly')&(old.side=='sell')&(old.bps==0)&(old.spread=='none')&(old.split=='all')]
 if len(old):
  old=old.iloc[0];add('strategy_reconciliation.csv',dict(currency=c,variant='last_binance_entry_IV_hourly',component='total',active_mean=d.usd_10000.mean(),R5_trade_mean=old.trade_mean,mean_error=d.usd_10000.mean()-old.trade_mean,calendar_mean=calendar.usd_10000.mean(),R5_calendar_mean=old['mean'],calendar_mean_error=calendar.usd_10000.mean()-old['mean'],calendar_n=len(calendar),R5_n=int(old.n),active_n=len(d),R5_trade_n=int(old.trade_n),macro_flag_disagreements=int((d.macro!=d.prior_macro_R5).sum()),weekend_flag_disagreements=int((d.weekend!=d.prior_weekend_R5).sum())))
 log(f'{c} 策略损益归因完成')

def robustness(p,rv):
 c=p.currency.iloc[0];base=p[(p.variant=='mark_native')&(p.window=='fixed')&(p.tenor==1)].copy()
 for estimator in ['bv','trv','medrv','bv_raw']:premium_rows(premium(base,estimator),'robustness.csv','mark_native_'+estimator)
 for v in ['mark_band02','mark_band10','mark_cover15','mark_strikes8','mark_last15']:
  g=p[(p.variant==v)&(p.window=='fixed')&(p.tenor==1)]
  if len(g):premium_rows(premium(g),'robustness.csv',v+'_bv')
  else:add('robustness.csv',dict(currency=c,tenor=1,split='all',variant=v+'_bv',component='total',metric='vrp',n=0,status='no_eligible_quotes'))
 for rvariant in ['subsample','rv15']:
  rr=rv[(rv.variant==rvariant)&(rv.window=='fixed')&(rv.tenor==1)][['date','rv','bv','trv','medrv']];g=base.drop(columns=['rv','bv','trv','medrv']).merge(rr,on='date',how='left',validate='one_to_one');premium_rows(premium(g),'robustness.csv','mark_native_'+rvariant+'_bv')
 for split,mask in [('2020-2021',base.date.dt.year.between(2020,2021)),('2022-2023',base.date.dt.year.between(2022,2023)),('2024-2026',base.date.dt.year.between(2024,2026)),('ETF_pre',base.date<'2024-01-11'),('ETF_post',base.date>='2024-01-11')]:premium_rows(premium(base[mask]),'robustness.csv','mark_native_bv',split)
 log(f'{c} 一天稳健性完成')

if __name__=='__main__':
 import argparse
 ap=argparse.ArgumentParser();ap.add_argument('--currency',choices=['BTC','ETH']);a=ap.parse_args()
 for c in [a.currency] if a.currency else ['BTC','ETH']:
  p,rv=build_panel(c);jump_statistics(c,rv);implied_stats(p);term_stats(p);attribution(p);calendar_stats(p);strategies(c,rv);robustness(p,rv)
 for name,rows in ROWS.items():save(rows,name if not a.currency else name.replace('.csv','_'+a.currency+'.csv'))
 ev=pd.read_csv(ROOT/'out/R3/shared/events_publication.csv');source=[]
 for rel,z in ev.groupby('source_file'):
  f=ROOT/rel;source.append(dict(path=rel,source_url=z.source_url.iloc[0],exists=f.exists(),sha256=sha(f) if f.exists() else '',events=len(z),ex_ante_supported=int(z.ex_ante_supported.eq(True).sum())))
 save(source,'macro_source_receipts.csv');save(ev,'macro_events.csv');log('所有统计表已保存')
