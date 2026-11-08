#!/usr/bin/env python3
from r3_common import *
from r3_zi import features
import re

def main():
 iv={c:iv_daily(c) for c in ['BTC','ETH']};qa=[]
 for asset in ASSETS[2:]:
  path=OUT/f'zi/predictions_{asset}.parquet';p=pd.read_parquet(path)
  for (spec,h),g in p.groupby(['spec','horizon']):
   match=re.fullmatch(r'(rv5|rv_sub|rv_robust)_(log|level)_w(\d+)_lag(\d+)',spec);measure,form,window,lag=match.groups();window=int(window);lag=int(lag)
   d,y,xs=features(asset,measure,form,iv);x=xs['M1'].shift(lag);target=forward_mean(y,h);valid=np.isfinite(x.to_numpy()).all(1)&np.isfinite(target)&(target>0);last=np.maximum.accumulate(np.where(valid,np.arange(len(d)),-1));positions=d.index.get_indexer(g.date);end=positions-h+1;begin=np.maximum(0,end-window);use=last[end-1]
   if (use<begin).any():raise ValueError('Missing mature training label '+asset+'/'+spec)
   dates=d.index[use]+pd.Timedelta(days=h+1);p.loc[g.index,'latest_label_end']=dates.to_numpy();qa.append(dict(asset=asset,spec=spec,horizon=h,n=len(g),latest_maturity_public_days=float(((g.forecast_public_at.to_numpy()-dates.to_numpy())/np.timedelta64(1,'D')).min()),passed=bool((dates.to_numpy()<=g.forecast_public_at.to_numpy()).all())))
  p.to_parquet(path,index=False)
 csv(qa,'zi/training_label_audit.csv');log('子训练标签成熟时点从有效历史标签独立重建并核对')
if __name__=='__main__':main()
