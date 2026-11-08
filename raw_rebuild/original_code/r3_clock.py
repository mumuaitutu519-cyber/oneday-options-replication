from r3_common import *
from r2_clock import Clock as BaseClock
class Clock(BaseClock):
 def __init__(self,assets=None):
  configs=['v3_w1000_trim0_hol0_lag0','v3_w365_trim0_hol0_lag0','v3_w1000_trim1_hol0_lag0']
  path=OUT/'shared/clock_profiles.parquet'
  if not path.exists():path=ROOT/'out/R2/clock_profiles.parquet'
  self.frame=pd.read_parquet(path,columns=['asset','config','date']+[f'w{j}' for j in range(168)],filters=[('asset','in',assets or ASSETS),('config','in',configs)])
  self.lookup={}
  for (asset,cfg),g in self.frame.groupby(['asset','config']):
   g=g.sort_values('date');self.lookup[asset,cfg]=(g.date.to_numpy('datetime64[D]').astype('int64'),g[[f'w{j}' for j in range(168)]].to_numpy(float))
  self.hs=np.array([],dtype='int64')
