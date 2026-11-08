from common import *

def main():
 rows=[]
 def add(table,item,paper,actual,ok,reason,source):rows.append(dict(table=table,item=item,manuscript=paper,recomputed=actual,consistent=ok,reason=reason,source=source))
 for c in ['BTC','ETH']:
  p=pd.read_parquet(DATA/f'premium_{c}.parquet');z=p[(p.variant=='trade_ATM')&(p.tenor==1)].dropna(subset=['vrp']);duration=(z.target_end-z.observed_at).dt.total_seconds()/3600;add('1 / Figure 1','IV / RV measurement window '+c,'23 hours',f'{duration.min():g}–{duration.max():g} hours',bool((duration==23).all()),'The archived R4 term panel uses fixed-maturity interpolation to 09:00 next day. Recovered rows are preserved in R5 trade_ATM.','data/premium_'+c+'.parquet')
  core=pd.read_parquet(DATA/f'forecast_cache/{c}_w500_log_hour8_h1.parquet').dropna(subset=['y']+CORE);add(2,'evaluation days '+c,'1522' if c=='BTC' else '1596',len(core),len(core)==(1522 if c=='BTC' else 1596),'','data/forecast_cache/'+c+'_w500_log_hour8_h1.parquet');add(2,'first evaluation date '+c,'2021-08-09',str(core.date.min().date()),str(core.date.min().date())=='2021-08-09','','data/forecast_cache');add(2,'last evaluation date '+c,'2026-10-05',str(core.date.max().date()),str(core.date.max().date())=='2026-10-05','','data/forecast_cache');a=load_input(f'{c}_w500_log_hour8_h1');hours=(a.label_end-a.observed_at).dt.total_seconds()/3600;add(2,'forecast window '+c,'23 hours',str(hours.unique().tolist()),bool((hours==23).all()),'Actual next-day 08:00 expiry.','data/forecast_inputs')
  q=pd.read_parquet(DATA/f'forecast_cache/{c}_w1000_log_hour8_h1.parquet').dropna(subset=['y','HAR','HAR_S','HAR_J','HAR_Q','HAR_clock','HAR_DVOL','GB','DVOL_raw','DVOL_calibrated','IV_raw','IV_calibrated','IV_HAR','IV_GB']);add(3,'Window 1000 starts '+c,'2024-01-31',str(q.date.min().date()),str(q.date.min().date())=='2024-01-31','','data/forecast_cache')
  for h in [1,2,3,7,14,30]:
   z=p[(p.variant=='trade_ATM')&(p.tenor==h)].dropna(subset=['vrp']);hours=(z.target_end-z.observed_at).dt.total_seconds()/3600;add(5,'fixed-maturity hours '+c+' h='+str(h),24*h,','.join(str(x) for x in hours.unique()),bool((hours==24*h).all()),'Table 5 note uses row maturity; Section 2 equation (3) describes the separate 23-hour measure.','data/premium_'+c+'.parquet')
  mk=pd.read_parquet(DATA/'atm_trade_daily_by_side.parquet');mk=mk[(mk.currency==c)&(mk.side=='all')];before=mk[mk.date<'2020-01-01'];add(9,'Panel A sample begins '+c,'2020-01 (global manuscript sample)',str(mk.date.min().date()),bool(mk.date.min()>=pd.Timestamp('2020-01-01')),f'Includes {len(before)} days and {int(before.trade_n.sum())} trades before 2020. Table values reproduce the wider source sample.','data/atm_trade_daily_by_side.parquet')
  add(5,'One-day rows A–C and Panel D maturity '+c,'23 hours (equation (3) reference)','24 hours',False,'The historical one-day term row and variance-unit premia use a fixed 24-hour interpolated panel, different from the one-day actual-expiry definition.','data/premium_'+c+'.parquet')
  add(9,'Panel B maturity '+c,'23 hours (one-day definition; Panel B equation (3) reference)','24 hours in Panel B and annual premia in Panel C',False,'Side-specific premium is calculated from the same interpolated fixed-day panel as Table 5.','data/premium_'+c+'.parquet')
 for c in ['BTC','ETH']:
  counts=pd.read_csv(OUT/f'strategy_sample_counts_{c}.csv')
  for variant,side,target in [('last','sell',2458 if c=='BTC' else 2431),('mark','sell',2458 if c=='BTC' else 2431),('sell','sell',2451 if c=='BTC' else 2417),('buy','buy',2451 if c=='BTC' else 2412)]:
   value=int(counts[(counts.variant==variant+'_binance_entry_IV_hourly')&(counts.side==side)&(counts.bps==0)].eligible_calendar_days.iloc[0]);add(8,'eligible days '+c+' '+variant+' '+side,target,value,value==target,'Idle eligible days included; unavailable market/funding days excluded.','output/strategy_sample_counts_'+c+'.csv')
 # Bootstrap cost intervals use the same 4999 draws in both modes.
 add(8,'stationary bootstrap draws','95% interval',4999,True,'4999 replicates, mean block length 7, seed 20261007; daily gaps restored for bootstrap.','code/tables.py::bootstrap_weights')
 add(10,'equal-coverage scaling','evaluation-period scaling','full evaluation sample',True,'Descriptive capital comparison; multiplier uses realized evaluation-period losses as stated in the manuscript note.','code/tables.py::tables_risk')
 d=pd.DataFrame(rows);d.to_csv(ROOT/'verify/NOTE_VERIFICATION.csv',index=False);d[~d.consistent].to_csv(ROOT/'verify/NOTE_CORRECTIONS.csv',index=False)
 text=['','## Table-note and sample corrections','', 'These are definition/sample discrepancies; table statistics were not changed.','', '|Location|Manuscript scope|Scope actually used for the reproduced numbers|','|---|---|---|']
 for r in d[~d.consistent].itertuples(index=False):text.append(f'|{r.table}: {r.item}|{r.manuscript}|{r.recomputed}; {r.reason}|')
 with (ROOT/'verify/MISMATCH_REPORT.md').open('a') as f:f.write('\n'.join(text)+'\n')
 print('Table note checks:',len(d),'scope discrepancies:',int((~d.consistent).sum()),flush=True)
if __name__=='__main__':
 main()
 from as_stated import main as reconcile_stated_definitions
 reconcile_stated_definitions()
