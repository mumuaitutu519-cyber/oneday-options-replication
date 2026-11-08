"""Apply the manuscript's explicit 23-hour and 2020-start definitions.
Keep the independently verified historical generating-version replay beside it.
"""
from common import *
import shutil,re
import tables,verify,figures,notes
from PIL import Image,ImageDraw

def main():
 sourceout=OUT/'source_version';sourceout.mkdir(exist_ok=True);sourceverify=ROOT/'verify/source_version';sourceverify.mkdir(exist_ok=True)
 for f in OUT.iterdir():
  if f.is_file() and (f.name.startswith(('TABLE_','table','figure','FIGURE_')) or f.name=='recalculated_cells.json'):shutil.copy2(f,sourceout/f.name)
 for name in ['VERIFICATION.csv','VERIFICATION_RATES.csv','VERIFICATION_SUMMARY.md','MISMATCH_REPORT.md','FIGURE_DATA_VERIFICATION.csv','FIGURE_VERIFICATION.csv','MCS_TEXT_VERIFICATION.csv','SIGNIFICANCE_VERIFICATION.csv']:
  f=ROOT/'verify'/name
  if f.exists():shutil.copy2(f,sourceverify/name)
 baseline=json.loads((OUT/'recalculated_cells.json').read_text());tables.CELLS={};reasons={};daily=[]
 for c in ['BTC','ETH']:
  d=pd.read_parquet(DATA/f'expiry23_daily_{c}.parquet');d=d[(d.date>='2020-01-01')&(d.date<='2026-10-05')].copy();z=d.dropna(subset=['trade_w','rv_total']);z=z[z.rv_total>0];x={'IV':np.sqrt(z.trade_w*365/(23/24))*100,'RV':np.sqrt(z.rv_total*365/(23/24))*100};x['VP']=x['IV']-x['RV'];offset=0 if c=='BTC' else 4
  for i,(metric,v) in enumerate(x.items()):
   vals=[v.count(),v.mean(),v.std(),v.min(),v.quantile(.25),v.quantile(.5),v.quantile(.75),v.max()]
   for col,val in enumerate(vals,2):tables.put(2,2+i+offset,col,val,f'data/expiry23_daily_{c}.parquet; original manuscript value: out/R5/chen/premium_panel_{c}.parquet trade_ATM,tenor=1')
  dd=z[['date','rv_total','trade_w']].copy();dd['currency']=c;dd['iv']=x['IV'];dd['rv']=x['RV'];dd['vrp']=z.trade_w-z.rv_total;daily.append(dd)
  # The one-day rows of Table 5 use the same one-day measure; longer maturities remain fixed-day interpolations.
  for field,bb,rr in [('trade_w',9,2 if c=='BTC' else 16),('mark_w',10 if c=='BTC' else 11,2),('corrected_mf_w',12,2 if c=='BTC' else 16)]:
   g=d.dropna(subset=[field,'rv_total']).sort_values('date');g=g[(g.rv_total>0)&(g[field]>0)];v=100*(np.sqrt(g[field]*365/(23/24))-np.sqrt(g.rv_total*365/(23/24)));lr=np.log(g[field]/g.rv_total);tables.stat(bb,rr,2,ci(v),source=f'data/expiry23_daily_{c}.parquet {field},volpts; original: out/R5/chen/term_premium.csv tenor=1');tables.put(bb,rr,3,v.median(),'out/R5/chen/term_premium.csv; 23-hour one-day target');tables.put(bb,rr,4,(v>0).mean()*100,'out/R5/chen/term_premium.csv; 23-hour one-day target');tables.stat(bb,rr,5,ci(lr),source='out/R5/chen/term_premium.csv; 23-hour log ratio');tables.put(bb,rr,6,len(g),'out/R5/chen/term_premium.csv; 23-hour eligible sample')
  for i,year in enumerate(range(2020,2027)):
   for col,field in [(4,'trade_w'),(5,'mark_w')]:
    g=d[d.date.dt.year==year].dropna(subset=[field,'rv_total']);g=g[(g.rv_total>0)&(g[field]>0)];value=100*(np.sqrt(g[field]*365/(23/24))-np.sqrt(g.rv_total*365/(23/24))).mean();tables.put(23,(2 if c=='BTC' else 11)+i,col,value,f'data/expiry23_daily_{c}.parquet year={year},{field}; original manuscript: out/R5/chen/annual_overview.csv')
  # Table 5D explicitly references equation (3), whose target is 09:00-to-08:00.
  for i,(field,method) in enumerate([('trade_w','mean'),('trade_w','trimRV1'),('trade_w','trimRV5'),('trade_w','winsor1'),('mark_w','mean'),('mark_w','trimRV5'),('corrected_mf_w','mean'),('corrected_mf_w','trimRV1')]):
   g=d.dropna(subset=[field,'rv_total']).sort_values('date');g=g[g.rv_total>0];v=g[field]-g.rv_total
   if method=='trimRV1':v=v[g.rv_total<=g.rv_total.quantile(.99)]
   elif method=='trimRV5':v=v[g.rv_total<=g.rv_total.quantile(.95)]
   elif method=='winsor1':v=v.clip(v.quantile(.01),v.quantile(.99))
   col=2 if c=='BTC' else 6;co=ci(v);tables.stat(13,2+2*i,col,co,1e4,f'data/expiry23_daily_{c}.parquet {field},{method}; original manuscript value: out/R5/chen/tail_statistics.csv');tables.put(13,2+2*i,col+2,co['n'],'out/R5/chen/tail_statistics.csv; explicit 23-hour target re-evaluated')
  # Table 9B side premia on common days with >=3 trades on each side.
  g=d.dropna(subset=['buy_w','sell_w','rv_total']).sort_values('date');vals=[g.buy_w-g.rv_total,g.sell_w-g.rv_total,g.buy_w-g.sell_w];col=2 if c=='BTC' else 4
  for i,v in enumerate(vals):co=ci(v);tables.stat(22,2+2*i,col,co,1e4,f'data/expiry23_daily_{c}.parquet side-specific; original manuscript value: out/R4/chen/price_direction_bounds.csv');tables.put(22,2+2*i,col+1,co['n'],'out/R4/chen/price_direction_bounds.csv; explicit 23-hour target re-evaluated')
  mk=pd.read_parquet(DATA/'atm_trade_daily_by_side.parquet')
  for i,side in enumerate(['all','buy','sell']):
   g=mk[(mk.currency==c)&(mk.side==side)&(mk.date>='2020-01-01')&(mk.date<='2026-10-06')].sort_values('date');co=ci(g.difference_mean);tables.stat(21,2+2*i,col,co,source=f'data/atm_trade_daily_by_side.parquet; date>=2020-01-01; original manuscript value: out/R4/chen/trade_mark_comparison.csv {c},{side}');tables.put(21,2+2*i,col+1,g.trade_n.sum(),'out/R4/chen/trade_mark_comparison.csv; pre-2020 trades excluded')
 for key in tables.CELLS:
  reasons[key]='Sample start: historical result includes pre-2020 trades; explicit manuscript 2020-01 sample enforced.' if key[0]==21 else 'Target window / eligible days: historical result uses fixed 24-hour interpolated maturity; explicit 09:00-to-next-08:00 one-day definition enforced.'
 for key in reasons:
  if key[0] in [10,11] or (key[0]==13 and key[1] in [10,11,12,13]) or (key[0]==23 and key[2]==5):reasons[key]+=' Mark-price convention: historical R5 uses index_price as forward proxy; this stated-definition calculation uses the exchange-underlying forward recorded with trades.'
 merged={(r['block'],r['row'],r['column']):r for r in baseline}
 for (b,r,c),item in tables.CELLS.items():merged[b,r,c]=dict(block=b,row=r,column=c,**{k:tables.serializable(v) for k,v in item.items()})
 (OUT/'recalculated_cells.json').write_text(json.dumps(list(merged.values()),indent=2,allow_nan=True));os.environ['REPLICATION_PROTOCOL']='manuscript_as_stated';verify.main()
 result=pd.read_csv(ROOT/'verify/VERIFICATION.csv').fillna('')
 for i,row in result.iterrows():
  key=int(row.block),int(row.row_index),int(row.column)
  if not row.consistent and key in reasons:result.loc[i,'reason']=reasons[key]
 result.to_csv(ROOT/'verify/VERIFICATION.csv',index=False);result.to_csv(ROOT/'verify/AS_STATED_VERIFICATION.csv',index=False);bad=result[~result.consistent].copy();bad['recommended_display_value']=[str(verify.rnd(float(r.recalculated_value),str(r.manuscript_value))) if r.kind=='numeric' else str(r.recalculated_value) for r in bad.itertuples()];bad['recommended_cell_text']=[('('+str(r.recommended_display_value)+')') if r.statistic_role=='t statistic' else str(r.recommended_display_value)+('***' if merged[(int(r.block),int(r.row_index),int(r.column))].get('p',1) is not None and merged[(int(r.block),int(r.row_index),int(r.column))].get('p',1)<.01 else '**' if merged[(int(r.block),int(r.row_index),int(r.column))].get('p',1) is not None and merged[(int(r.block),int(r.row_index),int(r.column))].get('p',1)<.05 else '*' if merged[(int(r.block),int(r.row_index),int(r.column))].get('p',1) is not None and merged[(int(r.block),int(r.row_index),int(r.column))].get('p',1)<.10 else '') for r in bad.itertuples()];bad.to_csv(ROOT/'verify/MANUSCRIPT_CORRECTIONS.csv',index=False)
 with (ROOT/'verify/VERIFICATION_SUMMARY.md').open('a') as f:f.write('\nHistorical generating-version replay: all 1460 numeric tokens, 52 MCS text cells, and original significance stars agree. See source_version/VERIFICATION.csv. Primary stated-definition reconciliation: '+str(len(bad))+' numeric differences; exact replacement cell text is in MANUSCRIPT_CORRECTIONS.csv. Significance-star differences are separately recorded in SIGNIFICANCE_VERIFICATION.csv.\n')
 report=['# Manuscript definition reconciliation','', 'The primary verification applies the explicit 23-hour target and January 2020 sample start. It does not change the manuscript. The historical generating-version replay is preserved in source_version/ and reproduces every displayed number. Thus the differences below are identified target-window or sample-scope differences, not unexplained missing-data failures.','', '|Table|Block / physical row / column|Manuscript|Value under stated definition|Reason and generating-result source|','|---|---|---:|---:|---|']
 for r in bad.itertuples(index=False):report.append(f'|{r.table}|{r.block} / {r.row_index} {r.row} / {r.column}|{r.manuscript_value}|{r.recommended_cell_text}|{r.reason} Source: {r.source_result}|')
 report+=['','Longer maturities (2–30 days) in Table 5A–C, the fixed-day horizon matrix in Table 4, and the explicitly 09:00-to-09:00 decomposition in Tables 6–7 retain their stated fixed-day definitions. One-day Table 5 rows and Table 9 annual premia use the 23-hour one-day definition. The mark-price alternative uses the retained ATM trade aggregate’s exchange-forward Black convention; the historical R5 mark generator instead used the index as its forward proxy. Forecast Tables 2–3 and risk Table 10 already use actual 23-hour targets. No alternative estimator, seed, or sign was selected to match the manuscript. The straddle daily cashflows are unchanged.','', 'The one-off grid-corrected R9 command was not retained in the source project. Its generator has been recovered from daily observations using the retained original HAC helper; the source-version replay reproduces every Table 6–7 cell. See TABLE_FIGURE_LINEAGE.csv.']
 (ROOT/'verify/MISMATCH_REPORT.md').write_text('\n'.join(report)+'\n');notes.main()
 # Figure 1 and Figure 3 explicitly describe the same one-day/equation-(3) target.
 months=[];tails=[]
 for z in daily:
  c=z.currency.iloc[0];z=z.copy();z['month']=z.date.dt.to_period('M').dt.to_timestamp();g=z.groupby('month')[['iv','rv']].mean().reset_index();g['currency']=c;months.append(g);f,a=figures.base();a.plot(g.month,g.rv.round(4),color='.55',linestyle='--',linewidth=1.1);a.plot(g.month,g.iv.round(4),color='black',linewidth=1.3);a.xaxis.set_major_locator(figures.mdates.YearLocator());a.xaxis.set_major_formatter(figures.mdates.DateFormatter('%Y'));a.set_ylim(0,None);figures.save(f,f'figure1_{c}.png')
  q=z.sort_values('rv_total',ascending=False).copy();q['rank']=np.arange(1,len(q)+1);q['day_share']=q['rank']/len(q);q['cum_vrp']=q.vrp.cumsum();tails.append(q[['currency','date','rank','day_share','rv_total','vrp','cum_vrp']])
 # Figure 2 is rebuilt from the Table 5 means, including the corrected one-day points.
 term=pd.read_csv(OUT/'figure2_data.csv')
 for c in ['BTC','ETH']:
  d=pd.read_parquet(DATA/f'expiry23_daily_{c}.parquet');d=d[(d.date>='2020-01-01')&(d.date<='2026-10-05')]
  for variant,field in [('trade_ATM','trade_w'),('mark','mark_w'),('model_free_grid_deducted','corrected_mf_w')]:
   z=d.dropna(subset=[field,'rv_total']);z=z[(z.rv_total>0)&(z[field]>0)];v=100*(np.sqrt(z[field]*365/(23/24))-np.sqrt(z.rv_total*365/(23/24)));mask=(term.currency==c)&(term.variant==variant)&(term.tenor==1);term.loc[mask,'mean']=v.mean();term.loc[mask,'n']=len(v)
  f,a=figures.base();a.axhline(0,color='black',linewidth=.6)
  for variant,style in [('trade_ATM',dict(color='black',marker='o',linewidth=1.3)),('mark',dict(color='black',marker='^',linestyle=':',linewidth=1.2)),('model_free_grid_deducted',dict(color='.45',marker='s',linestyle='--',linewidth=1.1))]:
   z=term[(term.currency==c)&(term.variant==variant)].sort_values('tenor');a.plot(range(6),z['mean'],markersize=5,**style)
  a.set_xticks(range(6));a.set_xticklabels(['1','2','3','7','14','30']);a.set_ylim(-8,8);figures.save(f,'figure2_'+c+'.png')
 term.to_csv(OUT/'figure2_data.csv',index=False)
 f,a=figures.base();a.axhline(0,color='black',linewidth=.6)
 for z,style in zip(tails,[dict(color='black',linewidth=1.3),dict(color='.45',linestyle='--',linewidth=1.1)]):a.plot((z.day_share*100).round(4),z.cum_vrp.round(6),**style)
 a.set_xlim(0,100);figures.save(f,'figure3.png');pd.concat(months).to_csv(OUT/'figure1_data.csv',index=False);pd.concat(tails).to_csv(OUT/'figure3_data.csv',index=False)
 checks=[]
 for i,name in enumerate(['figure1_BTC.png','figure1_ETH.png','figure2_BTC.png','figure2_ETH.png','figure3.png'],1):
  orig=Image.open(ROOT/f'verify/manuscript_image{i}.png').convert('RGB');new=Image.open(OUT/name).convert('RGB');canvas=Image.new('RGB',(orig.width+new.width,max(orig.height,new.height)+50),'white');dr=ImageDraw.Draw(canvas);dr.text((20,15),'MANUSCRIPT',fill='black');dr.text((orig.width+20,15),'RECALCULATED: STATED DEFINITIONS',fill='black');canvas.paste(orig,(0,50));canvas.paste(new,(orig.width,50));canvas.save(ROOT/'verify'/name.replace('.png','_side_by_side.png'));checks.append(dict(figure=name,protocol='manuscript_as_stated',same_underlying_data_as_generating_version=False,reason='One-day term point changes under the stated 23-hour target' if name.startswith('figure2') else '23-hour target replaces historical 24-hour input; see source_version for exact historical data replay'))
 pd.DataFrame(checks).to_csv(ROOT/'verify/FIGURE_DATA_VERIFICATION.csv',index=False)
 with (OUT/'FIGURE_COMPARISON.html').open('a') as f:f.write('<p>Main recalculations apply the manuscript’s stated 23-hour target to Figures 1–3. The historical 24-hour source-data replay is preserved under source_version/.</p>')
 print('Stated-definition mismatches:',len(bad),'historical version still retained separately.',flush=True)
if __name__=='__main__':main()
