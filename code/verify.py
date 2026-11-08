from common import *
import re,csv,html
from decimal import Decimal,ROUND_HALF_UP

def numbers(s):
 return re.findall(r'(?<![A-Za-z])[-+]?\d[\d,]*(?:\.\d+)?',s)
def rnd(value,display):
 places=len(display.split('.')[1]) if '.' in display else 0
 return Decimal(str(value)).quantize(Decimal(1).scaleb(-places),rounding=ROUND_HALF_UP)

def currency_label(b,r,c):
 if b==2:return 'BTC' if r<5 else 'ETH'
 if b in [3,7,10,16]:return 'BTC'
 if b in [4,8,11]:return 'ETH'
 if b==5:return 'BTC' if c<5 else 'ETH'
 if b==6:return 'BTC' if c<4 else 'ETH'
 if b in [9,12,14,15]:return 'BTC' if r<14 else 'ETH'
 if b==13:return 'BTC' if c<6 else 'ETH'
 if b==17:return 'BTC' if r<12 else 'ETH'
 if b in [18,20,21,22]:return 'BTC' if c<4 else 'ETH'
 if b==19:return 'BTC' if c==2 else 'ETH'
 if b==23:return 'BTC' if r<9 else 'ETH'
 if b in [24,25]:return 'BTC' if r<7 else 'ETH'
 return ''
def column_label(b,c):
 maps={2:{2:'Obs',3:'Mean',4:'SD',5:'Min',6:'P25',7:'P50',8:'P75',9:'Max'},3:{2:'Loss',3:'Gain over HAR',4:'Gain over strongest',5:'MCS 90%',6:'MCS 75%'},4:{2:'Loss',3:'Gain over HAR',4:'Gain over strongest',5:'MCS 90%',6:'MCS 75%'},5:{2:'IV-C vs HAR',3:'IV-C vs IV30',4:'IV+HAR vs GB',5:'IV-C vs HAR',6:'IV-C vs IV30',7:'IV+HAR vs GB'},6:{2:'IV',3:'IV+HAR',4:'IV',5:'IV+HAR'}}
 if b in maps:return maps[b].get(c,str(c))
 if b in [7,8]:return {2:'k=1',3:'k=2',4:'k=3',5:'k=7',6:'k=14',7:'k=30',8:'IV30',9:'Obs'}.get(c,str(c))
 if b in [9,10,11,12]:return {2:'Mean',3:'Median',4:'Positive',5:'Log ratio',6:'Obs'}.get(c,str(c))
 if b==13:return 'Mean' if c in [2,6] else 'Obs'
 if b in [14,15,16,17]:return {2:'Center',3:'Tail',4:'Total',5:'Obs'}.get(c,str(c))
 if b==18:return 'Seller' if c in [2,4] else 'Buyer'
 if b==19:return 'Value'
 if b==20:return 'Estimate' if c in [2,4] else 'Interval'
 if b in [21,22]:return 'Mean' if c in [2,4] else 'Trades' if b==21 else 'Obs'
 if b==23:return {2:'Seller share',3:'Trade minus mark',4:'Traded premium',5:'Mark premium',6:'Seller result'}.get(c,str(c))
 if b in [24,25]:return {2:'Exceed',3:'UC',4:'CC',5:'ES',6:'Margin'}.get(c,str(c))
 return str(c)

def main():
 blocks=json.loads((ROOT/'verify/manuscript_tables.json').read_text());calc={(r['block'],r['row'],r['column']):r for r in json.loads((OUT/'recalculated_cells.json').read_text())};rows=[];inventory=[];render=[]
 for b in blocks:
  prev='';display=[]
  for r,line in enumerate(b['rows']):
   label=line[0].strip() if line else ''
   if label:prev=label
   out=[]
   for c,s in enumerate(line):
    item=calc.get((b['block'],r,c));ns=numbers(s);structural=c<2 or (b['block'],r) in [(5,2),(6,2),(7,1),(8,1)] or b['table']==1 and b['block']==1 or (item is None and not re.fullmatch(r'[\s(\[+-]*\d[\d,.]*(?:[\s*,+-]+\d[\d,.]*)?[\s)\]*]*',s))
    if ns:
     for j,n in enumerate(ns):inventory.append(dict(table=b['table'],block=b['block'],row=r,column=c,token=j,manuscript_token=n,role='structure_or_definition' if structural else 'estimated_statistic',mapped=item is not None))
    if item is None:
     if ns and not structural and c>=2:
      for j,n in enumerate(ns):rows.append(dict(table=b['table'],panel=b['panel'],block=b['block'],row=prev,row_index=r,column=c,token=j,manuscript_value=n,recalculated_value='',difference='',consistent=False,status='UNMAPPED',source_result='',reason='No computed statistic mapped to manuscript cell'))
     out.append(s);continue
    v=item['value'];src=item['source']
    if isinstance(v,str):
     ok=s==v;rows.append(dict(table=b['table'],panel=b['panel'],block=b['block'],row=prev,row_index=r,column=c,token=0,manuscript_value=s,recalculated_value=v,difference='',consistent=ok,status='MATCH' if ok else 'MISMATCH',source_result=src,reason='' if ok else 'Model confidence-set membership differs'));out.append(v);continue
    vals=v if isinstance(v,list) else [v];pieces=[]
    if len(ns)!=len(vals):raise ValueError((b['block'],r,c,s,v))
    for j,(n,value) in enumerate(zip(ns,vals)):
     expected=Decimal(n.replace(',',''));got=rnd(value,n);ok=got==expected;reason=''
     if not ok:reason='Recalculation differs from manuscript at displayed precision; historical result and input provenance are retained. See MISMATCH_REPORT.md.'
     rows.append(dict(table=b['table'],panel=b['panel'],block=b['block'],row=prev,row_index=r,column=c,token=j,manuscript_value=n,recalculated_value=value,difference=float(Decimal(str(value))-expected),consistent=ok,status='MATCH' if ok else 'MISMATCH',source_result=src,reason=reason));pieces.append(format(got,'f'))
    txt='['+', '.join(pieces)+']' if isinstance(v,list) else '('+pieces[0]+')' if s.startswith('(') else pieces[0]
    if not isinstance(v,list) and not s.startswith('(') and item.get('p') is not None:
     p=item['p'];txt+='***' if p<.01 else '**' if p<.05 else '*' if p<.10 else ''
    out.append(txt)
   display.append(out)
  render.append(dict(**b,recalculated_rows=display))
 d=pd.DataFrame(rows);d['currency']=[currency_label(int(r.block),int(r.row_index),int(r.column)) for r in d.itertuples()];d['column_label']=[column_label(int(r.block),int(r.column)) for r in d.itertuples()];d['statistic_role']=['t statistic' if blocks[int(r.block)-1]['rows'][int(r.row_index)][int(r.column)].startswith('(') else 'value / interval endpoint' for r in d.itertuples()];d['kind']=d.manuscript_value.map(lambda x:'MCS text' if str(x) in ['Yes','No'] else 'numeric');d.to_csv(ROOT/'verify/VERIFICATION.csv',index=False);pd.DataFrame(inventory).to_csv(ROOT/'verify/ALL_TABLE_NUMERIC_TOKENS.csv',index=False)
 d[d.kind=='MCS text'].to_csv(ROOT/'verify/MCS_TEXT_VERIFICATION.csv',index=False)
 (OUT/'tables.json').write_text(json.dumps(render,ensure_ascii=False,indent=2));summary=d[d.kind=='numeric'].groupby('table').agg(checked=('consistent','size'),matched=('consistent','sum'));summary['rate']=summary.matched/summary.checked;summary.to_csv(ROOT/'verify/VERIFICATION_RATES.csv')
 mode=os.environ.get('REPLICATION_MODE','quick');lines=['# Verification summary','',f'Mode: {mode}; protocol: {os.environ.get('REPLICATION_PROTOCOL','historical_generating_version')}. The 2026-11-06 DOCX is the sole displayed-value authority. All 10 tables are rebuilt from analysis-ready observations. Numeric equality uses decimal rounding at each manuscript cell’s displayed precision. Confidence-set text is also checked separately; it is excluded from the numeric denominators in the table below. Structural numbers in headings, variable definitions, and row labels are inventoried separately.','', '|Table|Matched|Checked|Rate|','|---|---:|---:|---:|']
 for t,r in summary.iterrows():lines.append(f'|{t}|{int(r.matched)}|{int(r.checked)}|{r.rate:.2%}|')
 lines+=['',f'Unmapped estimated numbers: {int((d.status=="UNMAPPED").sum())}.','', 'Quick mode uses archived dated forecast and VaR quantile paths, recalculates losses, tests, regressions, confidence sets, premia, bootstrap intervals, and all figures. Full mode refits every rolling model and reconstructs VaR inputs, and uses those new forecasts for the same checks. Both modes use the supplied daily variance and straddle P&L inputs; raw extraction is a separate reference workflow.','', 'Table notes and sample definitions are checked separately in NOTE_VERIFICATION.csv. Figure data are saved in output/figure*_data.csv; manuscript images and regenerated images appear side by side in output/FIGURE_COMPARISON.html.','']
 (ROOT/'verify/VERIFICATION_SUMMARY.md').write_text('\n'.join(lines))
 bad=d[~d.consistent];report=['# Mismatch report','', 'No manuscript file was edited. No statistical estimate was changed to match a manuscript value.','', '|Table|Block / row / column / token|Manuscript|Recalculated at manuscript precision|Source result and keys|','|---|---|---:|---:|---|']
 for a in bad.itertuples(index=False):
  value='' if a.status=='UNMAPPED' else str(a.recalculated_value) if not isinstance(a.recalculated_value,(int,float)) else str(rnd(a.recalculated_value,str(a.manuscript_value)))
  report.append(f'|{a.table}|{a.block} / {a.row_index} {a.row} / {a.column} / {a.token}|{a.manuscript_value}|{value}|{a.source_result}|')
 if not len(bad):report+=['', 'All mapped manuscript numbers agree after rounding. There is no numerical correction list.']
 report+=['','## Version and sample facts','', 'ROUND10 is a separate variance-clock project and is not a source for these one-day-option tables. Tables 1–5 and 8–10 use ROUND4/5; Tables 6–7 use the grid-corrected ROUND9 decomposition. The original R4 term_panel and trade_mark intermediate files are no longer present at their recorded paths; the term panel was recovered from the preserved R5 trade_ATM rows, and trade/mark daily aggregates were recomputed from the original daily API trade archive. This package includes the recovered observations and the recovery code.','', 'Daily one-day descriptive/premium rows in Tables 1 and 5 use fixed 09:00-to-09:00 interpolated maturities in the original result pipeline; forecast inputs for Table 2 use the actual 23-hour next expiry. This definition difference is disclosed in NOTE_VERIFICATION.csv. No statistic has been recomputed under a different window and presented as the original number.','']
 (ROOT/'verify/MISMATCH_REPORT.md').write_text('\n'.join(report))
 starrows=[]
 for (b,r,c),item in calc.items():
  original=blocks[b-1]['rows'][r][c]
  if item.get('p') is not None:
   p=item['p'];new='***' if p<.01 else '**' if p<.05 else '*' if p<.10 else '';old=''.join(re.findall(r'\*+',original));starrows.append(dict(table=blocks[b-1]['table'],block=b,row=r,column=c,manuscript_stars=old,recalculated_stars=new,consistent=old==new))
 pd.DataFrame(starrows).to_csv(ROOT/'verify/SIGNIFICANCE_VERIFICATION.csv',index=False)
 for b in render:
  lines=[f'# Table {b["table"]} {b["panel"]}', '', f'Mode: {mode}. '+('Archived issued forecasts / VaR input paths are used; statistics and tests are recomputed. Use --full for rolling re-estimation.' if mode=='quick' and b['table'] in [2,3,4,10] else 'Statistics are recomputed from the supplied daily observations.'), '', '| '+' | '.join('C'+str(i) for i in range(max(map(len,b['recalculated_rows']))))+' |','|'+'---|'*max(map(len,b['recalculated_rows']))]
  for row in b['recalculated_rows']:lines.append('| '+' | '.join(x.replace('|','/') for x in row)+' |')
  (OUT/f'table{b["table"]}_block{b["block"]}.md').write_text('\n'.join(lines))
 for table in range(1,11):
  files=[OUT/f'table{table}_block{b["block"]}.md' for b in render if b['table']==table]
  (OUT/f'TABLE_{table}.md').write_text('\n\n'.join(f.read_text() for f in files))
  d[d.table==table].to_csv(OUT/f'TABLE_{table}.csv',index=False)
 print(summary.to_string(),flush=True);print('MISMATCHES',len(bad),'UNMAPPED',int((d.status=='UNMAPPED').sum()),flush=True)
if __name__=='__main__':main()
