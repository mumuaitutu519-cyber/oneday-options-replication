"""Summarize temporal eligibility without loading large raw archives."""
from common import *
def main():
 rows=[]
 for cfg in json.loads((DATA/'forecast_specs.json').read_text()):
  d=load_input(cfg['key']);origin=pd.to_datetime(d.observed_at);trade=pd.to_datetime(d.last_option_trade,unit='ms',errors='coerce');public=pd.to_datetime(d.dvol_public_at);row=dict(key=cfg['key'],input_rows=len(d),option_trade_after_origin=int((trade>origin).sum()),dvol_publication_after_origin=int((public>origin).sum()),training_audit_rows=0,training_labels_after_origin=0)
  f=WORK/f'forecast_audit_{cfg["key"]}.csv'
  if f.exists():
   a=pd.read_csv(f);row['training_audit_rows']=len(a);row['training_labels_after_origin']=int((pd.to_datetime(a.last_training_label)>pd.to_datetime(a.origin)).sum())
  rows.append(row)
 pd.DataFrame(rows).to_csv(ROOT/'verify/LOOKAHEAD_AUDIT.csv',index=False);
 from threadpoolctl import threadpool_info
 libraries=threadpool_info();limits=dict(workers=int(os.environ.get('REPLICATION_WORKERS','4')),arrow_cpu_threads=pa.cpu_count(),arrow_io_threads=pa.io_thread_count(),numeric_libraries=libraries)
 (ROOT/'verify/THREAD_LIMITS.json').write_text(json.dumps(limits,indent=2))
 assert all(x['num_threads']==1 for x in libraries),libraries
 print('Time-cutoff audit complete:',len(rows),'specifications',flush=True)
if __name__=='__main__':main()
