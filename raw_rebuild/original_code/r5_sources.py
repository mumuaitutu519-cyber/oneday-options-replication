from r5_common import *
import requests
import re,html
SOURCES={
 'deribit_fees':'https://support.deribit.com/hc/en-us/articles/25944746248989-Fees',
 'deribit_options':'https://support.deribit.com/hc/en-us/articles/31424939096093-Inverse-Options',
 'binance_fee_table':'https://www.binance.com/en/fee/futureFee',
 'binance_fee_formula':'https://www.binance.com/en/support/faq/detail/360033544231',
 'realized_kernel':'https://public.econ.duke.edu/~get/browse/courses/201/spr12/DOWNLOADS/MicroStructure/bhls_kernels_practice_08.pdf'}
def main():
 raw=OUT/'receipts';raw.mkdir(exist_ok=True);rows=[]
 for name,url in SOURCES.items():
  dest=raw/(name+('.pdf' if url.endswith('.pdf') else '.html'))
  if dest.exists():continue
  utc=datetime.datetime.now(datetime.timezone.utc).isoformat()
  try:
   r=requests.get(url,timeout=25,headers={'User-Agent':'Mozilla/5.0'});dest.write_bytes(r.content)
   if not url.endswith('.pdf'):(raw/(name+'.txt')).write_text(html.unescape(re.sub('<[^>]+>','\n',r.text)))
   rows.append(dict(source=name,url=url,retrieved_at_utc=utc,status_code=r.status_code,bytes=len(r.content),sha256=sha(dest),receipt=str(dest.relative_to(ROOT)),transport='requests',error=''))
  except Exception as e:rows.append(dict(source=name,url=url,retrieved_at_utc=utc,status_code=None,bytes=0,sha256='',receipt='',transport='requests',error=type(e).__name__+': '+str(e)))
 p=CHEN/'source_receipts.csv'
 if p.exists():rows=pd.concat([pd.read_csv(p),pd.DataFrame(rows)],ignore_index=True)
 csv(rows,'source_receipts.csv')
if __name__=='__main__':main()
