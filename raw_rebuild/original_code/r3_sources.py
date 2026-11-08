#!/usr/bin/env python3
from r3_common import *
import requests,re,html,concurrent.futures
from zoneinfo import ZoneInfo
from urllib.parse import urljoin
RAW=OUT/'shared/raw'
def fetch(label,url):
 p=RAW/(label+'.html')
 if p.exists():return p.read_text(errors='replace')
 if label.startswith('cme_') and (RAW/'cme_browser.txt').exists():return (RAW/'cme_browser.txt').read_text()
 try:r=requests.get(url,timeout=8);p.write_bytes(r.content);status=r.status_code
 except Exception as e:log('源下载失败 '+label+' '+repr(e));return ''
 with (OUT/'shared/source_requests.jsonl').open('a') as f:f.write(json.dumps(dict(label=label,url=url,status=status,path=str(p.relative_to(ROOT)),sha256=sha(p),accessed=datetime.datetime.now(datetime.timezone.utc).isoformat()))+'\n')
 return r.text if status==200 else ''
def stripped(s):return html.unescape(re.sub('<[^>]+>',' ',s))
def main():
 sources=[('cme_launch','https://www.cmegroup.com/media-room/press-releases/2026/6/01/cme_group_announceslaunchof247cryptocurrencyfuturesandoptionstra.html'),('cme_announcement','https://www.cmegroup.com/media-room/press-releases/2026/2/19/cme_group_to_launch247cryptocurrencyfuturesandoptionstradingonma.html'),('cme_hours','https://www.cmegroup.com/notices/electronic-trading/2026/05/20260525.html'),('cme_maintenance','https://www.cmegroup.com/notices/risk-management-and-monitoring/2026/05/20260521.html'),('cme_btc','https://www.cmegroup.com/notices/clearing/2017/12/Chadv17-468.html'),('cme_eth','https://www.cmegroup.com/notices/clearing/2021/01/Chadv21-028.html'),('cme_sol','https://www.cmegroup.com/notices/clearing/2025/02/25-075.html'),('cme_xrp','https://www.cmegroup.com/notices/clearing/2025/04/25-166.html'),('cme_ada_link','https://www.cmegroup.com/media-room/press-releases/2026/2/11/cme_group_announcesfirsttradesfornewcardanochainlinkandstellarcr.html'),('cme_avax','https://www.cmegroup.com/newsletters/quarterly-cryptocurrencies-report/2026-q2-cryptocurrency-highlights.html'),('cme_product_list','https://www.cmegroup.com/markets/cryptocurrencies/cryptocurrency-futures'),('binance_bracket','https://fapi.binance.com/fapi/v1/leverageBracket'),('binance_margin_help','https://www.binance.com/en/support/faq/detail/360033162192')]
 for label,url in sources:fetch(label,url)
 launch={'BTC':('2017-12-18','cme_btc'),'ETH':('2021-02-08','cme_eth'),'SOL':('2025-03-17','cme_sol'),'XRP':('2025-05-19','cme_xrp'),'ADA':('2026-02-09','cme_ada_link'),'LINK':('2026-02-09','cme_ada_link'),'AVAX':('2026-05-04','cme_avax')}
 csv([dict(asset=a,treated=a in launch,launch_date=launch[a][0] if a in launch else '',source_url=dict(sources)[launch[a][1]] if a in launch else dict(sources)['cme_product_list']) for a in ASSETS],'yin/treatment.csv')
 csv([dict(item='effective',local='2026-05-29 16:00 America/Chicago',utc='2026-05-29 21:00 UTC',url=dict(sources)['cme_hours']),dict(item='first_day_maintenance',local='2026-05-29 16:00-16:30 CT',utc='2026-05-29 21:00-21:30 UTC',url=dict(sources)['cme_maintenance']),dict(item='weekly_maintenance',local='Saturday 02:00-04:00 America/Chicago',utc='Saturday 07:00-09:00 summer;08:00-10:00 winter',url=dict(sources)['cme_maintenance']),dict(item='announcement',local='2026-02-19',utc='',url=dict(sources)['cme_announcement'])],'yin/institution.csv')
 # BLS browser raw captures: actual dates and all precise time text are retained.
 events=[];availability=[]
 for year in range(2020,2027):
  p=RAW/f'bls_{year}_browser.txt'
  if not p.exists():continue
  raw=p.read_text();found={}
  pattern=r'L\d+:\s*(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s*([A-Za-z]+\s+\d{1,2},\s+\d{4})\s*(?:\|\s*)?(\d{2}:\d{2}\s*[AP]M)\s*(?:\|\s*)?(Consumer Price Index|Employment Situation) for[^\n]*'
  for m in re.finditer(pattern,raw):
   typ='CPI' if m[4]=='Consumer Price Index' else 'NFP';local=pd.Timestamp(m[2]+' '+m[3]).tz_localize('America/New_York');utc=local.tz_convert('UTC');found[(typ,str(utc))]=dict(type=typ,datetime_utc=str(utc),datetime_local=str(local),year=year,source_url=f'https://www.bls.gov/schedule/{year}/home.htm',source_file=str(p.relative_to(ROOT)),scheduled=True,known_at='',predictive_calendar='retrospective_page_without_origin_snapshot')
  events.extend(found.values());availability.append(dict(provider='BLS',year=year,cpi=sum(k[0]=='CPI' for k in found),nfp=sum(k[0]=='NFP' for k in found),status='parsed' if found else 'unavailable'))
 # Current release-specific pages supplement an empty annual page.
 p=RAW/'bls_2026_release_late_browser.txt'
 if p.exists():
  typ='CPI'
  for line in p.read_text().splitlines():
   if '## Schedule of Releases for the Consumer Price Index' in line:typ='CPI'
   if '## Schedule of Releases for the Employment Situation' in line:typ='NFP'
   m=re.search(r'L\d+:.*?([A-Z][a-z]{2}\.?\s+\d{1,2},\s+2026)\s+(\d{2}:\d{2}\s+[AP]M)',line)
   if m:
    local=pd.Timestamp(m[1].replace('.','')+' '+m[2]).tz_localize('America/New_York');utc=local.tz_convert('UTC')
    if utc<pd.Timestamp('2026-10-07',tz='UTC'):events.append(dict(type=typ,datetime_utc=str(utc),datetime_local=str(local),year=2026,source_url='https://www.bls.gov/schedule/news_release/'+('cpi.htm' if typ=='CPI' else 'empsit.htm'),source_file=str(p.relative_to(ROOT)),scheduled=True,known_at='',predictive_calendar='retrospective_page_without_origin_snapshot'))
 # Statement links supply actual release dates, unlike the first meeting day.
 urls=[]
 for year in range(2020,2027):
  url=f'https://www.federalreserve.gov/monetarypolicy/fomchistorical{year}.htm' if year<=2020 else 'https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm'
  raw=fetch('fed_calendar_'+str(year),url)
  for token in re.findall(r'(?:href=["\'])([^"\']*monetary'+str(year)+r'\d{4}a\.htm)["\']',raw):urls.append(urljoin('https://www.federalreserve.gov',token))
 raw=fetch('fed_news2021','https://www.federalreserve.gov/newsevents/2021-monetary.htm')
 for token in re.findall(r'href="([^"]*monetary2021\d{4}a\.htm)"',raw):urls.append(urljoin('https://www.federalreserve.gov',token))
 urls=sorted(set(urls))
 for url in urls:
  token=re.search(r'monetary(\d{8})',url)[1];date=pd.Timestamp(token);raw=fetch('fed_statement_'+token,url);txt=' '.join(stripped(raw).split())
  m=re.search(r'For release at\s*(\d{1,2}):(\d{2})\s*([ap])\.?m\.?\s*(E[SD]T)',txt,re.I)
  if not m:availability.append(dict(provider='FOMC',year=date.year,statement=token,status='exact_time_missing',source_url=url));continue
  hour=int(m[1])%12+(12 if m[3].lower()=='p' else 0);local=(date+pd.Timedelta(hours=hour,minutes=int(m[2]))).tz_localize('America/New_York');utc=local.tz_convert('UTC')
  if utc>pd.Timestamp('2026-10-07',tz='UTC'):continue
  standard='Federal Reserve issues FOMC statement' in txt;emergency=token in ['20200303','20200315'];
  if not standard:availability.append(dict(provider='FOMC',year=date.year,statement=token,status='non_rate_statement',source_url=url));continue
  events.append(dict(type='FOMC',datetime_utc=str(utc),datetime_local=str(local),year=date.year,source_url=url,source_file=str((RAW/('fed_statement_'+token+'.html')).relative_to(ROOT)),scheduled=not emergency,known_at='',predictive_calendar='retrospective_statement_without_origin_snapshot',release_tz=m[4],dst_matches=m[4]==local.tzname()))
 csv(pd.DataFrame(events).drop_duplicates(['type','datetime_utc']).sort_values('datetime_utc'),'shared/events.csv');csv(availability,'shared/calendar_availability.csv');log('官方源与宏观事件表完成 '+str(len(events)))
if __name__=='__main__':main()
