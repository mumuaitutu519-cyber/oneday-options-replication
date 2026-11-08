from r2_common import *
import requests
SCHEDULES={
 '20200702a':('2020-07-02 10:00', {2021:['01-27','03-17','04-28','06-16','07-28','09-22','11-03','12-15']}),
 '20210604a':('2021-06-04 11:00', {2022:['01-26','03-16','05-04','06-15','07-27','09-21','11-02','12-14']}),
 '20220624a':('2022-06-24 14:00', {2023:['02-01','03-22','05-03','06-14','07-26','09-20','11-01','12-13']}),
 '20230623a':('2023-06-23 11:30', {2024:['01-31','03-20','05-01','06-12','07-31','09-18','11-07','12-18']}),
 '20240809a':('2024-08-09 13:30', {2025:['01-29','03-19','05-07','06-18','07-30','09-17','10-29','12-10'],2026:['01-28','03-18','04-29','06-17','07-29','09-16','10-28','12-09']})}
def prepare_events(offline=False):
 folder=OUT/'raw/calendars';folder.mkdir(parents=True,exist_ok=True);rows=[];receipts=[]
 for key,(publication,years) in SCHEDULES.items():
  url='https://www.federalreserve.gov/newsevents/pressreleases/monetary'+key+'.htm';p=folder/('fomc_schedule_'+key+'.html')
  if not p.exists() and not offline:
   try:
    r=requests.get(url,timeout=35);p.write_bytes(r.content);status=r.status_code
   except Exception as e:receipts.append(dict(url=url,status='failed',reason=repr(e)));continue
  else:status='cached' if p.exists() else 'missing'
  valid=p.exists() and 'tentative meeting schedule' in p.read_text(errors='replace').lower()
  receipts.append(dict(url=url,status=status,sha256=sha(p) if p.exists() else '',valid_schedule=valid))
  if not valid:continue
  pub=pd.Timestamp(publication,tz='America/New_York').tz_convert('UTC')
  for year,dates in years.items():
   for date in dates:
    ts=pd.Timestamp(str(year)+'-'+date+' 14:00',tz='America/New_York').tz_convert('UTC')
    rows.append(dict(event='FOMC',timestamp=ts.tz_localize(None),event_ms=ts.value//10**6,publication=pub.tz_localize(None),publication_ms=pub.value//10**6,url=url,source_sha256=sha(p)))
 csv(receipts,'event_calendar_receipts.csv');events=csv(rows,'scheduled_events.csv')
 csv([dict(version=4,status='estimated_FOMC_only' if len(events) else 'not_estimable',reason='已存提前公布的2021—2026联储议息日程；CPI及非农的BLS官方日历请求403，按指令跳过',source='scheduled_events.csv;event_calendar_receipts.csv;official_calendar_sources.csv',events=len(events))],'event_clock_availability.csv')
 return events
