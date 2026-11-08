from r9_common import *

def receipt():
 files=[Path(__file__).resolve(),ROOT/'code/r9_common.py']
 for c in ['BTC','ETH']:
  files.extend([O/f'quotes_{c}.parquet',R5/f'prices_binance_perp_{c}.parquet',O/f'raw_source_manifest_{c}.csv',O/f'futures_corridor_{c}.parquet'])
  d=pd.read_parquet(O/f'futures_corridor_{c}.parquet');assert (d.futures_known_at<d.date+pd.Timedelta(hours=9)).all();q=pd.read_parquet(O/f'quotes_{c}.parquet');z=d.merge(q[['date','expiry','variant','total']],on=['date','expiry','variant'],validate='one_to_one');good=z.center_futures.notna();assert (z.loc[good,'center_futures']>=0).all() and (z.loc[good,'center_futures']<=z.loc[good,'total']+1e-12).all()
 (O/'corridor_receipt.json').write_text(json.dumps([dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in files],indent=2))

def main():
 cache=O/'corridor_receipt.json'
 if cache.exists() and os.environ.get('R9_REBUILD')!='1':
  records=json.loads(cache.read_text())
  if all((ROOT/r['path']).exists() and sha(ROOT/r['path'])==r['sha256'] for r in records):log('期货中心走廊输入和中间量哈希核对通过，复用完整积分面板。');return
 for c in ['BTC','ETH']:
  prices=pd.read_parquet(R5/f'prices_binance_perp_{c}.parquet').close;q=pd.read_parquet(O/f'quotes_{c}.parquet');q=q[q.variant.isin(['trade_native','mark_native'])];meta=pd.read_parquet(ROOT/f'data_clean/option_instruments_{c}.parquet');extra=ROOT/f'out/R2/instrument_supplements_{c}.parquet'
  if extra.exists():meta=pd.concat([meta,pd.read_parquet(extra)],ignore_index=True)
  meta=meta.drop_duplicates('instrument_name')[['instrument_name','strike','expiration_timestamp','option_type','creation_timestamp']];rows=[]
  for date,z in q.groupby('date'):
   origin=date+pd.Timedelta(hours=9);known=origin-pd.Timedelta(minutes=1);future=prices.get(known,np.nan)
   if not np.isfinite(future):
    for rr in z.itertuples():rows.append(dict(currency=c,date=date,expiry=rr.expiry,variant=rr.variant,center_futures=np.nan,F_futures=np.nan,futures_known_at=known,status='missing_cached_pre09_futures'))
    continue
   raw=pd.read_parquet(ROOT/f'out/R2/raw/{c}/{date:%Y-%m-%d}/trades.parquet');raw=raw[(raw.timestamp>=(date+pd.Timedelta(hours=8)).value/1e6)&(raw.timestamp<origin.value/1e6)&(raw.price>0)&(raw.index_price>0)].copy()
   for col in ['combo_id','combo_trade_id','block_trade_id','block_rfq_id']:
    if col in raw:raw=raw[raw[col].isna()|raw[col].isin(['',0,False])]
   raw=raw.merge(meta,on='instrument_name',how='inner',validate='many_to_one');raw=raw[(raw.creation_timestamp<=raw.timestamp)&(raw.expiration_timestamp>raw.timestamp)];raw['trade_F']=pd.to_numeric(raw.get('underlying_price',raw.index_price),errors='coerce');raw['trade_F']=raw.trade_F.where(raw.trade_F>0,raw.index_price)
   for rr in z.itertuples():
    g=raw[raw.expiration_timestamp==rr.expiry.value/1e6].copy();kind='trade' if rr.variant=='trade_native' else 'mark';F0=g.trade_F if kind=='trade' else g.index_price;usd=g['price' if kind=='trade' else 'mark_price']*g.index_price;intr=np.where(g.option_type=='call',np.maximum(F0-g.strike,0),np.maximum(g.strike-F0,0));cap=np.where(g.option_type=='call',F0,g.strike);g['usd']=usd;g=g[np.isfinite(usd)&(usd>intr)&(usd<cap)];g=g[((g.option_type=='put')&(g.strike<rr.F))|((g.option_type=='call')&(g.strike>=rr.F))];k=g.groupby('strike').usd.median();x=k.index.to_numpy(float)/rr.F;y=k.to_numpy()/rr.F;bound=future/rr.F*np.exp(np.array([-1,1])*np.sqrt(rr.atm));center=np.nan;status='futures_corridor_support_missing'
    if np.isfinite(rr.total) and len(x)>=5 and x.min()<=bound[0] and x.max()>=bound[1]:
     if not np.any(np.isclose(x,1,atol=1e-12)):
      pos=np.searchsorted(x,1);x=np.insert(x,pos,1.);y=np.insert(y,pos,black([1],[1],[rr.atm],['call'])[0])
     xx=np.concatenate([np.geomspace(x[j],x[j+1],17)[:-1] for j in range(len(x)-1)]+[x[-1:]]);cx=np.r_[bound[0],xx[(xx>bound[0])&(xx<bound[1])],bound[1]];cy=np.interp(cx,x,y);center=2*np.trapezoid(cy/cx**2,cx);status='estimated'
    rows.append(dict(currency=c,date=date,expiry=rr.expiry,variant=rr.variant,center_futures=center,F_futures=future,futures_known_at=known,status=status))
  d=pd.DataFrame(rows);d.to_parquet(O/f'futures_corridor_{c}.parquet',index=False);save(d.groupby(['variant','status']).size().reset_index(name='n'),f'futures_corridor_coverage_{c}.csv');assert (d.futures_known_at<d.date+pd.Timedelta(hours=9)).all();log(f'{c} 期货价中心走廊完成：{int(d.center_futures.notna().sum())}/{len(d)}到期桶')
 receipt()
if __name__=='__main__':
 if '--receipt' in sys.argv:receipt()
 else:main()
