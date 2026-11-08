from common import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from PIL import Image,ImageOps,ImageDraw
plt.rcParams.update({'font.family':'DejaVu Serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'axes.unicode_minus':False,'xtick.major.width':.6,'ytick.major.width':.6})
def base():
 f,a=plt.subplots(figsize=(6,2.9));a.grid(axis='y',color='.88',linewidth=.5);a.set_axisbelow(True);return f,a
def save(f,name):f.tight_layout();f.savefig(OUT/name,dpi=300);plt.close(f)
def main():
 panels=[];term=[];tails=[]
 for c in ['BTC','ETH']:
  d=pd.read_parquet(DATA/f'premium_{c}.parquet');m=d[(d.variant=='trade_ATM')&(d.tenor==1)].dropna(subset=['vrp']).copy();m['iv']=100*np.sqrt(m.iv_annual);m['rv']=100*np.sqrt(m.rv_annual.clip(lower=0));m['month']=m.date.dt.to_period('M').dt.to_timestamp();g=m.groupby('month')[['iv','rv']].mean().reset_index();g['currency']=c;panels.append(g);f,a=base();a.plot(g.month,g.rv.round(4),color='.55',linestyle='--',linewidth=1.1,label='RV');a.plot(g.month,g.iv.round(4),color='black',linewidth=1.3,label='IV');a.xaxis.set_major_locator(mdates.YearLocator());a.xaxis.set_major_formatter(mdates.DateFormatter('%Y'));a.set_ylim(0,None);save(f,f'figure1_{c}.png')
  f,a=base();a.axhline(0,color='black',linewidth=.6);ten=[1,2,3,7,14,30]
  for v,style in [('trade_ATM',dict(color='black',marker='o',linewidth=1.3)),('mark',dict(color='black',marker='^',linestyle=':',linewidth=1.2)),('model_free_grid_deducted',dict(color='.45',marker='s',linestyle='--',linewidth=1.1))]:
   means=[]
   for h in ten:
    z=d[(d.variant==v)&(d.tenor==h)].dropna(subset=['atm_w','rv_total']);z=z[z.rv_total>0];val=z.volpts.mean();means.append(val);term.append(dict(currency=c,variant=v,tenor=h,mean=val,n=z.volpts.count()))
   a.plot(range(6),means,markersize=5,**style)
  a.set_xticks(range(6));a.set_xticklabels([str(h) for h in ten]);a.set_ylim(-8,8);save(f,f'figure2_{c}.png')
  z=m.sort_values('rv_total',ascending=False).copy();z['rank']=np.arange(1,len(z)+1);z['day_share']=z['rank']/len(z);z['cum_vrp']=z.vrp.cumsum();z['currency']=c;tails.append(z[['currency','date','rank','day_share','rv_total','vrp','cum_vrp']])
 f,a=base();a.axhline(0,color='black',linewidth=.6)
 for z,style in zip(tails,[dict(color='black',linewidth=1.3),dict(color='.45',linestyle='--',linewidth=1.1)]):a.plot((z.day_share*100).round(4),z.cum_vrp.round(6),**style)
 a.set_xlim(0,100);save(f,'figure3.png');pd.concat(panels).to_csv(OUT/'figure1_data.csv',index=False);pd.DataFrame(term).to_csv(OUT/'figure2_data.csv',index=False);pd.concat(tails).to_csv(OUT/'figure3_data.csv',index=False)
 names=['figure1_BTC.png','figure1_ETH.png','figure2_BTC.png','figure2_ETH.png','figure3.png'];body=['<!doctype html><meta charset="utf-8"><title>Manuscript and recalculated figures</title><style>body{font:16px sans-serif;margin:30px}section{display:grid;grid-template-columns:1fr 1fr;gap:20px}img{width:100%;height:auto}h2{margin-top:40px}</style><h1>Manuscript and recalculated figures</h1><p>Left: unchanged embedded manuscript image. Right: regenerated from packaged daily inputs. Underlying data: figure1_data.csv, figure2_data.csv, figure3_data.csv.</p>']
 checks=[]
 for i,name in enumerate(names,1):
  orig=ROOT/f'verify/manuscript_image{i}.png';new=OUT/name;im1=Image.open(orig).convert('RGB');im2=Image.open(new).convert('RGB');same=im1.size==im2.size and np.array_equal(np.asarray(im1),np.asarray(im2));canvas=Image.new('RGB',(im1.width+im2.width,max(im1.height,im2.height)+50),'white');draw=ImageDraw.Draw(canvas);draw.text((20,15),'MANUSCRIPT',fill='black');draw.text((im1.width+20,15),'RECALCULATED',fill='black');canvas.paste(im1,(0,50));canvas.paste(im2,(im1.width,50));canvas.save(ROOT/'verify'/(name.replace('.png','_side_by_side.png')))
  checks.append(dict(figure=name,manuscript_image=orig.name,width=im1.width,height=im1.height,pixel_identical=same));body.append(f'<h2>{name}</h2><section><div><h3>Manuscript</h3><img src="../verify/{orig.name}"></div><div><h3>Recalculated</h3><img src="{name}"></div></section>')
 (OUT/'FIGURE_COMPARISON.html').write_text('\n'.join(body));pd.DataFrame(checks).to_csv(ROOT/'verify/FIGURE_VERIFICATION.csv',index=False);source=json.loads((ROOT/'verify/figure_source_data.json').read_text());numeric=[]
 for c in ['BTC','ETH']:
  z=pd.concat(panels);z=z[z.currency==c];ref=source['fig']['ts'][c]
  for metric in ['iv','rv']:
   numeric.append(dict(figure=1,currency=c,series=metric,points=len(z),all_equal=bool(np.array_equal(z[metric].round(4).to_numpy(),np.array(ref[metric]))),precision='4 decimals in paper/build/tables5.json'))
  z=pd.DataFrame(term)
  for variant in ['trade_ATM','mark','model_free_grid_deducted']:
   a=z[(z.currency==c)&(z.variant==variant)];err=max(abs(r.mean-source['prem'][c][variant][str(r.tenor)]) for r in a.itertuples());numeric.append(dict(figure=2,currency=c,series=variant,points=len(a),all_equal=err<1e-10,precision='absolute tolerance 1e-10',max_abs_error=err))
  z=tails[0 if c=='BTC' else 1];ref=source['fig']['tail'][c];numeric.append(dict(figure=3,currency=c,series='cumulative premium',points=len(z),all_equal=bool(np.array_equal(z.cum_vrp.round(6).to_numpy(),np.array(ref['y'])) and np.array_equal((z.day_share*100).round(4).to_numpy(),np.array(ref['x']))),precision='x 4 decimals, y 6 decimals in source JSON'))
 pd.DataFrame(numeric).to_csv(ROOT/'verify/FIGURE_DATA_VERIFICATION.csv',index=False)
 print('All three figures regenerated; underlying daily/monthly data saved.',flush=True)
if __name__=='__main__':main()
