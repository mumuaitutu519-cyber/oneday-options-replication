import os,sys,time,json,platform,subprocess,importlib.metadata as md
from pathlib import Path
root=Path(__file__).resolve().parents[1];mode=sys.argv[1];offline=sys.argv[2]=='1';start=time.monotonic();records=[]
# All analysis stages are network-free; offline also blocks accidental Python socket connections.
if offline:
 os.environ['REPLICATION_NO_NETWORK']='1'
 os.environ['PYTHONPATH']=str(root/'code')
 subprocess.run([sys.executable,'-c',"import socket; assert socket.socket.connect.__name__=='blocked'; print('OFFLINE_SOCKET_BLOCK_CONFIRMED')"],check=True)
for script in ['extract.py']+(['estimate.py'] if mode=='full' else [])+['tables.py','verify.py','figures.py','notes.py','audit_inputs.py']:
 t=time.monotonic();subprocess.run([sys.executable,str(root/'code'/script)],cwd=root,check=True);records.append(dict(stage=script,seconds=time.monotonic()-t))
result=dict(mode=mode,offline=offline,wall_seconds=time.monotonic()-start,python=sys.version,platform=platform.platform(),machine=platform.machine(),workers=int(os.environ.get('REPLICATION_WORKERS','4')),numeric_threads=1,memory_budget_gb=8.5,seeds=dict(forecasts=20261007,stationary_bootstrap=20261007,mcs=20261007),stages=records,versions={x:md.version(x) for x in ['numpy','pandas','scipy','statsmodels','scikit-learn','pyarrow','matplotlib','Pillow']})
(root/'logs'/f'{mode}_runtime.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
