"""Execute with this interpreter, verify frozen offline portability, save evidence.
Run from any directory: python tests/execute_and_verify.py
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
import numpy as np
import pandas as pd

B=Path(__file__).resolve().parents[1]
NAME='etf_portfolio_research.ipynb'

def execute(path,cwd):
    n=nbformat.read(path,as_version=4)
    guard=nbformat.v4.new_code_cell("import yfinance as yf\ndef forbidden(*args, **kwargs):\n    raise AssertionError('Frozen mode attempted a network download')\nyf.download=forbidden")
    n.cells.insert(0,guard)
    km=KernelManager(kernel_name='python3')
    km.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    NotebookClient(n,km=km,timeout=180,resources={'metadata':{'path':str(cwd)}}).execute()
    n.cells.pop(0)
    for i,c in enumerate(c for c in n.cells if c.cell_type=='code'):
        c.execution_count=i+1
        assert not any(o.output_type=='error' for o in c.outputs)
    nbformat.write(n,path)
    return len([c for c in n.cells if c.cell_type=='code'])

def verify(root):
    s=json.loads((root/'results/summary.json').read_text())
    p=pd.read_csv(root/'data/close_adj.csv',index_col=0)
    x=np.diff(np.log(p.to_numpy()),axis=0)
    a=pd.read_csv(root/'results/assets.csv',index_col=0)
    np.testing.assert_allclose(a.mean_annual_log,x.mean(0)*252,rtol=1e-9)
    np.testing.assert_allclose(a.se_mean_annual,x.std(0,ddof=1)*252/np.sqrt(len(x)),rtol=1e-9)
    cov=(x-x.mean(0)).T@(x-x.mean(0))/(len(x)-1)*252
    np.testing.assert_allclose(pd.read_csv(root/'results/cov_annual.csv',index_col=0),cov,rtol=1e-9)
    for name in ['frontier_unrestricted','frontier_long_only']:
        f=pd.read_csv(root/f'results/{name}.csv')
        w=f[s['data']['tickers']].to_numpy()
        np.testing.assert_allclose(w.sum(1),1,atol=1e-8)
        np.testing.assert_allclose(w@a.mean_annual_log,f.target,atol=1e-8)
        np.testing.assert_allclose(np.einsum('ij,jk,ik->i',w,cov,w),f.volatility_log**2,atol=1e-8)
    full=s['full_sample']; d=full['diagnostic']
    if d['tangency_attained']:
        t=full['portfolios']['Tangency unrestricted']
        f=pd.read_csv(root/'results/frontier_unrestricted.csv')
        assert f.expected_return_log.max()>=t['expected_return_log']
        c=pd.read_csv(root/'results/cal.csv'); c=c[c.strategy=='Tangency unrestricted']
        np.testing.assert_allclose(c.expected_return_log,.03+t['sharpe_log_model']*c.volatility_log,atol=1e-9)
    split=s['holdout']['n_is']
    simple=p.pct_change(fill_method=None).iloc[1:].to_numpy()
    perf=pd.read_csv(root/'results/performance.csv')
    wealth=pd.read_csv(root/'results/oos_wealth.csv',index_col=0)
    for name,weights in s['holdout']['weights'].items():
        w=np.array([weights[t] for t in s['data']['tickers']])
        for period,r in [('IS',simple[:split]@w),('OOS',simple[split:]@w)]:
            row=perf[(perf.strategy==name)&(perf.period==period)].iloc[0]
            if np.all(1+r>0):
                np.testing.assert_allclose(row.sharpe_simple,np.sqrt(252)*(r-(1.03**(1/252)-1)).mean()/r.std(ddof=1),rtol=1e-8)
                if period=='OOS': np.testing.assert_allclose(wealth[name],np.r_[1,np.cumprod(1+r)],rtol=1e-9)
    return s

subprocess.run([sys.executable,str(B/'tests/test_assignment.py')],check=True)
count=execute(B/NAME,B)
s=verify(B)
with tempfile.TemporaryDirectory(prefix='etf-portability-',dir=os.environ.get('TMPDIR')) as tmp:
    relocated=Path(tmp)/'submission'
    shutil.copytree(B,relocated,ignore=shutil.ignore_patterns('results','__pycache__'))
    count2=execute(relocated/NAME,relocated)
    verify(relocated)
    for path in (B/'results').glob('*.csv'):
        assert path.read_bytes()==(relocated/'results'/path.name).read_bytes(),path.name
    # Desktop-style relative subfolder mode, likewise no repository present.
    shutil.copy2(B/NAME,Path(tmp)/NAME)
    shutil.copytree(relocated,Path(tmp)/'etf-portfolio-assignment')
    execute(Path(tmp)/NAME,Path(tmp))
    verify(Path(tmp)/'etf-portfolio-assignment')
if (B.parent/NAME).is_file():
    shutil.copy2(B/NAME,B.parent/NAME)
    assert (B/NAME).read_bytes()==(B.parent/NAME).read_bytes()
manifest=json.loads((B/'data/snapshot_manifest.json').read_text())
assert hashlib.sha256((B/'data/close_adj.csv').read_bytes()).hexdigest()==manifest['sha256']
result={'regression_tests':6,'status':'PASS','executed_code_cells':count,'error_cells':0,
        'runs':['bundle clean kernel','relocated bundle without repository','relocated Desktop relative subfolder'],
        'frozen_download_disabled':True,'relocated_csv_byte_identical':True,'desktop_notebook_identical':True,
        'snapshot_copy_identical':True,'data':s['data'],'python':sys.version.split()[0]}
(B/'tests/execution_results.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
