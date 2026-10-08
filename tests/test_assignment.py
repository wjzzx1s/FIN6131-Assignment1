"""Executable regression tests. Run: python tests/test_assignment.py [notebook]."""
import ast
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

NOTEBOOK = Path(sys.argv.pop(1)) if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / 'etf_portfolio_research.ipynb'
CELLS = json.loads(NOTEBOOK.read_text())['cells']
CODE = '\n'.join(''.join(c['source']) for c in CELLS if c['cell_type'] == 'code')
NS = dict(np=np, pd=pd, minimize=minimize, M=252, RF=.03, START='2021-10-09', END='2026-10-09', CAP=3., RF_DAILY=1.03**(1/252)-1,
          N=10, ones=np.ones(10), mu_np=np.zeros(10), Sigma_np=np.eye(10), TAN_CAP=3., RETRIES=3, TICKERS=['SPY','QQQ','IWM','EFA','EEM','AGG','TLT','HYG','GLD','VNQ'])
# Extract actual notebook functions without downloads or plotting.
for node in ast.parse(CODE).body:
    if isinstance(node, (ast.FunctionDef, ast.Import, ast.ImportFrom)):
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(NOTEBOOK), 'exec'), NS)

class AssignmentTests(unittest.TestCase):
    def test_annual_mean_standard_error(self):
        rng = np.random.default_rng(18)
        logs = rng.normal(.0002, .01, (61, 10))
        prices = pd.DataFrame(100*np.exp(np.cumsum(logs, axis=0)),
                              index=pd.date_range('2020-01-01', periods=61), columns=NS['TICKERS'])
        expected = 252*np.diff(np.log(prices.to_numpy()), axis=0).std(axis=0, ddof=1)/np.sqrt(60)
        if 'estimate' in NS:
            _, _, actual = NS['estimate'](np.log(prices).diff().iloc[1:])
        else:
            env = dict(NS, adj=prices, display=lambda *a: None)
            cell = next(''.join(c['source']) for c in CELLS if c['cell_type']=='code' and 'se_mean_annual' in ''.join(c['source']))
            exec(cell, env)
            actual = env['summary']['se_mean_annual'].to_numpy()
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_portability_contract(self):
        self.assertNotIn('config/universe.csv', CODE)
        self.assertNotIn('/home/', CODE)
        self.assertGreaterEqual(sum(c['cell_type']=='code' for c in CELLS), 10)
        self.assertLessEqual(sum(c['cell_type']=='code' for c in CELLS), 15)
        self.assertNotIn('np.linalg.inv', CODE)

    def test_tangency_sign_and_supremum(self):
        self.assertIn('unrestricted', NS)
        S = np.eye(3)
        mu = np.array([-.2, .01, .08])
        answer = NS['unrestricted'](mu, S)
        self.assertIsNone(answer['tangency'])
        A, B, C = 3., mu.sum(), mu@mu
        self.assertAlmostEqual(answer['supremum'], np.sqrt(C-B*B/A))
        positive = NS['unrestricted'](np.array([.04,.09,.2]), S)
        w = positive['tangency']
        np.testing.assert_allclose(w, (np.array([.04,.09,.2])-.03)/.24)
        zero = NS['unrestricted'](np.array([-.07,.03,.13]), S)
        self.assertIsNone(zero['tangency'])

    def test_solver_failure_and_constraints(self):
        self.assertIn('checked', NS)
        from types import SimpleNamespace
        with self.assertRaises(RuntimeError):
            NS['checked'](SimpleNamespace(success=False, message='failed', x=np.ones(3)/3), np.ones(3))
        with self.assertRaises(RuntimeError):
            NS['checked'](SimpleNamespace(success=True, message='', x=np.ones(3)), np.ones(3))
        mu, S = np.array([.05,.1,.15]), np.diag([.03,.05,.08])
        w = NS['constrained'](mu,S,target=.12)
        self.assertAlmostEqual(w.sum(),1,places=8)
        self.assertAlmostEqual(w@mu,.12,places=8)
        self.assertGreaterEqual(w.min(),-1e-8)

    def test_exact_simple_accounting_initial_wealth_and_insolvency(self):
        self.assertIn('performance', NS)
        simple = np.array([[-.2, .1],[.1, .05]])
        p, wealth = NS['performance'](simple, np.array([1.,0.]))
        np.testing.assert_allclose(wealth,[1.,.8,.88])
        self.assertAlmostEqual(p['max_drawdown'],-.2)
        expected = np.sqrt(252)*((simple[:,0]-NS['RF_DAILY']).mean())/simple[:,0].std(ddof=1)
        self.assertAlmostEqual(p['sharpe_simple'],expected)
        p, wealth = NS['performance'](simple,np.array([5.,-4.]))
        self.assertTrue(p['insolvent'])
        self.assertIsNone(p['cagr'])
        self.assertEqual(wealth[-1],0)

    def test_download_validation(self):
        self.assertIn('validate_prices', NS)
        x=pd.DataFrame(np.ones((4,10)),index=pd.date_range('2021-10-11',periods=4),columns=NS['TICKERS'])
        NS['validate_prices'](x)
        x.iloc[1,2]=np.nan
        with self.assertRaises(ValueError): NS['validate_prices'](x)
        x.iloc[1,2]=-1
        with self.assertRaises(ValueError): NS['validate_prices'](x)

if __name__ == '__main__':
    unittest.main(verbosity=2)
