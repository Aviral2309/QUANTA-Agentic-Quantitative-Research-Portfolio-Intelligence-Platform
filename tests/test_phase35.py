import numpy as np
import pandas as pd
from quanta.quant.robust import shrink_expected_returns, shrink_covariance
from quanta.validation.bootstrap import block_bootstrap_ci
from quanta.validation.evidence import evidence_verdict
from quanta.portfolio.robust_optimizer import optimize_robust


def sample(n=700,k=6):
    rng=np.random.default_rng(7); return pd.DataFrame(rng.normal(.0004,.012,(n,k)),columns=[f'S{i}' for i in range(k)])

def test_shrinkage_shapes():
    r=sample(); assert len(shrink_expected_returns(r))==6; assert shrink_covariance(r).shape==(6,6)

def test_robust_optimizer_constraints():
    cs=optimize_robust(sample(),.06,max_weight=.25,top3_limit=.65)
    assert cs
    for c in cs:
        w=sorted(c.weights.values(),reverse=True); assert abs(sum(w)-1)<1e-5; assert max(w)<=.25001; assert sum(w[:3])<=.65001

def test_bootstrap_and_evidence():
    r=sample(k=1)['S0']; b=block_bootstrap_ci(r,samples=100); assert b['status']=='COMPLETED'
    v=evidence_verdict({'cagr':.04,'excess_cagr':.02,'information_ratio':.3,'sharpe_ratio':.7,'max_drawdown':-.1},b,'PASS',{'status':'COMPLETED','profitable_window_ratio':.7,'positive_sharpe_window_ratio':.7})
    assert v['signal'] in {'STRONG','MODERATE','WEAK','AVOID'}
