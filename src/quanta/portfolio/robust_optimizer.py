from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from quanta.quant.robust import shrink_expected_returns, shrink_covariance

@dataclass
class CandidatePortfolio:
    name: str
    weights: dict[str,float]
    expected_return: float
    annualized_volatility: float
    sharpe_ratio: float


def _stats(w, mu, cov, rf):
    er=float(w@mu); vol=float(np.sqrt(max(w@cov@w,0.0))); sh=(er-rf)/vol if vol>1e-12 else -1e9
    return er,vol,sh


def optimize_robust(returns: pd.DataFrame, rf: float, periods: int=252, max_weight: float=.20,
                    top3_limit: float=.50, return_shrinkage: float=.60, covariance_shrinkage: float=.25,
                    l2_penalty: float=.02, min_position: float=0.0, restarts: int=12) -> list[CandidatePortfolio]:
    x=returns.dropna(axis=1,how='all').dropna()
    if x.shape[1]<2 or len(x)<40: raise ValueError('Insufficient clean data for robust optimization')
    assets=list(x.columns); n=len(assets)
    if max_weight*n < 1-1e-9: raise ValueError('max_weight makes portfolio infeasible')
    mu_s=shrink_expected_returns(x,periods,return_shrinkage).reindex(assets).values
    cov=shrink_covariance(x,periods,covariance_shrinkage).reindex(index=assets,columns=assets).values
    bounds=[(min_position,max_weight)]*n
    cons=[{'type':'eq','fun':lambda w: np.sum(w)-1},
          {'type':'ineq','fun':lambda w: top3_limit-np.sort(w)[-3:].sum()}]
    rng=np.random.default_rng(42)
    def solve(name,obj):
        best=None
        for i in range(max(1,restarts)):
            w=np.ones(n)/n if i==0 else rng.dirichlet(np.ones(n))
            w=np.minimum(w,max_weight); w=w/w.sum()
            res=minimize(obj,w,method='SLSQP',bounds=bounds,constraints=cons,options={'maxiter':3000,'ftol':1e-10})
            if res.success and (best is None or res.fun<best.fun): best=res
        if best is None: return None
        w=np.clip(best.x,0,None); w=w/w.sum(); er,vol,sh=_stats(w,mu_s,cov,rf)
        return CandidatePortfolio(name,{a:float(v) for a,v in zip(assets,w) if v>1e-8},er,vol,sh)
    def neg_sharpe(w):
        er,vol,sh=_stats(w,mu_s,cov,rf); return -sh + l2_penalty*float(np.sum(w*w))
    def min_var(w): return float(w@cov@w) + l2_penalty*float(np.sum(w*w))
    candidates=[solve('regularized_max_sharpe',neg_sharpe), solve('minimum_variance',min_var)]
    # equal weight is a transparent baseline and obeys limits when feasible
    ew=np.ones(n)/n
    if ew.max()<=max_weight+1e-9 and np.sort(ew)[-3:].sum()<=top3_limit+1e-9:
        er,vol,sh=_stats(ew,mu_s,cov,rf); candidates.append(CandidatePortfolio('equal_weight',{a:float(v) for a,v in zip(assets,ew)},er,vol,sh))
    return [c for c in candidates if c is not None]
