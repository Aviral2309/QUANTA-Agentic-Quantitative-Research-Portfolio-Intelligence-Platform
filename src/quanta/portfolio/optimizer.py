from __future__ import annotations
import numpy as np, pandas as pd
from scipy.optimize import minimize
from quanta.domain.models import OptimizationResult

def optimize_max_sharpe(returns: pd.DataFrame, annual_rf: float=0.0, periods: int=252, max_weight: float=0.20, minimum_weights: dict[str,float]|None=None, restarts: int=5) -> OptimizationResult:
    r=returns.dropna(how="all").copy(); cols=[c for c in r.columns if r[c].notna().sum()>=30]
    r=r[cols].dropna()
    n=len(cols)
    if n==0: raise ValueError("No assets with sufficient common returns")
    if max_weight*n < 1-1e-9: raise ValueError(f"Infeasible max_weight={max_weight} for {n} assets")
    mu=(r.mean()*periods).to_numpy(float); cov=(r.cov()*periods).to_numpy(float)
    cov += np.eye(n)*1e-10
    mins=np.zeros(n)
    if minimum_weights:
        for i,c in enumerate(cols): mins[i]=float(minimum_weights.get(c,0.0))
    if mins.sum()>1+1e-9: raise ValueError("Minimum weights sum above 1")
    bounds=[(mins[i],max_weight) for i in range(n)]
    if sum(b[1] for b in bounds)<1-1e-9 or sum(b[0] for b in bounds)>1+1e-9: raise ValueError("Weight constraints are infeasible")
    def neg_sharpe(w):
        er=float(w@mu); vol=float(np.sqrt(max(w@cov@w,1e-16))); return -((er-annual_rf)/vol)
    cons={"type":"eq","fun":lambda w: np.sum(w)-1}
    starts=[np.repeat(1/n,n)]
    rng=np.random.default_rng(42)
    for _ in range(max(0,restarts-1)):
        z=rng.random(n); z=z/z.sum(); z=np.clip(z,mins,[b[1] for b in bounds]); z=z/z.sum(); starts.append(z)
    best=None
    for x0 in starts:
        res=minimize(neg_sharpe,x0,method="SLSQP",bounds=bounds,constraints=cons,options={"maxiter":2000,"ftol":1e-12})
        if res.success and (best is None or res.fun<best.fun): best=res
    if best is None:
        return OptimizationResult(weights={},expected_return=0,annualized_volatility=0,sharpe_ratio=0,success=False,message="SLSQP failed")
    w=best.x; er=float(w@mu); vol=float(np.sqrt(w@cov@w)); sharpe=float((er-annual_rf)/vol) if vol>0 else 0
    return OptimizationResult(weights={c:float(v) for c,v in zip(cols,w)},expected_return=er,annualized_volatility=vol,sharpe_ratio=sharpe,success=True,message=str(best.message))
