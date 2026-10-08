from __future__ import annotations
import numpy as np
import pandas as pd

def _s(r):
    s=pd.Series(r,dtype=float).dropna()
    if not np.isfinite(s).all() or (s<=-1).any(): raise ValueError("Invalid daily returns")
    return s
def annualized_return(r,periods=252):
    s=_s(r); return float(s.mean()*periods) if len(s) else float("nan")
def annualized_volatility(r,periods=252):
    s=_s(r); return float(s.std(ddof=1)*np.sqrt(periods)) if len(s)>1 else float("nan")
def cagr(r,periods=252):
    s=_s(r)
    return float(np.expm1(np.log1p(s).sum()*periods/len(s))) if len(s) else float("nan")
def sharpe(r,rf=0,periods=252):
    s=_s(r); v=annualized_volatility(s,periods)
    return float((annualized_return(s,periods)-rf)/v) if v>0 else float("nan")
def max_drawdown(r):
    s=_s(r)
    if s.empty:return float("nan")
    equity=np.r_[1.,np.cumprod(1+s.to_numpy())]
    return float(np.min(equity/np.maximum.accumulate(equity)-1))
def tracking_error(r,b,periods=252):
    a,z=_s(r).align(_s(b),join="inner")
    return annualized_volatility(a-z,periods)
def information_ratio(r,b,periods=252):
    a,z=_s(r).align(_s(b),join="inner")
    v=tracking_error(a,z,periods)
    return float((a-z).mean()*periods/v) if v>0 else float("nan")
def downside_deviation(r,periods=252):
    s=_s(r); return float(np.sqrt(np.mean(np.minimum(s.to_numpy(),0)**2))*np.sqrt(periods)) if len(s) else float("nan")
def sortino(r,rf=0,periods=252):
    d=downside_deviation(r,periods)
    return float((annualized_return(r,periods)-rf)/d) if d>0 else float("nan")
