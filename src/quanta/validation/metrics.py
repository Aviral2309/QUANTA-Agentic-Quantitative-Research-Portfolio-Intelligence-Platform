from __future__ import annotations
import numpy as np, pandas as pd

def equity_curve(returns: pd.Series) -> pd.Series:
    return (1+returns.fillna(0)).cumprod()
def max_drawdown(returns: pd.Series) -> float:
    eq=equity_curve(returns); dd=eq/eq.cummax()-1; return float(dd.min())
def cagr(returns: pd.Series, periods: int=252) -> float:
    r=returns.dropna();
    if len(r)==0: return float("nan")
    return float((1+r).prod()**(periods/len(r))-1)
