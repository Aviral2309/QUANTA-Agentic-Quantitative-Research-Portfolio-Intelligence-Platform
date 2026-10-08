from __future__ import annotations
import numpy as np
import pandas as pd


def annualized_return(r, periods=252): return float(pd.Series(r).dropna().mean()*periods)
def annualized_volatility(r, periods=252): return float(pd.Series(r).dropna().std(ddof=1)*np.sqrt(periods))
def sharpe(r, rf=0.0, periods=252):
    v=annualized_volatility(r,periods); return (annualized_return(r,periods)-rf)/v if v>0 else 0.0
def max_drawdown(r):
    e=(1+pd.Series(r).dropna()).cumprod(); return float((e/e.cummax()-1).min()) if len(e) else 0.0
def cagr(r, periods=252):
    s=pd.Series(r).dropna(); return float((1+s).prod()**(periods/len(s))-1) if len(s) else 0.0
def tracking_error(r,b,periods=252): return annualized_volatility(pd.Series(r).align(pd.Series(b),join='inner')[0]-pd.Series(r).align(pd.Series(b),join='inner')[1],periods)
def information_ratio(r,b,periods=252):
    rr,bb=pd.Series(r).align(pd.Series(b),join='inner'); te=annualized_volatility(rr-bb,periods); return float((rr-bb).mean()*periods/te) if te>0 else 0.0
def downside_deviation(r,periods=252):
    s=pd.Series(r).dropna(); d=np.minimum(s,0); return float(np.sqrt(np.mean(d*d))*np.sqrt(periods))
def sortino(r,rf=0.0,periods=252):
    dd=downside_deviation(r,periods); return (annualized_return(r,periods)-rf)/dd if dd>0 else 0.0
