from __future__ import annotations
import numpy as np, pandas as pd

def calculate_returns(prices: pd.DataFrame, method: str="simple") -> pd.DataFrame:
    if method=="log": return np.log(prices/prices.shift(1)).replace([np.inf,-np.inf],np.nan)
    return prices.pct_change(fill_method=None).replace([np.inf,-np.inf],np.nan)

def annualized_mean(r: pd.Series, periods: int=252) -> float:
    return float(r.dropna().mean()*periods)

def cagr_from_prices(p: pd.Series, periods: int=252) -> float:
    s=p.dropna()
    if len(s)<2 or s.iloc[0]<=0: return float("nan")
    years=(len(s)-1)/periods
    return float((s.iloc[-1]/s.iloc[0])**(1/years)-1) if years>0 else float("nan")

def split_train_test(df: pd.DataFrame, validation_fraction: float=0.25):
    cut=max(1, int(len(df)*(1-validation_fraction)))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()
