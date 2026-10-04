from __future__ import annotations
import numpy as np, pandas as pd
from quanta.domain.models import BacktestResult
from .metrics import cagr, max_drawdown

def backtest_static(weights: dict[str,float], test_returns: pd.DataFrame, annual_rf: float=0.0, periods: int=252, benchmark: pd.Series|None=None, transaction_cost_bps: float=0.0) -> tuple[BacktestResult,pd.DataFrame]:
    assets=[a for a in weights if a in test_returns.columns]
    if not assets: raise ValueError("No weighted assets present in test returns")
    w=np.array([weights[a] for a in assets],float); w=w/w.sum()
    r=test_returns[assets].dropna().dot(w)
    if len(r)==0: raise ValueError("No complete test return rows")
    if transaction_cost_bps>0: r.iloc[0] -= transaction_cost_bps/10000.0
    vol=float(r.std(ddof=1)*np.sqrt(periods)); cg=cagr(r,periods); sh=float((r.mean()*periods-annual_rf)/vol) if vol>0 else 0
    bt=BacktestResult(observations=len(r),total_return=float((1+r).prod()-1),cagr=cg,annualized_volatility=vol,sharpe_ratio=sh,max_drawdown=max_drawdown(r))
    frame=pd.DataFrame({"portfolio_return":r,"portfolio_equity":(1+r).cumprod()})
    if benchmark is not None:
        b=benchmark.reindex(frame.index).dropna(); common=frame.index.intersection(b.index)
        if len(common):
            br=b.loc[common]; bt.benchmark_total_return=float((1+br).prod()-1); bt.benchmark_cagr=cagr(br,periods)
            frame.loc[common,"benchmark_return"]=br; frame.loc[common,"benchmark_equity"]=(1+br).cumprod()
    return bt,frame
