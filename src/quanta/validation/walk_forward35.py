from __future__ import annotations
import numpy as np
import pandas as pd
from quanta.portfolio.robust_optimizer import optimize_robust
from quanta.validation.advanced_metrics import cagr, sharpe, max_drawdown, annualized_volatility


def walk_forward_robust(returns: pd.DataFrame, assets: list[str], rf: float, periods: int=252,
                        train_days: int=504, test_days: int=63, step_days: int=63,
                        max_weight: float=.20, top3_limit: float=.50, transaction_cost_bps: float=10.0) -> dict:
    x=returns[[a for a in assets if a in returns.columns]].dropna()
    windows=[]; realized=[]; prev=None
    for start in range(0,max(0,len(x)-train_days-test_days+1),step_days):
        tr=x.iloc[start:start+train_days]; te=x.iloc[start+train_days:start+train_days+test_days]
        if len(te)<test_days: break
        candidates=optimize_robust(tr,rf,periods,max_weight,top3_limit)
        if not candidates: continue
        # Selection uses training objective only.
        chosen=max(candidates,key=lambda c:c.sharpe_ratio)
        w=pd.Series(chosen.weights).reindex(te.columns).fillna(0.0)
        r=te.dot(w)
        turnover=float(w.abs().sum()) if prev is None else float((w-prev.reindex(w.index).fillna(0)).abs().sum())
        cost=turnover*transaction_cost_bps/10000.0
        if len(r): r.iloc[0]-=cost
        realized.append(r); windows.append({'start':str(te.index.min()),'end':str(te.index.max()),'model':chosen.name,
            'cagr':cagr(r,periods),'sharpe':sharpe(r,rf,periods),'max_drawdown':max_drawdown(r),'turnover':turnover,'cost':cost})
        prev=w
    if not realized: return {'status':'INSUFFICIENT_DATA','windows':0,'window_details':[]}
    allr=pd.concat(realized).sort_index(); profitable=sum(w['cagr']>0 for w in windows)/len(windows); possh=sum(w['sharpe']>0 for w in windows)/len(windows)
    return {'status':'COMPLETED','windows':len(windows),'profitable_window_ratio':profitable,'positive_sharpe_window_ratio':possh,
            'overall':{'cagr':cagr(allr,periods),'sharpe':sharpe(allr,rf,periods),'annualized_volatility':annualized_volatility(allr,periods),'max_drawdown':max_drawdown(allr)},
            'window_details':windows,'returns':allr}
