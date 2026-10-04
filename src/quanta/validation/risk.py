from __future__ import annotations
import numpy as np, pandas as pd
from quanta.domain.models import RiskFinding, RiskReview, BacktestResult

def review_portfolio(weights: dict[str,float], returns: pd.DataFrame, backtest: BacktestResult|None, limits: dict[str,float]) -> RiskReview:
    findings=[]
    if not weights: return RiskReview(status="REJECT",findings=[RiskFinding(severity="HIGH",code="EMPTY_PORTFOLIO",message="No portfolio weights produced")])
    maxw=max(weights.values())
    if maxw>limits.get("max_single_weight",1)+1e-9: findings.append(RiskFinding(severity="HIGH",code="SINGLE_WEIGHT",message="Single-name weight exceeds policy",evidence={"observed":maxw}))
    top3=sum(sorted(weights.values(),reverse=True)[:3])
    if top3>limits.get("max_top3_concentration",1)+1e-9: findings.append(RiskFinding(severity="HIGH",code="TOP3_CONCENTRATION",message="Top-3 concentration exceeds policy",evidence={"observed":top3}))
    cols=[c for c in weights if c in returns.columns]
    if len(cols)>1:
        c=returns[cols].corr().abs(); vals=c.where(~np.eye(len(c),dtype=bool)).stack()
        avg=float(vals.mean()) if len(vals) else 0.0
        if avg>limits.get("max_avg_pairwise_correlation",1): findings.append(RiskFinding(severity="MEDIUM",code="CORRELATION",message="Average pairwise correlation is high",evidence={"observed":avg}))
    if backtest:
        if backtest.annualized_volatility>limits.get("max_annualized_volatility",99): findings.append(RiskFinding(severity="HIGH",code="VOLATILITY",message="Annualized volatility exceeds policy",evidence={"observed":backtest.annualized_volatility}))
        if abs(backtest.max_drawdown)>limits.get("max_drawdown_abs",99): findings.append(RiskFinding(severity="HIGH",code="DRAWDOWN",message="Maximum drawdown exceeds policy",evidence={"observed":backtest.max_drawdown}))
    reject=any(f.severity=="HIGH" for f in findings)
    return RiskReview(status="REJECT" if reject else "PASS",findings=findings)
