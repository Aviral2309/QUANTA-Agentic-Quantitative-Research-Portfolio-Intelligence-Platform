from __future__ import annotations
from pathlib import Path
import pandas as pd, statsmodels.api as sm
from quanta.domain.models import FactorRegressionResult

def run_factor_regression(portfolio_returns: pd.Series, factor_csv: str, periods: int=252) -> FactorRegressionResult:
    p=Path(factor_csv)
    if not p.exists(): return FactorRegressionResult(status="SKIPPED",message=f"Factor file not found: {factor_csv}")
    try:
        f=pd.read_csv(p,index_col=0,parse_dates=True)
        required=[c for c in ["MKT_RF","SMB","HML","MOM"] if c in f.columns]
        if len(required)<3: return FactorRegressionResult(status="FAILED",message="Need at least MKT_RF, SMB, HML (MOM optional)")
        rf=f["RF"] if "RF" in f.columns else 0.0
        y=portfolio_returns.reindex(f.index)-rf
        df=pd.concat([y.rename("portfolio_excess"),f[required]],axis=1).dropna()
        model=sm.OLS(df.portfolio_excess,sm.add_constant(df[required])).fit()
        alpha_daily=float(model.params["const"])
        return FactorRegressionResult(status="COMPLETED",alpha_annual=alpha_daily*periods,alpha_t_stat=float(model.tvalues["const"]),coefficients={c:float(model.params[c]) for c in required},r_squared=float(model.rsquared),message="Factor regression completed")
    except Exception as e:
        return FactorRegressionResult(status="FAILED",message=str(e))
