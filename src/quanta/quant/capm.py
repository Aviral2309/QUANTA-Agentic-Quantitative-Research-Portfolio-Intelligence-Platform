from __future__ import annotations
import numpy as np, pandas as pd, statsmodels.api as sm
from quanta.domain.models import CAPMResult

def periodic_rf(annual_rate: float, periods: int=252) -> float:
    return (1+annual_rate)**(1/periods)-1

def estimate_capm(ticker: str, stock_r: pd.Series, market_r: pd.Series, annual_rf: float, periods: int=252, market_return_method: str="annualized_mean", threshold: float=0.0) -> CAPMResult:
    pair=pd.concat([stock_r.rename("stock"), market_r.rename("market")], axis=1).dropna()
    if len(pair)<30: raise ValueError(f"{ticker}: insufficient aligned observations")
    rf_p=periodic_rf(annual_rf, periods)
    y=pair.stock-rf_p; x=sm.add_constant(pair.market-rf_p)
    model=sm.OLS(y,x).fit()
    beta=float(model.params["market"]); alpha_daily=float(model.params["const"])
    if market_return_method=="cagr":
        # Return-series geometric equivalent; avoids requiring raw prices here.
        market_annual=float((1+pair.market).prod()**(periods/len(pair))-1)
        stock_annual=float((1+pair.stock).prod()**(periods/len(pair))-1)
    else:
        market_annual=float(pair.market.mean()*periods)
        stock_annual=float(pair.stock.mean()*periods)
    required=float(annual_rf+beta*(market_annual-annual_rf))
    alpha_screen=float(stock_annual-required)
    return CAPMResult(ticker=ticker, observations=len(pair), beta=beta, regression_alpha_daily=alpha_daily, r_squared=float(model.rsquared), market_return_annual=market_annual, required_return_annual=required, actual_return_annual=stock_annual, capm_alpha_annual=alpha_screen, classification="positive_alpha" if alpha_screen>threshold else "non_positive_alpha")
