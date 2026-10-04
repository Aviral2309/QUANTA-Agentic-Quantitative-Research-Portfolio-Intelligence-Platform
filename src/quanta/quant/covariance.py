import pandas as pd

def annualized_covariance(returns: pd.DataFrame, periods: int=252) -> pd.DataFrame:
    return returns.cov()*periods
