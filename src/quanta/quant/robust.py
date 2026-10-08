from __future__ import annotations
import numpy as np
import pandas as pd


def shrink_expected_returns(returns: pd.DataFrame, periods: int = 252, strength: float = 0.60) -> pd.Series:
    """Shrink noisy sample means toward the cross-sectional grand mean."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    mu = returns.mean(skipna=True) * periods
    prior = float(mu.median()) if len(mu) else 0.0
    return (1.0-strength)*mu + strength*prior


def shrink_covariance(returns: pd.DataFrame, periods: int = 252, strength: float = 0.25) -> pd.DataFrame:
    """Stable diagonal shrinkage: (1-lambda)S + lambda*diag(S)."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    clean=returns.dropna(how="all")
    sample=clean.cov()*periods
    target=pd.DataFrame(np.diag(np.diag(sample)), index=sample.index, columns=sample.columns)
    return (1.0-strength)*sample + strength*target
