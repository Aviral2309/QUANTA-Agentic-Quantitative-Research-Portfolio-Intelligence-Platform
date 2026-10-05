from __future__ import annotations

import numpy as np
import pandas as pd


def equity_curve(
    returns: pd.Series,
) -> pd.Series:
    """
    Convert periodic returns into a cumulative equity curve.

    Example:
        1%, -2%, 3%
            ↓
        cumulative portfolio value starting from 1.0
    """

    clean = (
        pd.to_numeric(
            returns,
            errors="coerce",
        )
        .fillna(0.0)
    )

    return (
        1.0 + clean
    ).cumprod()


def max_drawdown(
    returns: pd.Series,
) -> float:
    """
    Calculate maximum peak-to-trough drawdown.

    Returned as a negative decimal.

    Example:
        -0.15 = -15% maximum drawdown
    """

    if returns is None or len(returns) == 0:
        return float("nan")

    equity = equity_curve(
        returns
    )

    if equity.empty:
        return float("nan")

    running_max = (
        equity.cummax()
    )

    drawdown = (
        equity / running_max
    ) - 1.0

    return float(
        drawdown.min()
    )


def cagr(
    returns: pd.Series,
    periods: int = 252,
) -> float:
    """
    Calculate compound annual growth rate.

    Parameters
    ----------
    returns:
        Periodic portfolio returns.

    periods:
        Number of periods per year.
        252 is normally used for daily trading data.
    """

    clean = (
        pd.to_numeric(
            returns,
            errors="coerce",
        )
        .dropna()
    )

    if clean.empty:
        return float("nan")

    cumulative_growth = float(
        (1.0 + clean).prod()
    )

    # CAGR is undefined if portfolio wealth
    # becomes zero or negative.
    if cumulative_growth <= 0:
        return float("nan")

    years = (
        len(clean)
        / periods
    )

    if years <= 0:
        return float("nan")

    return float(
        cumulative_growth
        ** (1.0 / years)
        - 1.0
    )


def annualized_return(
    returns: pd.Series,
    periods: int = 252,
) -> float:
    """
    Arithmetic annualized mean return.

    Useful for Sharpe-ratio calculations.
    This is intentionally separate from CAGR.
    """

    clean = (
        pd.to_numeric(
            returns,
            errors="coerce",
        )
        .dropna()
    )

    if clean.empty:
        return float("nan")

    return float(
        clean.mean()
        * periods
    )


def annualized_volatility(
    returns: pd.Series,
    periods: int = 252,
) -> float:
    """
    Annualized standard deviation of periodic returns.
    """

    clean = (
        pd.to_numeric(
            returns,
            errors="coerce",
        )
        .dropna()
    )

    if len(clean) < 2:
        return float("nan")

    return float(
        clean.std()
        * np.sqrt(periods)
    )


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods: int = 252,
) -> float:
    """
    Annualized Sharpe ratio.

    risk_free_rate must be supplied as an annual decimal.

    Example:
        0.065 = 6.5%
    """

    ann_return = annualized_return(
        returns,
        periods,
    )

    ann_volatility = annualized_volatility(
        returns,
        periods,
    )

    if (
        np.isnan(ann_return)
        or np.isnan(ann_volatility)
        or ann_volatility <= 0
    ):
        return float("nan")

    return float(
        (
            ann_return
            - risk_free_rate
        )
        / ann_volatility
    )


def total_return(
    returns: pd.Series,
) -> float:
    """
    Calculate cumulative total return over the supplied period.
    """

    clean = (
        pd.to_numeric(
            returns,
            errors="coerce",
        )
        .dropna()
    )

    if clean.empty:
        return float("nan")

    return float(
        (1.0 + clean).prod()
        - 1.0
    )


def summarize_returns(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods: int = 252,
) -> dict:
    """
    Generate the standard QUANTA performance metric set.
    """

    clean = (
        pd.to_numeric(
            returns,
            errors="coerce",
        )
        .dropna()
    )

    return {
        "observations": int(
            len(clean)
        ),

        "total_return": total_return(
            clean
        ),

        "cagr": cagr(
            clean,
            periods,
        ),

        "annualized_return": annualized_return(
            clean,
            periods,
        ),

        "volatility": annualized_volatility(
            clean,
            periods,
        ),

        "sharpe": sharpe_ratio(
            clean,
            risk_free_rate,
            periods,
        ),

        "max_drawdown": max_drawdown(
            clean
        ),
    }