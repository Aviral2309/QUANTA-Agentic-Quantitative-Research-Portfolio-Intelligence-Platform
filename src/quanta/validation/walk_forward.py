from __future__ import annotations

import numpy as np
import pandas as pd

from quanta.portfolio.risk_optimizer import (
    optimize_risk_constrained_sharpe,
)
from quanta.validation.metrics import (
    max_drawdown,
)


def _window_metrics(
    returns: pd.Series,
    risk_free_rate: float,
    annualization_factor: int,
) -> dict:

    clean = (
        returns
        .dropna()
    )

    if clean.empty:
        return {
            "observations": 0,
            "total_return": None,
            "cagr": None,
            "volatility": None,
            "sharpe": None,
            "max_drawdown": None,
        }

    equity = (
        1.0 + clean
    ).cumprod()

    observations = len(clean)

    years = (
        observations
        / annualization_factor
    )

    total_return = float(
        equity.iloc[-1] - 1.0
    )

    if (
        years > 0
        and equity.iloc[-1] > 0
    ):
        cagr = float(
            equity.iloc[-1]
            ** (1.0 / years)
            - 1.0
        )
    else:
        cagr = None

    volatility = float(
        clean.std()
        * np.sqrt(
            annualization_factor
        )
    )

    annual_return = float(
        clean.mean()
        * annualization_factor
    )

    sharpe = (
        (
            annual_return
            - risk_free_rate
        )
        / volatility
        if volatility > 0
        else None
    )

    drawdown = float(
        max_drawdown(
            clean
        )
    )

    return {
        "observations": observations,
        "total_return": total_return,
        "cagr": cagr,
        "volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": drawdown,
    }


def run_walk_forward_validation(
    returns: pd.DataFrame,
    assets: list[str],
    risk_free_rate: float,
    annualization_factor: int,
    max_weight: float,
    max_top3_concentration: float,
    minimum_weights: dict[str, float] | None,
    train_days: int,
    test_days: int,
    step_days: int,
    minimum_windows: int,
    transaction_cost_bps: float = 0.0,
    optimizer_restarts: int = 8,
) -> tuple[dict, pd.DataFrame]:
    """
    Rolling walk-forward validation.

    For each window:

        historical training window
            ↓
        optimize using training only
            ↓
        freeze weights
            ↓
        evaluate on next test window

    No test-window returns are supplied to the optimizer.
    """

    available_assets = [
        ticker
        for ticker in assets
        if ticker in returns.columns
    ]

    data = (
        returns[available_assets]
        .sort_index()
        .copy()
    )

    total_required = (
        train_days
        + test_days
    )

    if len(data) < total_required:

        summary = {
            "status": "INSUFFICIENT_DATA",
            "windows": 0,
            "message": (
                "Not enough observations for "
                "walk-forward validation."
            ),
        }

        return (
            summary,
            pd.DataFrame(),
        )

    records: list[dict] = []

    test_series_parts: list[pd.Series] = []

    window_number = 0

    start = 0

    while (
        start
        + train_days
        + test_days
        <= len(data)
    ):

        train = data.iloc[
            start:
            start + train_days
        ]

        test = data.iloc[
            start + train_days:
            start + train_days + test_days
        ]

        if (
            len(train) < train_days
            or len(test) < test_days
        ):
            break

        try:
            optimization = (
                optimize_risk_constrained_sharpe(
                    returns=train,
                    risk_free_rate=risk_free_rate,
                    annualization_factor=(
                        annualization_factor
                    ),
                    max_weight=max_weight,
                    max_top3_concentration=(
                        max_top3_concentration
                    ),
                    minimum_weights=(
                        minimum_weights
                    ),
                    restarts=(
                        optimizer_restarts
                    ),
                )
            )

        except Exception as exc:

            records.append(
                {
                    "window": window_number + 1,
                    "status": "OPTIMIZATION_FAILED",
                    "error": str(exc),
                }
            )

            start += step_days
            window_number += 1
            continue

        weights = pd.Series(
            optimization.weights,
            dtype=float,
        )

        weights = weights.reindex(
            available_assets
        ).fillna(0.0)

        portfolio_returns = (
            test[available_assets]
            .fillna(0.0)
            .mul(
                weights,
                axis=1,
            )
            .sum(axis=1)
        )

        # Apply one-off transaction-cost haircut
        # at the start of each test/rebalance window.
        if (
            transaction_cost_bps > 0
            and not portfolio_returns.empty
        ):

            portfolio_returns = (
                portfolio_returns.copy()
            )

            portfolio_returns.iloc[0] -= (
                transaction_cost_bps
                / 10000.0
            )

        metrics = _window_metrics(
            portfolio_returns,
            risk_free_rate,
            annualization_factor,
        )

        record = {
            "window": window_number + 1,
            "status": "OK",

            "train_start": (
                train.index[0]
                .date()
                .isoformat()
            ),

            "train_end": (
                train.index[-1]
                .date()
                .isoformat()
            ),

            "test_start": (
                test.index[0]
                .date()
                .isoformat()
            ),

            "test_end": (
                test.index[-1]
                .date()
                .isoformat()
            ),

            **metrics,
        }

        records.append(
            record
        )

        portfolio_returns.name = (
            f"window_{window_number + 1}"
        )

        test_series_parts.append(
            portfolio_returns
        )

        start += step_days
        window_number += 1

    successful = [
        record
        for record in records
        if record.get("status") == "OK"
    ]

    if len(successful) < minimum_windows:

        summary = {
            "status": "INSUFFICIENT_WINDOWS",
            "windows": len(successful),
            "required_windows": minimum_windows,
        }

        return (
            summary,
            pd.DataFrame(records),
        )

    combined = pd.concat(
        test_series_parts
    ).sort_index()

    # In case overlapping windows are ever configured.
    combined = combined[
        ~combined.index.duplicated(
            keep="first"
        )
    ]

    overall = _window_metrics(
        combined,
        risk_free_rate,
        annualization_factor,
    )

    profitable_windows = sum(
        1
        for record in successful
        if (
            record.get("total_return")
            is not None
            and record["total_return"] > 0
        )
    )

    positive_sharpe_windows = sum(
        1
        for record in successful
        if (
            record.get("sharpe")
            is not None
            and record["sharpe"] > 0
        )
    )

    summary = {
        "status": "COMPLETED",
        "windows": len(successful),

        "profitable_windows": (
            profitable_windows
        ),

        "profitable_window_ratio": (
            profitable_windows
            / len(successful)
        ),

        "positive_sharpe_windows": (
            positive_sharpe_windows
        ),

        "positive_sharpe_window_ratio": (
            positive_sharpe_windows
            / len(successful)
        ),

        "overall": overall,
    }

    return (
        summary,
        pd.DataFrame(records),
    )