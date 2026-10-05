from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_risk_free_rate(
    csv_path: str | None,
    date_column: str,
    rate_column: str,
    research_date: str,
    fallback_rate: float,
    fallback_to_static: bool = True,
) -> tuple[float, dict]:
    """
    Load the latest available risk-free observation on or before
    the research date.

    Rates must be expressed as decimals:

        0.065 = 6.5%

    Returns
    -------
    rate:
        Annualized risk-free rate.

    metadata:
        Provenance information for audit/reporting.
    """

    fallback = {
        "rate": float(fallback_rate),
        "source": "static_config",
        "observation_date": None,
    }

    if not csv_path:
        return fallback_rate, fallback

    path = Path(csv_path)

    if not path.exists():
        if fallback_to_static:
            return fallback_rate, fallback

        raise FileNotFoundError(
            f"Risk-free dataset not found: {path}"
        )

    df = pd.read_csv(path)

    if (
        date_column not in df.columns
        or rate_column not in df.columns
    ):
        if fallback_to_static:
            return fallback_rate, fallback

        raise ValueError(
            "Risk-free CSV requires columns "
            f"'{date_column}' and '{rate_column}'."
        )

    df = df[
        [date_column, rate_column]
    ].copy()

    df[date_column] = pd.to_datetime(
        df[date_column],
        errors="coerce",
    )

    df[rate_column] = pd.to_numeric(
        df[rate_column],
        errors="coerce",
    )

    df = df.dropna()

    research_timestamp = pd.Timestamp(
        research_date
    )

    eligible = df[
        df[date_column] <= research_timestamp
    ]

    if eligible.empty:
        if fallback_to_static:
            return fallback_rate, fallback

        raise RuntimeError(
            "No risk-free observation exists "
            "on or before research date."
        )

    row = (
        eligible
        .sort_values(date_column)
        .iloc[-1]
    )

    rate = float(
        row[rate_column]
    )

    metadata = {
        "rate": rate,
        "source": str(path),
        "observation_date": (
            row[date_column]
            .date()
            .isoformat()
        ),
    }

    return rate, metadata