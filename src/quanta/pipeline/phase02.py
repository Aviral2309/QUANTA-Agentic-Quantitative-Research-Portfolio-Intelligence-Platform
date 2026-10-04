from __future__ import annotations

from pathlib import Path

import pandas as pd

from quanta.portfolio.optimizer import optimize_max_sharpe
from quanta.portfolio.diversification import (
    select_low_correlation_diversifiers,
)
from quanta.quant.factors import score_stocks
from quanta.validation.backtest import backtest_static
from quanta.validation.factor_regression import run_factor_regression
from quanta.reporting.export import write_json


def _normalize_ticker(ticker: str) -> str:
    """
    Normalize an NSE ticker to the Yahoo Finance convention.

    Example:
        RELIANCE -> RELIANCE.NS
        RELIANCE.NS -> RELIANCE.NS
    """
    ticker = str(ticker).strip().upper()

    if not ticker:
        return ticker

    if not ticker.endswith(".NS"):
        ticker = f"{ticker}.NS"

    return ticker


def _load_sentiment(path: str | None) -> pd.DataFrame:
    """
    Load optional news-sentiment data.

    Expected schema:
        ticker, news_sentiment

    News sentiment is optional in Phase 2. If the file is missing,
    empty, invalid, or does not contain the expected schema, return
    an empty but merge-safe DataFrame.

    This prevents an optional external signal from crashing the
    deterministic portfolio pipeline.
    """

    required_columns = ["ticker", "news_sentiment"]

    if not path:
        return pd.DataFrame(columns=required_columns)

    sentiment_path = Path(path)

    if not sentiment_path.exists():
        return pd.DataFrame(columns=required_columns)

    try:
        df = pd.read_csv(sentiment_path)

    except (pd.errors.EmptyDataError, OSError, UnicodeDecodeError):
        return pd.DataFrame(columns=required_columns)

    if df.empty:
        return pd.DataFrame(columns=required_columns)

    # Normalize column names
    df.columns = [
        str(column).strip().lower()
        for column in df.columns
    ]

    # ticker is mandatory for joining
    if "ticker" not in df.columns:
        return pd.DataFrame(columns=required_columns)

    # sentiment itself is optional
    if "news_sentiment" not in df.columns:
        df["news_sentiment"] = 0.0

    df = df[required_columns].copy()

    df["ticker"] = (
        df["ticker"]
        .astype(str)
        .map(_normalize_ticker)
    )

    df["news_sentiment"] = pd.to_numeric(
        df["news_sentiment"],
        errors="coerce",
    ).fillna(0.0)

    df = df.drop_duplicates(
        subset=["ticker"],
        keep="last",
    )

    return df.reset_index(drop=True)


def _build_volatility_frame(
    train_returns: pd.DataFrame,
    candidates: list[str],
    annualization_factor: int,
) -> pd.DataFrame:
    """
    Calculate annualized volatility for candidate stocks.

    Explicitly constructs the DataFrame instead of relying on the
    name of a pandas index. This avoids merge errors caused by the
    Series index being named something other than 'ticker'.
    """

    available_candidates = [
        ticker
        for ticker in candidates
        if ticker in train_returns.columns
    ]

    if not available_candidates:
        return pd.DataFrame(
            columns=["ticker", "volatility"]
        )

    volatility_series = (
        train_returns[available_candidates].std()
        * annualization_factor ** 0.5
    )

    volatility = pd.DataFrame(
        {
            "ticker": volatility_series.index.astype(str),
            "volatility": volatility_series.values,
        }
    )

    volatility["ticker"] = (
        volatility["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    volatility["volatility"] = pd.to_numeric(
        volatility["volatility"],
        errors="coerce",
    )

    return volatility


def _prepare_metadata(
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate and normalize the metadata received from Phase 1.
    """

    if metadata is None or metadata.empty:
        raise RuntimeError(
            "Phase 1 metadata is empty. "
            "Phase 2 requires stock metadata."
        )

    metadata = metadata.copy()

    # Handle ticker accidentally stored as index
    if "ticker" not in metadata.columns:
        if metadata.index.name in ("ticker", "Ticker"):
            metadata = metadata.reset_index()

    # Handle common alternate column name
    if "ticker" not in metadata.columns:
        ticker_aliases = [
            "Ticker",
            "symbol",
            "Symbol",
            "SYMBOL",
        ]

        for alias in ticker_aliases:
            if alias in metadata.columns:
                metadata = metadata.rename(
                    columns={alias: "ticker"}
                )
                break

    if "ticker" not in metadata.columns:
        raise RuntimeError(
            "Phase 1 metadata does not contain a 'ticker' column. "
            f"Available columns: {metadata.columns.tolist()}"
        )

    metadata["ticker"] = (
        metadata["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    metadata = metadata.drop_duplicates(
        subset=["ticker"],
        keep="last",
    )

    return metadata


def run_phase02(
    cfg,
    phase1: dict,
) -> dict:
    """
    Execute QUANTA Phase 2.

    Workflow
    --------
    1. Read Phase 1 positive-alpha candidates.
    2. Run Solver #1 (maximum Sharpe).
    3. Select highest-weight candidates.
    4. Construct factor features.
    5. Add volatility and optional sentiment.
    6. Calculate multi-factor scores.
    7. Select core stocks.
    8. Select low-correlation diversifiers.
    9. Run Solver #2.
    10. Perform out-of-sample backtest.
    11. Run optional factor regression.
    12. Export Phase 2 artifacts.
    """

    # ---------------------------------------------------------
    # 1. PHASE 1 INPUTS
    # ---------------------------------------------------------

    out = Path(phase1["run_dir"])
    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    train = phase1["train_returns"].copy()
    test = phase1["test_returns"].copy()
    capm = phase1["capm_results"].copy()
    metadata = _prepare_metadata(
        phase1["metadata"]
    )

    if train.empty:
        raise RuntimeError(
            "Training return dataset is empty."
        )

    if test.empty:
        raise RuntimeError(
            "Test return dataset is empty."
        )

    if capm.empty:
        raise RuntimeError(
            "CAPM results are empty."
        )

    # ---------------------------------------------------------
    # 2. NORMALIZE CAPM TICKER COLUMN
    # ---------------------------------------------------------

    if "ticker" not in capm.columns:
        raise RuntimeError(
            "CAPM results do not contain a 'ticker' column."
        )

    capm["ticker"] = (
        capm["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    if "classification" not in capm.columns:
        raise RuntimeError(
            "CAPM results do not contain a "
            "'classification' column."
        )

    # ---------------------------------------------------------
    # 3. POSITIVE-ALPHA UNIVERSE
    # ---------------------------------------------------------

    positive = (
        capm.loc[
            capm["classification"] == "positive_alpha",
            "ticker",
        ]
        .dropna()
        .tolist()
    )

    # Only retain stocks for which we actually have
    # training-return data.
    positive = [
        ticker
        for ticker in positive
        if ticker in train.columns
    ]

    if len(positive) < 5:
        raise RuntimeError(
            f"Only {len(positive)} positive-alpha stocks "
            "are available. At least 5 are required "
            "for Phase 2."
        )

    # ---------------------------------------------------------
    # 4. SOLVER #1
    # ---------------------------------------------------------

    # Ensure the maximum weight constraint remains feasible.
    max_weight_solver1 = max(
        cfg.portfolio.max_weight,
        (1.0 / len(positive)) + 1e-6,
    )

    opt1 = optimize_max_sharpe(
        train[positive],
        cfg.risk_free.annual_rate,
        cfg.annualization_factor,
        max_weight_solver1,
        restarts=cfg.portfolio.optimizer_restarts,
    )

    write_json(
        out / "solver1.json",
        opt1,
    )

    # ---------------------------------------------------------
    # 5. SELECT SOLVER #1 CANDIDATES
    # ---------------------------------------------------------

    candidate_count = min(
        cfg.portfolio.candidate_count,
        len(opt1.weights),
    )

    candidates = [
        ticker
        for ticker, _ in sorted(
            opt1.weights.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:candidate_count]
    ]

    candidates = [
        ticker
        for ticker in candidates
        if ticker in train.columns
    ]

    if not candidates:
        raise RuntimeError(
            "Solver #1 produced no usable candidates."
        )

    # ---------------------------------------------------------
    # 6. BUILD FACTOR FEATURE DATASET
    # ---------------------------------------------------------

    feature = metadata[
        metadata["ticker"].isin(candidates)
    ].copy()

    if feature.empty:
        raise RuntimeError(
            "No candidate metadata could be matched "
            "after Solver #1."
        )

    # ---------------------------------------------------------
    # 7. VOLATILITY
    # ---------------------------------------------------------

    volatility = _build_volatility_frame(
        train_returns=train,
        candidates=candidates,
        annualization_factor=(
            cfg.annualization_factor
        ),
    )

    if "ticker" not in volatility.columns:
        raise RuntimeError(
            "Internal error: volatility frame "
            "does not contain ticker."
        )

    feature = feature.merge(
        volatility[
            ["ticker", "volatility"]
        ],
        on="ticker",
        how="left",
    )

    # ---------------------------------------------------------
    # 8. NEWS SENTIMENT
    # ---------------------------------------------------------

    sentiment = _load_sentiment(
        cfg.factor_scoring.sentiment_csv
    )

    # Final defensive schema check.
    if "ticker" not in sentiment.columns:
        sentiment = pd.DataFrame(
            columns=[
                "ticker",
                "news_sentiment",
            ]
        )

    if "news_sentiment" not in sentiment.columns:
        sentiment["news_sentiment"] = 0.0

    feature = feature.merge(
        sentiment[
            ["ticker", "news_sentiment"]
        ],
        on="ticker",
        how="left",
    )

    # Missing sentiment = neutral.
    feature["news_sentiment"] = (
        pd.to_numeric(
            feature["news_sentiment"],
            errors="coerce",
        )
        .fillna(0.0)
    )

    # ---------------------------------------------------------
    # 9. EXPORT RAW FACTOR FEATURES
    # ---------------------------------------------------------

    feature.to_csv(
        out / "factor_features.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 10. MULTI-FACTOR SCORING
    # ---------------------------------------------------------

    scored = score_stocks(
        feature,
        cfg.factor_scoring.weights,
    )

    if scored.empty:
        raise RuntimeError(
            "Factor scoring returned no stocks."
        )

    if "ticker" not in scored.columns:
        raise RuntimeError(
            "Factor scoring output does not contain "
            "a ticker column."
        )

    scored.to_csv(
        out / "factor_scores.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # 11. CORE STOCK SELECTION
    # ---------------------------------------------------------

    core_count = min(
        cfg.portfolio.core_count,
        len(scored),
    )

    core = (
        scored
        .head(core_count)["ticker"]
        .tolist()
    )

    if not core:
        raise RuntimeError(
            "No core stocks were selected."
        )

    # ---------------------------------------------------------
    # 12. BUILD DIVERSIFIER UNIVERSE
    # ---------------------------------------------------------

    universe = phase1["universe"].copy()

    if "ticker" not in universe.columns:
        raise RuntimeError(
            "Phase 1 universe does not contain "
            "a ticker column."
        )

    selected_universe = (
        universe["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .tolist()
    )

    remaining = [
        ticker
        for ticker in selected_universe
        if ticker not in core
        and ticker in train.columns
    ]

    # ---------------------------------------------------------
    # 13. LOW-CORRELATION DIVERSIFIERS
    # ---------------------------------------------------------

    diversifier_count = min(
        cfg.portfolio.diversifier_count,
        len(remaining),
    )

    if diversifier_count > 0:
        diversification = (
            select_low_correlation_diversifiers(
                train,
                core,
                remaining,
                diversifier_count,
            )
        )
    else:
        diversification = pd.DataFrame(
            columns=[
                "ticker",
                "average_correlation",
            ]
        )

    diversification.to_csv(
        out / "diversifiers.csv",
        index=False,
    )

    if (
        not diversification.empty
        and "ticker" in diversification.columns
    ):
        diversifiers = (
            diversification["ticker"]
            .tolist()
        )
    else:
        diversifiers = []

    # ---------------------------------------------------------
    # 14. FINAL PORTFOLIO UNIVERSE
    # ---------------------------------------------------------

    final_assets = list(
        dict.fromkeys(
            core + diversifiers
        )
    )

    final_assets = [
        ticker
        for ticker in final_assets
        if ticker in train.columns
    ]

    if len(final_assets) < 2:
        raise RuntimeError(
            "Final asset universe contains fewer "
            "than two stocks."
        )

    # ---------------------------------------------------------
    # 15. CORE MINIMUM-WEIGHT CONSTRAINT
    # ---------------------------------------------------------

    minimum_core_weights = {
        ticker: (
            cfg.portfolio.minimum_core_weight
        )
        for ticker in core
        if ticker in final_assets
    }

    # Basic feasibility guard.
    required_core_weight = (
        len(minimum_core_weights)
        * cfg.portfolio.minimum_core_weight
    )

    if required_core_weight > 1.0:
        raise RuntimeError(
            "Core minimum-weight constraints are "
            "infeasible: total required core weight "
            f"is {required_core_weight:.2%}."
        )

    # ---------------------------------------------------------
    # 16. SOLVER #2
    # ---------------------------------------------------------

    final_max_weight = max(
        cfg.portfolio.max_weight,
        (1.0 / len(final_assets)) + 1e-6,
    )

    opt2 = optimize_max_sharpe(
        train[final_assets],
        cfg.risk_free.annual_rate,
        cfg.annualization_factor,
        final_max_weight,
        minimum_core_weights,
        cfg.portfolio.optimizer_restarts,
    )

    write_json(
        out / "solver2.json",
        opt2,
    )

    # ---------------------------------------------------------
    # 17. OUT-OF-SAMPLE BENCHMARK
    # ---------------------------------------------------------

    benchmark = None

    if cfg.benchmark_ticker in test.columns:
        benchmark = test[
            cfg.benchmark_ticker
        ]

    # ---------------------------------------------------------
    # 18. OUT-OF-SAMPLE BACKTEST
    # ---------------------------------------------------------

    backtest_result, curve = backtest_static(
        opt2.weights,
        test,
        cfg.risk_free.annual_rate,
        cfg.annualization_factor,
        benchmark,
        cfg.portfolio.transaction_cost_bps,
    )

    curve.to_csv(
        out / "backtest_curve.csv"
    )

    write_json(
        out / "backtest.json",
        backtest_result,
    )

    # ---------------------------------------------------------
    # 19. FACTOR REGRESSION
    # ---------------------------------------------------------

    factor_result = run_factor_regression(
        curve["portfolio_return"],
        cfg.validation.factor_csv,
        cfg.annualization_factor,
    )

    write_json(
        out / "factor_regression.json",
        factor_result,
    )

    # ---------------------------------------------------------
    # 20. PHASE 2 SUMMARY
    # ---------------------------------------------------------

    phase2_summary = {
        "positive_alpha_count": len(positive),
        "solver1_candidate_count": len(candidates),
        "core_count": len(core),
        "diversifier_count": len(diversifiers),
        "final_asset_count": len(final_assets),
        "core": core,
        "diversifiers": diversifiers,
        "final_assets": final_assets,
    }

    write_json(
        out / "phase02_summary.json",
        phase2_summary,
    )

    # ---------------------------------------------------------
    # 21. RETURN PIPELINE STATE
    # ---------------------------------------------------------

    return {
        **phase1,

        "solver1": opt1,
        "candidates": candidates,

        "factor_features": feature,
        "factor_scores": scored,

        "core": core,
        "diversifiers": diversifiers,
        "final_assets": final_assets,

        "solver2": opt2,

        "backtest": backtest_result,
        "backtest_curve": curve,

        "factor_regression": factor_result,

        "phase02_summary": phase2_summary,
    }