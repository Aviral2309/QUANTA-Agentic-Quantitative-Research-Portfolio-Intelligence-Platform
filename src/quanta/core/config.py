from __future__ import annotations

from pathlib import Path
from typing import Dict

import yaml
from pydantic import BaseModel, Field


class UniverseConfig(BaseModel):
    constituents_csv: str
    ticker_column: str = "ticker"
    name_column: str = "company_name"
    market_cap_column: str = "market_cap"

    top_n: int = 50
    bottom_n: int = 50

    auto_enrich_market_cap: bool = True
    metadata_workers: int = 12


class RiskFreeConfig(BaseModel):
    annual_rate: float = 0.065

    series_csv: str | None = None
    date_column: str = "date"
    rate_column: str = "annual_rate"

    fallback_to_static: bool = True


class CAPMConfig(BaseModel):
    market_return_method: str = "annualized_mean"
    positive_alpha_threshold: float = 0.0


class PortfolioConfig(BaseModel):
    max_weight: float = 0.20

    candidate_count: int = 10
    core_count: int = 5
    diversifier_count: int = 5

    minimum_core_weight: float = 0.05

    optimizer_restarts: int = 12

    transaction_cost_bps: float = 10.0


class FactorScoringConfig(BaseModel):
    weights: Dict[str, float] = Field(default_factory=dict)

    sentiment_csv: str = "data/raw/news_sentiment.csv"


class WalkForwardConfig(BaseModel):
    enabled: bool = True

    train_days: int = 504
    test_days: int = 63
    step_days: int = 63

    minimum_windows: int = 2


class ValidationConfig(BaseModel):
    factor_csv: str = "data/raw/india_factors.csv"

    factor_required: bool = False

    benchmark_ticker: str = "^CRSLDX"

    min_test_observations: int = 40

    walk_forward: WalkForwardConfig = Field(
        default_factory=WalkForwardConfig
    )


class AgenticConfig(BaseModel):
    use_llm: bool = False

    model: str = "gpt-5-mini"

    max_reoptimization_loops: int = 3


class RiskLimitsConfig(BaseModel):
    max_single_weight: float = 0.20

    max_top3_concentration: float = 0.50

    max_avg_pairwise_correlation: float = 0.75

    max_annualized_volatility: float = 0.45

    max_drawdown_abs: float = 0.45


class ResearchConfig(BaseModel):
    project_name: str = "QUANTA"

    research_date: str

    benchmark_ticker: str = "^CRSLDX"

    start_date: str
    end_date: str

    interval: str = "1d"

    annualization_factor: int = 252

    minimum_observations: int = 180

    validation_fraction: float = 0.25

    universe: UniverseConfig

    risk_free: RiskFreeConfig

    capm: CAPMConfig

    portfolio: PortfolioConfig

    factor_scoring: FactorScoringConfig

    validation: ValidationConfig

    agentic: AgenticConfig

    risk_limits: RiskLimitsConfig


def load_config(
    path: str | Path,
) -> ResearchConfig:

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:

        raw = yaml.safe_load(handle)

    return ResearchConfig.model_validate(raw)