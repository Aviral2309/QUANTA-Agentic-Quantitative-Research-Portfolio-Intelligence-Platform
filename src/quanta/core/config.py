from __future__ import annotations
from pathlib import Path
from typing import Literal
import yaml
from pydantic import BaseModel, Field, model_validator

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
    annual_rate: float = Field(ge=0, le=1)
    source_note: str = ""

class CAPMConfig(BaseModel):
    market_return_method: Literal["annualized_mean", "cagr"] = "annualized_mean"
    positive_alpha_threshold: float = 0.0

class PortfolioConfig(BaseModel):
    max_weight: float = Field(default=0.20, gt=0, le=1)
    candidate_count: int = 10
    core_count: int = 5
    diversifier_count: int = 5
    minimum_core_weight: float = 0.10
    optimizer_restarts: int = 8
    transaction_cost_bps: float = 10

class FactorScoringConfig(BaseModel):
    weights: dict[str, float]
    sentiment_csv: str = "data/raw/news_sentiment.csv"
    @model_validator(mode="after")
    def weights_sum(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"factor_scoring.weights must sum to 1.0, got {total}")
        return self

class ValidationConfig(BaseModel):
    factor_csv: str = "data/raw/india_factors.csv"
    factor_required: bool = False
    benchmark_for_backtest: str
    min_test_observations: int = 40

class AgenticConfig(BaseModel):
    use_llm: bool = False
    model: str = "gpt-5-mini"
    max_reoptimization_loops: int = 2
    risk_limits: dict[str, float]

class ResearchConfig(BaseModel):
    project_name: str = "QUANTA"
    research_date: str
    benchmark_ticker: str
    start_date: str
    end_date: str
    interval: str = "1d"
    annualization_factor: int = 252
    minimum_observations: int = 180
    validation_fraction: float = Field(default=0.25, gt=0, lt=0.5)
    universe: UniverseConfig
    risk_free: RiskFreeConfig
    capm: CAPMConfig
    portfolio: PortfolioConfig
    factor_scoring: FactorScoringConfig
    validation: ValidationConfig
    agentic: AgenticConfig


def load_config(path: str | Path) -> ResearchConfig:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return ResearchConfig.model_validate(payload)
