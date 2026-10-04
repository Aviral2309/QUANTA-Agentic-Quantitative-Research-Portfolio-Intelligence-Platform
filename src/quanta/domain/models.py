from __future__ import annotations
from typing import Literal
from pydantic import BaseModel, Field

class CAPMResult(BaseModel):
    ticker: str
    observations: int
    beta: float
    regression_alpha_daily: float
    r_squared: float
    market_return_annual: float
    required_return_annual: float
    actual_return_annual: float
    capm_alpha_annual: float
    classification: Literal["positive_alpha", "non_positive_alpha"]

class OptimizationResult(BaseModel):
    weights: dict[str, float]
    expected_return: float
    annualized_volatility: float
    sharpe_ratio: float
    success: bool
    message: str

class BacktestResult(BaseModel):
    observations: int
    total_return: float
    cagr: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    benchmark_total_return: float | None = None
    benchmark_cagr: float | None = None

class RiskFinding(BaseModel):
    severity: Literal["LOW", "MEDIUM", "HIGH"]
    code: str
    message: str
    evidence: dict = Field(default_factory=dict)

class RiskReview(BaseModel):
    status: Literal["PASS", "REJECT"]
    findings: list[RiskFinding] = Field(default_factory=list)

class FactorRegressionResult(BaseModel):
    status: Literal["SKIPPED", "COMPLETED", "FAILED"]
    alpha_annual: float | None = None
    alpha_t_stat: float | None = None
    coefficients: dict[str, float] = Field(default_factory=dict)
    r_squared: float | None = None
    message: str = ""
