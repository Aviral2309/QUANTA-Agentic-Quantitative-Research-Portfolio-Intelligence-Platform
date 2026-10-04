"""Typed deterministic tool catalog used by the Phase-3 harness.

The graph currently calls the Python functions directly for reliability. This registry documents
what is permitted to be exposed to an LLM/tool-calling layer later.
"""
TOOL_REGISTRY = {
    "market": ["adjusted_close", "company_metadata"],
    "quant": ["calculate_returns", "estimate_capm", "annualized_covariance"],
    "portfolio": ["optimize_max_sharpe", "score_stocks", "select_low_correlation_diversifiers"],
    "validation": ["backtest_static", "run_factor_regression", "review_portfolio"],
}
