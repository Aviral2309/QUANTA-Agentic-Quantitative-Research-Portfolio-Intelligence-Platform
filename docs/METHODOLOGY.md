# Quantitative Methodology

## CAPM regression

For asset i:

`R_i - R_f = alpha_reg + beta_i (R_m - R_f) + epsilon`

The project intentionally distinguishes the regression intercept `alpha_reg` from the screening quantity:

`CAPM screening alpha = observed annual return - [R_f + beta_i (R_m - R_f)]`

The latter is a **CAPM-relative screening signal**, not proof of intrinsic undervaluation.

## Universe design

Top 50 + bottom 50 NIFTY 500 constituents by market capitalization intentionally samples cap extremes. This is not representative of the full NIFTY 500 and should be described as a deliberate bimodal research design.

## Portfolio objective

Both solvers use the same objective: maximize estimated Sharpe ratio subject to long-only, fully-invested and concentration constraints. This consistency prevents the second stage from silently optimizing a different economic objective.

## Multi-factor score

Default inputs: P/E, P/B, EBIT proxy, market cap, realized volatility, news sentiment, average trading volume. Lower P/E/P/B/volatility score better; higher EBIT/cap/sentiment/volume score better. Scores are cross-sectional percentile ranks on 0–100.

## Validation

Portfolio weights are fitted on training returns and evaluated on a later held-out period. Metrics: total return, CAGR, annualized volatility, Sharpe and maximum drawdown. An optional India factor CSV can be used for MKT-RF/SMB/HML/MOM regression. A statistically positive factor alpha is evidence relative to included risk factors, not automatic proof of manager skill.
