from quanta.validation.risk import review_portfolio

def deterministic_risk_review(phase2: dict, limits: dict):
    return review_portfolio(phase2["solver2"].weights,phase2["train_returns"],phase2.get("backtest"),limits)
