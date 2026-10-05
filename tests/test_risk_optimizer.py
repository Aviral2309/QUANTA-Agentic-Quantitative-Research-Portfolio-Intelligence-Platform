import numpy as np
import pandas as pd

from quanta.portfolio.risk_optimizer import (
    optimize_risk_constrained_sharpe,
)


def test_top3_concentration_is_enforced():

    rng = np.random.default_rng(
        42
    )

    returns = pd.DataFrame(
        rng.normal(
            0.0005,
            0.01,
            size=(700, 10),
        ),
        columns=[
            f"S{i}"
            for i in range(10)
        ],
    )

    result = (
        optimize_risk_constrained_sharpe(
            returns=returns,
            risk_free_rate=0.06,
            annualization_factor=252,
            max_weight=0.20,
            max_top3_concentration=0.50,
            restarts=4,
        )
    )

    weights = sorted(
        result.weights.values(),
        reverse=True,
    )

    assert max(weights) <= 0.2001

    assert (
        sum(weights[:3])
        <= 0.5001
    )

    assert abs(
        sum(weights) - 1.0
    ) < 1e-5