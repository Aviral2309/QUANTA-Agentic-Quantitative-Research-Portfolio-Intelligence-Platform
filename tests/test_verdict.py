from quanta.validation.verdict import (
    determine_research_verdict,
)


def test_weak_performance_does_not_mean_risk_failure():

    verdict = (
        determine_research_verdict(
            risk_review={
                "status": "PASS"
            },
            backtest={
                "cagr": -0.10,
                "sharpe": -1.0,
            },
            walk_forward={
                "status": "COMPLETED"
            },
            test_observations=100,
            minimum_test_observations=40,
        )
    )

    assert (
        verdict["validation_status"]
        == "VALIDATED"
    )

    assert (
        verdict["performance_status"]
        == "WEAK"
    )

    assert (
        verdict["display_status"]
        == "VALIDATED_WITH_WEAK_PERFORMANCE"
    )


def test_risk_rejection_is_separate():

    verdict = (
        determine_research_verdict(
            risk_review={
                "status": "REJECT"
            },
            backtest={
                "cagr": 0.20,
                "sharpe": 1.5,
            },
            walk_forward=None,
            test_observations=100,
            minimum_test_observations=40,
        )
    )

    assert (
        verdict["validation_status"]
        == "RISK_REJECTED"
    )