from __future__ import annotations


def determine_research_verdict(
    risk_review: dict,
    backtest: dict,
    walk_forward: dict | None,
    test_observations: int,
    minimum_test_observations: int,
) -> dict:
    """
    Separate:

    1. data validation
    2. risk-policy validation
    3. out-of-sample performance

    Poor performance does NOT trigger portfolio reoptimization.
    """

    if (
        test_observations
        < minimum_test_observations
    ):

        validation_status = (
            "INSUFFICIENT_DATA"
        )

    elif (
        risk_review.get("status")
        == "REJECT"
    ):

        validation_status = (
            "RISK_REJECTED"
        )

    else:

        validation_status = (
            "VALIDATED"
        )

    cagr = backtest.get(
        "cagr"
    )

    sharpe = backtest.get(
        "sharpe"
    )

    if (
        cagr is None
        or sharpe is None
    ):

        performance_status = (
            "UNKNOWN"
        )

    elif (
        cagr > 0
        and sharpe > 0
    ):

        performance_status = (
            "POSITIVE"
        )

    else:

        performance_status = (
            "WEAK"
        )

    walk_forward_status = (
        walk_forward.get(
            "status"
        )
        if walk_forward
        else "NOT_RUN"
    )

    if (
        validation_status == "VALIDATED"
        and performance_status == "WEAK"
    ):

        display_status = (
            "VALIDATED_WITH_WEAK_PERFORMANCE"
        )

    else:

        display_status = (
            validation_status
        )

    return {
        "validation_status": (
            validation_status
        ),

        "performance_status": (
            performance_status
        ),

        "walk_forward_status": (
            walk_forward_status
        ),

        "display_status": (
            display_status
        ),
    }