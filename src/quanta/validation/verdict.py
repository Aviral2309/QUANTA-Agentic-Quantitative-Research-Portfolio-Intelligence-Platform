from __future__ import annotations

from typing import Any


# ============================================================
# GENERIC ACCESS HELPERS
# ============================================================


def _value(
    obj: Any,
    key: str,
    default=None,
):
    """
    Read a value from either:
    - dict
    - Pydantic model
    - normal Python object
    """

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(key, default)

    return getattr(obj, key, default)


def _serialize(obj: Any):
    """
    Convert Pydantic/domain objects into JSON-friendly values.
    """

    if obj is None:
        return None

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if hasattr(obj, "dict"):
        return obj.dict()

    return obj


# ============================================================
# RESEARCH VERDICT ENGINE
# ============================================================


def determine_research_verdict(
    risk_review,
    backtest,
    walk_forward=None,
    test_observations: int | None = None,
    minimum_test_observations: int = 40,
) -> dict:
    """
    Determine QUANTA's final research verdict.

    Separates:
    1. methodological validation
    2. portfolio risk
    3. held-out performance
    4. walk-forward validation

    Important
    ---------
    Weak held-out performance does NOT automatically imply
    risk failure.

    Likewise, strong held-out performance does NOT override
    a portfolio risk-policy violation.

    Held-out results are reporting/validation information and
    must never feed back into portfolio optimization.
    """

    # ========================================================
    # RISK STATUS
    # ========================================================

    risk_status = _value(
        risk_review,
        "status",
        "UNKNOWN",
    )

    risk_findings = _value(
        risk_review,
        "findings",
        [],
    )

    # ========================================================
    # TEST OBSERVATIONS
    # ========================================================

    # Explicit argument takes priority because older QUANTA
    # callers/tests already supply test_observations.
    observations = test_observations

    # If not explicitly supplied, try to infer it from the
    # BacktestResult/domain object.
    if observations is None:
        observations = _value(
            backtest,
            "observations",
            None,
        )

    if observations is None:
        observations = _value(
            backtest,
            "test_observations",
            None,
        )

    if observations is None:
        observations = _value(
            backtest,
            "n_observations",
            None,
        )

    # ========================================================
    # HELD-OUT BACKTEST METRICS
    # ========================================================

    cagr = _value(
        backtest,
        "cagr",
        None,
    )

    sharpe = _value(
        backtest,
        "sharpe",
        None,
    )

    # Support newer naming convention too.
    if sharpe is None:
        sharpe = _value(
            backtest,
            "sharpe_ratio",
            None,
        )

    max_drawdown = _value(
        backtest,
        "max_drawdown",
        None,
    )

    total_return = _value(
        backtest,
        "total_return",
        None,
    )

    benchmark_cagr = _value(
        backtest,
        "benchmark_cagr",
        None,
    )

    benchmark_total_return = _value(
        backtest,
        "benchmark_total_return",
        None,
    )

    # ========================================================
    # VALIDATION STATUS
    # ========================================================

    if (
        observations is not None
        and int(observations)
        < int(minimum_test_observations)
    ):
        validation_status = "INSUFFICIENT_DATA"

    elif risk_status == "REJECT":
        validation_status = "RISK_REJECTED"

    else:
        validation_status = "VALIDATED"

    # ========================================================
    # PERFORMANCE STATUS
    # ========================================================

    if cagr is None and sharpe is None:
        performance_status = "UNKNOWN"

    else:
        cagr_positive = (
            cagr is not None
            and float(cagr) > 0.0
        )

        sharpe_positive = (
            sharpe is not None
            and float(sharpe) > 0.0
        )

        if cagr_positive and sharpe_positive:
            performance_status = "POSITIVE"

        else:
            performance_status = "WEAK"

    # ========================================================
    # WALK-FORWARD STATUS
    # ========================================================

    if walk_forward is None:
        walk_forward_status = "NOT_RUN"

    else:
        supplied_wf_status = _value(
            walk_forward,
            "status",
            None,
        )

        if supplied_wf_status is not None:
            walk_forward_status = str(
                supplied_wf_status
            )

        else:
            windows = _value(
                walk_forward,
                "windows",
                None,
            )

            if windows is None:
                windows = _value(
                    walk_forward,
                    "window_count",
                    None,
                )

            if windows is None:
                windows = _value(
                    walk_forward,
                    "n_windows",
                    None,
                )

            if windows is not None:
                try:
                    window_count = int(windows)
                except (TypeError, ValueError):
                    window_count = 0

                walk_forward_status = (
                    "COMPLETED"
                    if window_count > 0
                    else "UNKNOWN"
                )

            else:
                walk_forward_status = "UNKNOWN"

    # ========================================================
    # DISPLAY STATUS
    # ========================================================

    if validation_status == "INSUFFICIENT_DATA":

        display_status = "INSUFFICIENT_DATA"

    elif validation_status == "RISK_REJECTED":

        display_status = "RISK_REJECTED"

    elif (
        validation_status == "VALIDATED"
        and performance_status == "WEAK"
    ):

        display_status = (
            "VALIDATED_WITH_WEAK_PERFORMANCE"
        )

    elif (
        validation_status == "VALIDATED"
        and performance_status == "POSITIVE"
    ):

        display_status = "VALIDATED"

    else:
        display_status = validation_status

    # ========================================================
    # SERIALIZE RISK FINDINGS
    # ========================================================

    serialized_findings = []

    for finding in risk_findings or []:
        serialized_findings.append(
            _serialize(finding)
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    return {
        "validation_status": validation_status,

        "performance_status": performance_status,

        "walk_forward_status": walk_forward_status,

        "display_status": display_status,

        "risk_status": risk_status,

        "test_observations": (
            int(observations)
            if observations is not None
            else None
        ),

        "cagr": (
            float(cagr)
            if cagr is not None
            else None
        ),

        "sharpe_ratio": (
            float(sharpe)
            if sharpe is not None
            else None
        ),

        "max_drawdown": (
            float(max_drawdown)
            if max_drawdown is not None
            else None
        ),

        "total_return": (
            float(total_return)
            if total_return is not None
            else None
        ),

        "benchmark_cagr": (
            float(benchmark_cagr)
            if benchmark_cagr is not None
            else None
        ),

        "benchmark_total_return": (
            float(benchmark_total_return)
            if benchmark_total_return is not None
            else None
        ),

        "risk_findings": serialized_findings,
    }