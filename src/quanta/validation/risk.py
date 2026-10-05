from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


# ============================================================
# DOMAIN MODELS
# ============================================================


class RiskFinding(BaseModel):
    """
    One deterministic portfolio-risk finding.
    """

    severity: str
    code: str
    message: str

    actual: float | None = None
    limit: float | None = None


class RiskReview(BaseModel):
    """
    Complete QUANTA portfolio-risk review.

    Supports both:

        review.status

    and:

        review.get("status")

    so it remains compatible with the original tests
    and the Phase 3 LangGraph harness.
    """

    status: str

    findings: list[RiskFinding] = Field(
        default_factory=list
    )

    max_single_weight: float = 0.0
    top3_concentration: float = 0.0

    avg_pairwise_correlation: float | None = None
    annualized_volatility: float | None = None

    positions: int = 0

    def get(
        self,
        key: str,
        default=None,
    ):
        """
        Dictionary-style .get() compatibility.

        Example:
            review.get("status")
        """

        return getattr(
            self,
            key,
            default,
        )

    def __getitem__(
        self,
        key: str,
    ):
        """
        Dictionary-style [] compatibility.

        Example:
            review["status"]
        """

        return getattr(
            self,
            key,
        )

    def to_dict(
        self,
    ) -> dict:
        """
        Explicit dictionary conversion helper.
        """

        return self.model_dump()


# ============================================================
# HELPERS
# ============================================================


def _finding(
    severity: str,
    code: str,
    message: str,
    actual: float | None = None,
    limit: float | None = None,
) -> RiskFinding:
    """
    Create a standardized QUANTA risk finding.
    """

    return RiskFinding(
        severity=severity,
        code=code,
        message=message,
        actual=actual,
        limit=limit,
    )


def _limit(
    limits,
    name: str,
    default=None,
):
    """
    Read a risk limit from either:

    1. dictionary
    2. Pydantic/config object

    Examples
    --------

    Dictionary:

        {
            "max_single_weight": 0.20
        }

    Pydantic object:

        RiskLimitsConfig(
            max_single_weight=0.20
        )
    """

    if isinstance(
        limits,
        dict,
    ):
        return limits.get(
            name,
            default,
        )

    return getattr(
        limits,
        name,
        default,
    )


# ============================================================
# MAIN RISK ENGINE
# ============================================================


def review_portfolio(
    weights: dict[str, float],
    returns: pd.DataFrame,
    annualization_factor: int | None,
    limits,
    max_drawdown: float | None = None,
) -> RiskReview:
    """
    Deterministic QUANTA portfolio-risk review.

    Checks
    ------
    1. Empty portfolio
    2. Maximum single-position weight
    3. Top-3 concentration
    4. Average pairwise absolute correlation
    5. Annualized portfolio volatility
    6. Maximum drawdown, when explicitly supplied

    Important
    ---------
    This function should normally operate on information
    available BEFORE held-out validation.

    Held-out test performance must not be used to modify
    portfolio weights.

    max_drawdown should therefore only be supplied when:

    - calculated from training data, or
    - used strictly for post-validation reporting.

    Compatibility
    -------------
    `limits` may be either a dictionary or the production
    RiskLimitsConfig object.

    `annualization_factor=None` is also supported because
    lightweight unit tests may only want concentration
    validation.
    """

    findings: list[RiskFinding] = []

    # ========================================================
    # LOAD RISK POLICY
    # ========================================================

    max_single_limit = float(
        _limit(
            limits,
            "max_single_weight",
            1.0,
        )
    )

    top3_limit = float(
        _limit(
            limits,
            "max_top3_concentration",
            1.0,
        )
    )

    correlation_limit = float(
        _limit(
            limits,
            "max_avg_pairwise_correlation",
            1.0,
        )
    )

    volatility_limit = _limit(
        limits,
        "max_annualized_volatility",
        None,
    )

    drawdown_limit = _limit(
        limits,
        "max_drawdown_abs",
        None,
    )

    # ========================================================
    # EMPTY PORTFOLIO
    # ========================================================

    if not weights:

        findings.append(
            _finding(
                severity="HIGH",
                code="EMPTY_PORTFOLIO",
                message=(
                    "Portfolio contains no positions."
                ),
            )
        )

        return RiskReview(
            status="REJECT",
            findings=findings,
            max_single_weight=0.0,
            top3_concentration=0.0,
            avg_pairwise_correlation=None,
            annualized_volatility=None,
            positions=0,
        )

    # ========================================================
    # CLEAN WEIGHTS
    # ========================================================

    positive_weights: dict[str, float] = {}

    for ticker, weight in weights.items():

        if weight is None:
            continue

        try:
            numeric_weight = float(
                weight
            )
        except (
            TypeError,
            ValueError,
        ):
            continue

        if not np.isfinite(
            numeric_weight
        ):
            continue

        if numeric_weight <= 1e-8:
            continue

        positive_weights[
            str(ticker)
        ] = numeric_weight

    if not positive_weights:

        findings.append(
            _finding(
                severity="HIGH",
                code="EMPTY_PORTFOLIO",
                message=(
                    "Portfolio contains no positive "
                    "positions."
                ),
            )
        )

        return RiskReview(
            status="REJECT",
            findings=findings,
            max_single_weight=0.0,
            top3_concentration=0.0,
            avg_pairwise_correlation=None,
            annualized_volatility=None,
            positions=0,
        )

    # ========================================================
    # NORMALIZE WEIGHTS
    # ========================================================

    total_weight = float(
        sum(
            positive_weights.values()
        )
    )

    if total_weight <= 0:

        findings.append(
            _finding(
                severity="HIGH",
                code="INVALID_WEIGHTS",
                message=(
                    "Portfolio weights have an invalid "
                    "total."
                ),
            )
        )

        return RiskReview(
            status="REJECT",
            findings=findings,
            positions=len(
                positive_weights
            ),
        )

    normalized_weights = {
        ticker: (
            weight
            / total_weight
        )
        for ticker, weight
        in positive_weights.items()
    }

    # ========================================================
    # ORDER POSITIONS
    # ========================================================

    ordered = sorted(
        normalized_weights.items(),
        key=lambda item: item[1],
        reverse=True,
    )

    max_single = float(
        ordered[0][1]
    )

    top3 = float(
        sum(
            weight
            for _, weight
            in ordered[:3]
        )
    )

    # ========================================================
    # MAXIMUM SINGLE POSITION
    # ========================================================

    if (
        max_single
        > max_single_limit
        + 1e-6
    ):

        findings.append(
            _finding(
                severity="HIGH",
                code="MAX_SINGLE_WEIGHT",
                message=(
                    "Largest position exceeds "
                    "portfolio policy."
                ),
                actual=max_single,
                limit=max_single_limit,
            )
        )

    # ========================================================
    # TOP-3 CONCENTRATION
    # ========================================================

    if (
        top3
        > top3_limit
        + 1e-6
    ):

        findings.append(
            _finding(
                severity="HIGH",
                code="TOP3_CONCENTRATION",
                message=(
                    "Top-3 concentration exceeds "
                    "portfolio policy."
                ),
                actual=top3,
                limit=top3_limit,
            )
        )

    # ========================================================
    # AVAILABLE RETURN SERIES
    # ========================================================

    assets = [
        ticker
        for ticker in normalized_weights
        if ticker in returns.columns
    ]

    avg_correlation: float | None = None

    annualized_volatility: (
        float | None
    ) = None

    # ========================================================
    # CORRELATION RISK
    # ========================================================

    if len(assets) >= 2:

        clean_returns = (
            returns[
                assets
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        correlation = (
            clean_returns
            .corr()
            .abs()
        )

        upper_mask = np.triu(
            np.ones(
                correlation.shape,
                dtype=bool,
            ),
            k=1,
        )

        upper_triangle = (
            correlation.where(
                upper_mask
            )
        )

        correlation_values = (
            upper_triangle
            .stack()
        )

        if (
            not
            correlation_values.empty
        ):

            avg_correlation = float(
                correlation_values.mean()
            )

        else:

            avg_correlation = 0.0

        if (
            avg_correlation
            > correlation_limit
            + 1e-6
        ):

            findings.append(
                _finding(
                    severity="MEDIUM",
                    code=(
                        "PAIRWISE_CORRELATION"
                    ),
                    message=(
                        "Average pairwise absolute "
                        "correlation exceeds policy."
                    ),
                    actual=(
                        avg_correlation
                    ),
                    limit=(
                        correlation_limit
                    ),
                )
            )

    # ========================================================
    # ANNUALIZED VOLATILITY
    # ========================================================

    if (
        len(assets) >= 2
        and annualization_factor
        is not None
        and volatility_limit
        is not None
    ):

        clean_returns = (
            returns[
                assets
            ]
            .apply(
                pd.to_numeric,
                errors="coerce",
            )
        )

        weight_vector = np.array(
            [
                normalized_weights[
                    ticker
                ]
                for ticker in assets
            ],
            dtype=float,
        )

        vector_sum = float(
            weight_vector.sum()
        )

        if vector_sum > 0:

            weight_vector = (
                weight_vector
                / vector_sum
            )

        covariance = (
            clean_returns.cov()
            * float(
                annualization_factor
            )
        ).values

        if (
            covariance.size > 0
            and np.all(
                np.isfinite(
                    covariance
                )
            )
        ):

            variance = float(
                weight_vector
                @ covariance
                @ weight_vector
            )

            annualized_volatility = float(
                np.sqrt(
                    max(
                        variance,
                        0.0,
                    )
                )
            )

            if (
                annualized_volatility
                > float(
                    volatility_limit
                )
                + 1e-6
            ):

                findings.append(
                    _finding(
                        severity="HIGH",
                        code="VOLATILITY",
                        message=(
                            "Portfolio annualized "
                            "volatility exceeds policy."
                        ),
                        actual=(
                            annualized_volatility
                        ),
                        limit=float(
                            volatility_limit
                        ),
                    )
                )

    # ========================================================
    # MAXIMUM DRAWDOWN
    # ========================================================

    if (
        max_drawdown is not None
        and drawdown_limit
        is not None
    ):

        observed_drawdown = abs(
            float(
                max_drawdown
            )
        )

        if (
            observed_drawdown
            > float(
                drawdown_limit
            )
            + 1e-6
        ):

            findings.append(
                _finding(
                    severity="HIGH",
                    code="MAX_DRAWDOWN",
                    message=(
                        "Maximum drawdown exceeds "
                        "portfolio policy."
                    ),
                    actual=(
                        observed_drawdown
                    ),
                    limit=float(
                        drawdown_limit
                    ),
                )
            )

    # ========================================================
    # FINAL RISK STATUS
    # ========================================================

    high_findings = [
        finding
        for finding in findings
        if (
            finding.severity
            == "HIGH"
        )
    ]

    status = (
        "REJECT"
        if high_findings
        else "PASS"
    )

    # ========================================================
    # RETURN DOMAIN MODEL
    # ========================================================

    return RiskReview(
        status=status,

        findings=findings,

        max_single_weight=(
            max_single
        ),

        top3_concentration=(
            top3
        ),

        avg_pairwise_correlation=(
            avg_correlation
        ),

        annualized_volatility=(
            annualized_volatility
        ),

        positions=len(
            positive_weights
        ),
    )