from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from quanta.domain.models import (
    OptimizationResult,
)


def optimize_risk_constrained_sharpe(
    returns: pd.DataFrame,
    risk_free_rate: float,
    annualization_factor: int,
    max_weight: float,
    max_top3_concentration: float,
    minimum_weights: dict[str, float] | None = None,
    restarts: int = 12,
) -> OptimizationResult:
    """
    QUANTA risk-constrained maximum-Sharpe optimizer.

    Objective
    ---------
    Maximize:

        (Expected Return - Risk-Free Rate)
        ----------------------------------
             Portfolio Volatility

    Subject to
    ----------
    1. Long-only portfolio
    2. Sum(weights) = 1
    3. Individual weight <= max_weight
    4. Optional minimum weights
    5. Sum of the three largest weights
       <= max_top3_concentration

    The top-3 rule is implemented by constraining every
    possible three-asset combination.

    If every combination satisfies:

        wi + wj + wk <= limit

    then the three largest portfolio weights must also
    satisfy the limit.
    """

    # ---------------------------------------------------------
    # CLEAN INPUT RETURNS
    # ---------------------------------------------------------

    clean = (
        returns
        .copy()
        .apply(
            pd.to_numeric,
            errors="coerce",
        )
        .dropna(
            axis=1,
            how="all",
        )
        .dropna(
            axis=0,
            how="all",
        )
    )

    assets = list(
        clean.columns
    )

    n_assets = len(
        assets
    )

    if n_assets < 2:
        raise ValueError(
            "Risk-constrained optimization requires "
            "at least two assets."
        )

    if annualization_factor <= 0:
        raise ValueError(
            "annualization_factor must be positive."
        )

    if max_weight <= 0:
        raise ValueError(
            "max_weight must be positive."
        )

    if not (
        0
        < max_top3_concentration
        <= 1
    ):
        raise ValueError(
            "max_top3_concentration must be "
            "between 0 and 1."
        )

    minimum_weights = (
        minimum_weights
        or {}
    )

    # ---------------------------------------------------------
    # BASIC FEASIBILITY CHECKS
    # ---------------------------------------------------------

    if (
        n_assets
        * max_weight
        < 1.0 - 1e-8
    ):
        raise ValueError(
            "Infeasible portfolio: number of assets "
            "multiplied by max_weight is below 1."
        )

    minimum_total = sum(
        float(
            minimum_weights.get(
                asset,
                0.0,
            )
        )
        for asset in assets
    )

    if minimum_total > 1.0 + 1e-8:
        raise ValueError(
            "Infeasible portfolio: minimum weights "
            "sum to more than 100%."
        )

    # A top-3 constraint can itself make the portfolio
    # impossible when too few assets are available.
    #
    # Example:
    # 3 assets and top3 <= 0.50 can never sum to 1.
    if (
        n_assets <= 3
        and max_top3_concentration
        < 1.0 - 1e-8
    ):
        raise ValueError(
            "Infeasible top-3 concentration policy "
            "for three or fewer assets."
        )

    # ---------------------------------------------------------
    # EXPECTED RETURNS
    # ---------------------------------------------------------

    expected_returns = (
        clean.mean()
        * annualization_factor
    )

    mu = (
        expected_returns
        .reindex(
            assets
        )
        .values
        .astype(float)
    )

    # ---------------------------------------------------------
    # COVARIANCE MATRIX
    # ---------------------------------------------------------

    covariance_df = (
        clean.cov()
        * annualization_factor
    )

    covariance = (
        covariance_df
        .reindex(
            index=assets,
            columns=assets,
        )
        .values
        .astype(float)
    )

    if not np.all(
        np.isfinite(
            covariance
        )
    ):
        raise ValueError(
            "Covariance matrix contains "
            "non-finite values."
        )

    # Small ridge term for numerical stability.
    covariance = (
        covariance
        + np.eye(
            n_assets
        )
        * 1e-10
    )

    # ---------------------------------------------------------
    # PORTFOLIO FUNCTIONS
    # ---------------------------------------------------------

    def portfolio_return(
        weights: np.ndarray,
    ) -> float:

        return float(
            weights @ mu
        )

    def portfolio_volatility(
        weights: np.ndarray,
    ) -> float:

        variance = float(
            weights
            @ covariance
            @ weights
        )

        return float(
            np.sqrt(
                max(
                    variance,
                    1e-16,
                )
            )
        )

    def portfolio_sharpe(
        weights: np.ndarray,
    ) -> float:

        expected_return = (
            portfolio_return(
                weights
            )
        )

        volatility = (
            portfolio_volatility(
                weights
            )
        )

        if volatility <= 0:
            return -np.inf

        return float(
            (
                expected_return
                - risk_free_rate
            )
            / volatility
        )

    def objective(
        weights: np.ndarray,
    ) -> float:

        sharpe = (
            portfolio_sharpe(
                weights
            )
        )

        if not np.isfinite(
            sharpe
        ):
            return 1e12

        return -sharpe

    # ---------------------------------------------------------
    # POSITION BOUNDS
    # ---------------------------------------------------------

    bounds: list[
        tuple[float, float]
    ] = []

    for asset in assets:
        lower = float(
            minimum_weights.get(
                asset,
                0.0,
            )
        )

        upper = float(
            max_weight
        )

        if lower < 0:
            raise ValueError(
                f"Minimum weight for {asset} "
                "cannot be negative."
            )

        if lower > upper:
            raise ValueError(
                f"Infeasible bounds for {asset}: "
                f"minimum={lower:.4f}, "
                f"maximum={upper:.4f}"
            )

        bounds.append(
            (
                lower,
                upper,
            )
        )

    # ---------------------------------------------------------
    # CONSTRAINT: SUM OF WEIGHTS = 1
    # ---------------------------------------------------------

    constraints: list[dict] = [
        {
            "type": "eq",
            "fun": lambda w: (
                float(
                    np.sum(w)
                )
                - 1.0
            ),
        }
    ]

    # ---------------------------------------------------------
    # CONSTRAINT: TOP-3 CONCENTRATION
    # ---------------------------------------------------------

    for (
        i,
        j,
        k,
    ) in combinations(
        range(
            n_assets
        ),
        3,
    ):
        constraints.append(
            {
                "type": "ineq",

                "fun": (
                    lambda w,
                    i=i,
                    j=j,
                    k=k:
                    float(
                        max_top3_concentration
                        - (
                            w[i]
                            + w[j]
                            + w[k]
                        )
                    )
                ),
            }
        )

    # ---------------------------------------------------------
    # INITIAL FEASIBLE PORTFOLIO
    # ---------------------------------------------------------

    def equal_weight_initial() -> np.ndarray:
        """
        Equal weighting is naturally feasible for QUANTA's
        normal 10-stock final universe under a 50% top-3
        concentration policy.
        """

        initial = np.full(
            n_assets,
            1.0 / n_assets,
            dtype=float,
        )

        # Respect explicit minimums.
        for index, asset in enumerate(
            assets
        ):
            minimum = float(
                minimum_weights.get(
                    asset,
                    0.0,
                )
            )

            initial[index] = max(
                initial[index],
                minimum,
            )

        total = float(
            initial.sum()
        )

        if total <= 0:
            raise RuntimeError(
                "Unable to create initial portfolio."
            )

        initial = (
            initial / total
        )

        return initial

    # ---------------------------------------------------------
    # MULTI-START OPTIMIZATION
    # ---------------------------------------------------------

    rng = np.random.default_rng(
        42
    )

    best_result = None

    number_of_restarts = max(
        int(
            restarts
        ),
        1,
    )

    for restart in range(
        number_of_restarts
    ):
        if restart == 0:
            initial = (
                equal_weight_initial()
            )

        else:
            random_weights = (
                rng.dirichlet(
                    np.ones(
                        n_assets
                    )
                )
            )

            lower_bounds = np.array(
                [
                    bound[0]
                    for bound in bounds
                ],
                dtype=float,
            )

            upper_bounds = np.array(
                [
                    bound[1]
                    for bound in bounds
                ],
                dtype=float,
            )

            initial = np.clip(
                random_weights,
                lower_bounds,
                upper_bounds,
            )

            # Blend random initialization with equal
            # weights. This greatly increases the chance
            # of starting close to the feasible top-3
            # concentration region.
            equal = (
                equal_weight_initial()
            )

            initial = (
                0.50 * initial
                + 0.50 * equal
            )

            initial = (
                initial
                / initial.sum()
            )

        result = minimize(
            objective,
            initial,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={
                "maxiter": 3000,
                "ftol": 1e-10,
                "disp": False,
            },
        )

        if not result.success:
            continue

        candidate = np.array(
            result.x,
            dtype=float,
        )

        # -----------------------------------------------------
        # VERIFY CONSTRAINTS MANUALLY
        # -----------------------------------------------------

        if abs(
            candidate.sum()
            - 1.0
        ) > 1e-5:
            continue

        if (
            candidate.max()
            > max_weight
            + 1e-5
        ):
            continue

        ordered_weights = np.sort(
            candidate
        )[::-1]

        top3 = float(
            ordered_weights[:3]
            .sum()
        )

        if (
            top3
            > max_top3_concentration
            + 1e-5
        ):
            continue

        minimum_violation = False

        for index, asset in enumerate(
            assets
        ):
            minimum = float(
                minimum_weights.get(
                    asset,
                    0.0,
                )
            )

            if (
                candidate[index]
                < minimum
                - 1e-5
            ):
                minimum_violation = True
                break

        if minimum_violation:
            continue

        if (
            best_result is None
            or result.fun
            < best_result.fun
        ):
            best_result = result

    # ---------------------------------------------------------
    # ENSURE OPTIMIZATION SUCCEEDED
    # ---------------------------------------------------------

    if best_result is None:
        raise RuntimeError(
            "Risk-constrained portfolio optimization "
            "failed to find a feasible solution."
        )

    weights_array = np.array(
        best_result.x,
        dtype=float,
    )

    # Remove tiny numerical noise.
    weights_array[
        np.abs(
            weights_array
        )
        < 1e-10
    ] = 0.0

    # ---------------------------------------------------------
    # FINAL METRICS
    # ---------------------------------------------------------

    expected_return = (
        portfolio_return(
            weights_array
        )
    )

    annualized_volatility = (
        portfolio_volatility(
            weights_array
        )
    )

    sharpe_ratio = (
        (
            expected_return
            - risk_free_rate
        )
        / annualized_volatility
        if annualized_volatility > 0
        else float("nan")
    )

    weights = {
        asset: float(
            weight
        )
        for asset, weight
        in zip(
            assets,
            weights_array,
        )
    }

    # ---------------------------------------------------------
    # FINAL DEFENSIVE VALIDATION
    # ---------------------------------------------------------

    sorted_weights = sorted(
        weights.values(),
        reverse=True,
    )

    final_top3 = sum(
        sorted_weights[:3]
    )

    if (
        final_top3
        > max_top3_concentration
        + 1e-5
    ):
        raise RuntimeError(
            "Optimizer returned a portfolio that "
            "violates top-3 concentration policy."
        )

    if (
        max(
            sorted_weights
        )
        > max_weight
        + 1e-5
    ):
        raise RuntimeError(
            "Optimizer returned a portfolio that "
            "violates maximum-position policy."
        )

    # ---------------------------------------------------------
    # RETURN ORIGINAL QUANTA DOMAIN MODEL
    # ---------------------------------------------------------

    return OptimizationResult(
        success=True,

        message=str(
            best_result.message
        ),

        weights=weights,

        expected_return=float(
            expected_return
        ),

        annualized_volatility=float(
            annualized_volatility
        ),

        sharpe_ratio=float(
            sharpe_ratio
        ),
    )