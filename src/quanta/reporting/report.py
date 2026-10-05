from __future__ import annotations

from pathlib import Path
from typing import Any


# ============================================================
# GENERIC COMPATIBILITY HELPERS
# ============================================================


def _value(
    obj: Any,
    key: str,
    default=None,
):
    """
    Safely read a value from:

    - dictionary
    - Pydantic model
    - dataclass/domain object
    - normal Python object

    This allows the reporting layer to consume both older
    dictionary-based QUANTA outputs and newer typed models.
    """

    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(
            key,
            default,
        )

    return getattr(
        obj,
        key,
        default,
    )


def _as_dict(
    obj: Any,
) -> dict:
    """
    Convert common QUANTA objects into dictionaries.

    Supports:
    - dict
    - Pydantic v2
    - Pydantic v1
    - ordinary objects
    """

    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    if hasattr(obj, "dict"):
        return obj.dict()

    if hasattr(obj, "__dict__"):
        return dict(obj.__dict__)

    return {}


# ============================================================
# FORMATTING HELPERS
# ============================================================


def _pct(
    value,
) -> str:
    """
    Format a decimal as percentage.

    Example:
        0.152 -> 15.20%
    """

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2%}"

    except (
        TypeError,
        ValueError,
    ):
        return "N/A"


def _number(
    value,
    decimals: int = 3,
) -> str:
    """
    Format numeric values safely.
    """

    if value is None:
        return "N/A"

    try:
        return (
            f"{float(value):.{decimals}f}"
        )

    except (
        TypeError,
        ValueError,
    ):
        return "N/A"


def _integer(
    value,
) -> str:
    """
    Format integer-like values safely.
    """

    if value is None:
        return "N/A"

    try:
        return str(
            int(value)
        )

    except (
        TypeError,
        ValueError,
    ):
        return "N/A"


# ============================================================
# RESEARCH REPORT
# ============================================================


def build_research_report(
    run_id: str,
    cfg,
    phase2: dict,
    risk_review,
    verdict: dict,
    walk_forward=None,
    risk_free_metadata: dict | None = None,
    audit_summary: list[str] | None = None,
) -> str:
    """
    Build the final QUANTA research report.

    Important architecture rule:

        Agents reason.
        Tools calculate.
        Harness controls.
        Validators verify.
        Humans approve.

    This report consumes already-computed research outputs.
    It does not alter portfolio weights or feed held-out
    results back into optimization.
    """

    # ========================================================
    # PHASE 2 RESULTS
    # ========================================================

    solver = phase2[
        "solver2"
    ]

    backtest = phase2[
        "backtest"
    ]

    capm = phase2.get(
        "capm_results"
    )

    # ========================================================
    # CAPM SUMMARY
    # ========================================================

    positive_alpha_count = (
        int(
            (
                capm[
                    "classification"
                ]
                == "positive_alpha"
            ).sum()
        )
        if (
            capm is not None
            and not capm.empty
        )
        else None
    )

    # ========================================================
    # NORMALIZE SOLVER DATA
    # ========================================================

    solver_weights = _value(
        solver,
        "weights",
        {},
    )

    if solver_weights is None:
        solver_weights = {}

    solver_expected_return = _value(
        solver,
        "expected_return",
        None,
    )

    solver_volatility = _value(
        solver,
        "annualized_volatility",
        _value(
            solver,
            "volatility",
            None,
        ),
    )

    solver_sharpe = _value(
        solver,
        "sharpe_ratio",
        _value(
            solver,
            "sharpe",
            None,
        ),
    )

    # ========================================================
    # NORMALIZE RISK DATA
    # ========================================================

    risk_status = _value(
        risk_review,
        "status",
        "UNKNOWN",
    )

    risk_max_single = _value(
        risk_review,
        "max_single_weight",
        None,
    )

    risk_top3 = _value(
        risk_review,
        "top3_concentration",
        None,
    )

    risk_correlation = _value(
        risk_review,
        "avg_pairwise_correlation",
        None,
    )

    risk_volatility = _value(
        risk_review,
        "annualized_volatility",
        None,
    )

    risk_positions = _value(
        risk_review,
        "positions",
        None,
    )

    risk_findings = _value(
        risk_review,
        "findings",
        [],
    )

    if risk_findings is None:
        risk_findings = []

    # ========================================================
    # NORMALIZE BACKTEST DATA
    # ========================================================

    backtest_cagr = _value(
        backtest,
        "cagr",
        None,
    )

    backtest_volatility = _value(
        backtest,
        "annualized_volatility",
        _value(
            backtest,
            "volatility",
            None,
        ),
    )

    backtest_sharpe = _value(
        backtest,
        "sharpe",
        _value(
            backtest,
            "sharpe_ratio",
            None,
        ),
    )

    backtest_drawdown = _value(
        backtest,
        "max_drawdown",
        None,
    )

    backtest_total_return = _value(
        backtest,
        "total_return",
        None,
    )

    backtest_observations = _value(
        backtest,
        "observations",
        _value(
            backtest,
            "test_observations",
            _value(
                backtest,
                "n_observations",
                verdict.get(
                    "test_observations"
                ),
            ),
        ),
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
    # REPORT HEADER
    # ========================================================

    lines = [
        "# QUANTA Research Report",
        "",
        f"**Run ID:** `{run_id}`",
        "",
        (
            "**Final status:** "
            f"**{verdict.get('display_status', 'UNKNOWN')}**"
        ),
        "",
        "---",
        "",
        "## 1. Research Configuration",
        "",
        (
            "- Research date: "
            f"{cfg.research_date}"
        ),
        (
            "- Research period: "
            f"{cfg.start_date} → "
            f"{cfg.end_date}"
        ),
        (
            "- Benchmark: "
            f"`{cfg.benchmark_ticker}`"
        ),
        (
            "- Annualization factor: "
            f"{cfg.annualization_factor}"
        ),
        (
            "- Validation fraction: "
            f"{cfg.validation_fraction:.0%}"
        ),
    ]

    # ========================================================
    # RISK-FREE RATE
    # ========================================================

    if risk_free_metadata:

        lines.extend(
            [
                (
                    "- Risk-free rate: "
                    f"{_pct(_value(risk_free_metadata, 'rate'))}"
                ),
                (
                    "- Risk-free source: "
                    f"{_value(risk_free_metadata, 'source', 'N/A')}"
                ),
                (
                    "- Risk-free observation date: "
                    f"{_value(risk_free_metadata, 'observation_date', 'N/A')}"
                ),
            ]
        )

    # ========================================================
    # UNIVERSE CONSTRUCTION
    # ========================================================

    lines.extend(
        [
            "",
            "## 2. Universe Construction",
            "",
            (
                "- Initial universe: "
                "NIFTY 500 constituents"
            ),
            (
                "- Large-cap extreme: "
                f"Top {cfg.universe.top_n}"
            ),
            (
                "- Small-cap extreme: "
                f"Bottom {cfg.universe.bottom_n}"
            ),
        ]
    )

    if positive_alpha_count is not None:

        lines.append(
            (
                "- Positive CAPM-screening-alpha "
                f"stocks: {positive_alpha_count}"
            )
        )

    lines.extend(
        [
            "",
            (
                "> **Methodology warning:** "
                "The top/bottom market-cap sample is "
                "deliberately non-representative. "
                "The middle of the NIFTY 500 universe "
                "is omitted."
            ),
            "",
        ]
    )

    # ========================================================
    # PORTFOLIO CONSTRUCTION
    # ========================================================

    lines.extend(
        [
            "## 3. Portfolio Construction",
            "",
            (
                "- Objective: maximum Sharpe "
                "under long-only constraints"
            ),
            (
                "- Core selection: "
                "multi-factor scoring"
            ),
            (
                "- Diversification overlay: "
                "low-correlation selection"
            ),
            (
                "- Final risk-aware optimization: "
                "single-position and top-3 "
                "concentration policies"
            ),
            "",
            "### Final Portfolio",
            "",
            "| Ticker | Weight |",
            "|---|---:|",
        ]
    )

    for ticker, weight in sorted(
        solver_weights.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        lines.append(
            (
                f"| {ticker} | "
                f"{_pct(weight)} |"
            )
        )

    lines.extend(
        [
            "",
            "### Optimizer Statistics",
            "",
            (
                "- Expected annual return: "
                f"{_pct(solver_expected_return)}"
            ),
            (
                "- Expected annual volatility: "
                f"{_pct(solver_volatility)}"
            ),
            (
                "- Expected Sharpe ratio: "
                f"{_number(solver_sharpe)}"
            ),
        ]
    )

    # ========================================================
    # RISK REVIEW
    # ========================================================

    lines.extend(
        [
            "",
            "## 4. Risk Review",
            "",
            (
                "**Risk status:** "
                f"**{risk_status}**"
            ),
            "",
            (
                "- Number of positions: "
                f"{_integer(risk_positions)}"
            ),
            (
                "- Largest position: "
                f"{_pct(risk_max_single)}"
            ),
            (
                "- Top-3 concentration: "
                f"{_pct(risk_top3)}"
            ),
            (
                "- Average pairwise correlation: "
                f"{_number(risk_correlation)}"
            ),
            (
                "- Estimated annualized volatility: "
                f"{_pct(risk_volatility)}"
            ),
            "",
        ]
    )

    if risk_findings:

        lines.extend(
            [
                "### Risk Findings",
                "",
                "| Severity | Code | Finding |",
                "|---|---|---|",
            ]
        )

        for finding in risk_findings:

            severity = _value(
                finding,
                "severity",
                "UNKNOWN",
            )

            code = _value(
                finding,
                "code",
                "UNKNOWN",
            )

            message = _value(
                finding,
                "message",
                "",
            )

            # Protect Markdown table structure.
            message = str(
                message
            ).replace(
                "|",
                "\\|",
            )

            lines.append(
                (
                    f"| {severity} | "
                    f"{code} | "
                    f"{message} |"
                )
            )

    else:

        lines.append(
            (
                "No material risk-policy violations "
                "were detected."
            )
        )

    # ========================================================
    # HELD-OUT VALIDATION
    # ========================================================

    lines.extend(
        [
            "",
            "## 5. Held-Out Validation",
            "",
            (
                "- Test observations: "
                f"{_integer(backtest_observations)}"
            ),
            (
                "- Total return: "
                f"{_pct(backtest_total_return)}"
            ),
            (
                "- CAGR: "
                f"{_pct(backtest_cagr)}"
            ),
            (
                "- Annualized volatility: "
                f"{_pct(backtest_volatility)}"
            ),
            (
                "- Sharpe ratio: "
                f"{_number(backtest_sharpe)}"
            ),
            (
                "- Maximum drawdown: "
                f"{_pct(backtest_drawdown)}"
            ),
            (
                "- Performance classification: "
                f"**{verdict.get('performance_status', 'UNKNOWN')}**"
            ),
            "",
            (
                "> The held-out test period is used "
                "for evaluation only. Poor test "
                "performance does not trigger "
                "portfolio re-optimization."
            ),
        ]
    )

    # ========================================================
    # BENCHMARK COMPARISON
    # ========================================================

    if (
        benchmark_cagr is not None
        or benchmark_total_return is not None
    ):

        lines.extend(
            [
                "",
                "### Benchmark Comparison",
                "",
                (
                    "- Benchmark CAGR: "
                    f"{_pct(benchmark_cagr)}"
                ),
                (
                    "- Benchmark total return: "
                    f"{_pct(benchmark_total_return)}"
                ),
            ]
        )

    # ========================================================
    # WALK-FORWARD VALIDATION
    # ========================================================

    if walk_forward:

        wf_status = _value(
            walk_forward,
            "status",
            "UNKNOWN",
        )

        wf_windows = _value(
            walk_forward,
            "windows",
            _value(
                walk_forward,
                "window_count",
                _value(
                    walk_forward,
                    "n_windows",
                    0,
                ),
            ),
        )

        # windows may itself be a list.
        if isinstance(
            wf_windows,
            (
                list,
                tuple,
            ),
        ):
            wf_window_count = len(
                wf_windows
            )
        else:
            wf_window_count = wf_windows

        profitable_ratio = _value(
            walk_forward,
            "profitable_window_ratio",
            None,
        )

        positive_sharpe_ratio = _value(
            walk_forward,
            "positive_sharpe_window_ratio",
            None,
        )

        overall = _value(
            walk_forward,
            "overall",
            {},
        )

        if overall is None:
            overall = {}

        overall_cagr = _value(
            overall,
            "cagr",
            None,
        )

        overall_sharpe = _value(
            overall,
            "sharpe",
            _value(
                overall,
                "sharpe_ratio",
                None,
            ),
        )

        overall_drawdown = _value(
            overall,
            "max_drawdown",
            None,
        )

        overall_volatility = _value(
            overall,
            "annualized_volatility",
            _value(
                overall,
                "volatility",
                None,
            ),
        )

        lines.extend(
            [
                "",
                "## 6. Walk-Forward Validation",
                "",
                (
                    "- Status: "
                    f"**{wf_status}**"
                ),
                (
                    "- Successful windows: "
                    f"{_integer(wf_window_count)}"
                ),
            ]
        )

        if str(
            wf_status
        ).upper() == "COMPLETED":

            lines.extend(
                [
                    (
                        "- Profitable-window ratio: "
                        f"{_pct(profitable_ratio)}"
                    ),
                    (
                        "- Positive-Sharpe-window ratio: "
                        f"{_pct(positive_sharpe_ratio)}"
                    ),
                    (
                        "- Combined walk-forward CAGR: "
                        f"{_pct(overall_cagr)}"
                    ),
                    (
                        "- Combined walk-forward volatility: "
                        f"{_pct(overall_volatility)}"
                    ),
                    (
                        "- Combined walk-forward Sharpe: "
                        f"{_number(overall_sharpe)}"
                    ),
                    (
                        "- Combined walk-forward drawdown: "
                        f"{_pct(overall_drawdown)}"
                    ),
                ]
            )

        lines.extend(
            [
                "",
                (
                    "> Current walk-forward validation "
                    "re-optimizes the selected asset set "
                    "using chronological training windows. "
                    "It does not yet reconstruct the full "
                    "historical NIFTY 500 universe and "
                    "point-in-time selection process for "
                    "every rebalance date."
                ),
            ]
        )

    else:

        lines.extend(
            [
                "",
                "## 6. Walk-Forward Validation",
                "",
                "- Status: **NOT_RUN**",
            ]
        )

    # ========================================================
    # AGENTIC DECISION TRACE
    # ========================================================

    lines.extend(
        [
            "",
            "## 7. Agentic Decision Trace",
            "",
        ]
    )

    if audit_summary:

        for event in audit_summary:

            lines.append(
                f"- {event}"
            )

    else:

        lines.append(
            (
                "- Deterministic planner → risk "
                "critic → constrained optimizer → "
                "validator → report."
            )
        )

    lines.extend(
        [
            "",
            (
                "> QUANTA follows the principle: "
                "**Agents reason. Tools calculate. "
                "The harness controls. Validators "
                "verify. Humans approve.**"
            ),
        ]
    )

    # ========================================================
    # FINAL RESEARCH VERDICT
    # ========================================================

    lines.extend(
        [
            "",
            "## 8. Research Verdict",
            "",
            (
                "- Validation: "
                f"**{verdict.get('validation_status', 'UNKNOWN')}**"
            ),
            (
                "- Risk: "
                f"**{verdict.get('risk_status', risk_status)}**"
            ),
            (
                "- Performance: "
                f"**{verdict.get('performance_status', 'UNKNOWN')}**"
            ),
            (
                "- Walk-forward: "
                f"**{verdict.get('walk_forward_status', 'UNKNOWN')}**"
            ),
            (
                "- Overall research status: "
                f"**{verdict.get('display_status', 'UNKNOWN')}**"
            ),
        ]
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    display_status = verdict.get(
        "display_status",
        "UNKNOWN",
    )

    lines.extend(
        [
            "",
            "### Interpretation",
            "",
        ]
    )

    if (
        display_status
        == "VALIDATED_WITH_WEAK_PERFORMANCE"
    ):

        lines.append(
            (
                "The research pipeline completed its "
                "methodological validation, but the "
                "portfolio demonstrated weak held-out "
                "performance. This is retained as a "
                "valid negative research result and "
                "does not trigger optimization against "
                "the test period."
            )
        )

    elif (
        display_status
        == "RISK_REJECTED"
    ):

        lines.append(
            (
                "The portfolio failed one or more "
                "configured risk policies after the "
                "allowed bounded re-optimization "
                "attempts. The portfolio is therefore "
                "reported as risk-rejected."
            )
        )

    elif (
        display_status
        == "INSUFFICIENT_DATA"
    ):

        lines.append(
            (
                "The available held-out dataset was "
                "insufficient for the configured "
                "validation requirement. No strong "
                "performance conclusion should be "
                "drawn from this run."
            )
        )

    elif (
        display_status
        == "VALIDATED"
    ):

        lines.append(
            (
                "The portfolio passed the configured "
                "risk and validation checks and "
                "produced positive held-out research "
                "performance under the current "
                "methodology."
            )
        )

    else:

        lines.append(
            (
                "The research run completed with an "
                "unclassified final state. Review the "
                "risk, validation, and audit sections "
                "before interpreting the result."
            )
        )

    # ========================================================
    # LIMITATIONS
    # ========================================================

    lines.extend(
        [
            "",
            "## 9. Limitations",
            "",
            (
                "- Current-index constituents may "
                "introduce survivorship bias when "
                "used historically."
            ),
            (
                "- The top-50/bottom-50 market-cap "
                "sampling approach deliberately omits "
                "the middle of the NIFTY 500 universe "
                "and is therefore non-representative."
            ),
            (
                "- Free market-data endpoints can "
                "contain missing or inconsistent "
                "fundamental fields."
            ),
            (
                "- CAPM screening alpha is not "
                "equivalent to intrinsic-value "
                "undervaluation."
            ),
            (
                "- Fundamentals must be point-in-time "
                "for rigorous historical research."
            ),
            (
                "- News sentiment quality depends on "
                "the supplied dated news dataset."
            ),
            (
                "- Factor-regression conclusions "
                "depend on the supplied India factor "
                "dataset."
            ),
            (
                "- Current walk-forward validation "
                "does not yet rebuild the complete "
                "historical universe, market-cap "
                "ranking, CAPM screen, factor score, "
                "core selection, and diversifier "
                "selection at every historical "
                "rebalance date."
            ),
            (
                "- Historical performance does not "
                "guarantee future performance."
            ),
        ]
    )

    # ========================================================
    # RESEARCH DISCLAIMER
    # ========================================================

    lines.extend(
        [
            "",
            "## 10. Research Disclaimer",
            "",
            (
                "QUANTA is an educational and "
                "quantitative research system. "
                "Outputs do not constitute investment "
                "advice and the system does not "
                "execute real-money trades."
            ),
            "",
            "---",
            "",
            (
                "*Generated by QUANTA — Agentic "
                "Quantitative Research & Portfolio "
                "Intelligence Platform.*"
            ),
        ]
    )

    return "\n".join(
        lines
    )


# ============================================================
# REPORT WRITER
# ============================================================


def save_research_report(
    path: str | Path,
    content: str,
) -> Path:
    """
    Persist the generated Markdown research report.
    """

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content,
        encoding="utf-8",
    )

    return path