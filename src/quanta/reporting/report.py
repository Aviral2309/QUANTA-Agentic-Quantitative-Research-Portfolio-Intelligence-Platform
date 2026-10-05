from __future__ import annotations

from pathlib import Path


def _pct(
    value,
) -> str:

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.2%}"

    except (TypeError, ValueError):
        return "N/A"


def _number(
    value,
    decimals: int = 3,
) -> str:

    if value is None:
        return "N/A"

    try:
        return f"{float(value):.{decimals}f}"

    except (TypeError, ValueError):
        return "N/A"


def build_research_report(
    run_id: str,
    cfg,
    phase2: dict,
    risk_review: dict,
    verdict: dict,
    walk_forward: dict | None = None,
    risk_free_metadata: dict | None = None,
    audit_summary: list[str] | None = None,
) -> str:

    solver = phase2[
        "solver2"
    ]

    backtest = phase2[
        "backtest"
    ]

    capm = phase2.get(
        "capm_results"
    )

    positive_alpha_count = (
        int(
            (
                capm["classification"]
                == "positive_alpha"
            ).sum()
        )
        if capm is not None
        and not capm.empty
        else None
    )

    lines = [
        "# QUANTA Research Report",
        "",
        f"**Run ID:** `{run_id}`",
        "",
        (
            "**Final status:** "
            f"**{verdict['display_status']}**"
        ),
        "",
        "---",
        "",
        "## 1. Research Configuration",
        "",
        f"- Research date: {cfg.research_date}",
        f"- Research period: {cfg.start_date} → {cfg.end_date}",
        f"- Benchmark: `{cfg.benchmark_ticker}`",
        (
            "- Annualization factor: "
            f"{cfg.annualization_factor}"
        ),
        (
            "- Validation fraction: "
            f"{cfg.validation_fraction:.0%}"
        ),
    ]

    if risk_free_metadata:

        lines.extend(
            [
                (
                    "- Risk-free rate: "
                    f"{_pct(risk_free_metadata.get('rate'))}"
                ),
                (
                    "- Risk-free source: "
                    f"{risk_free_metadata.get('source')}"
                ),
                (
                    "- Risk-free observation date: "
                    f"{risk_free_metadata.get('observation_date')}"
                ),
            ]
        )

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
                f"- Large-cap extreme: "
                f"Top {cfg.universe.top_n}"
            ),
            (
                f"- Small-cap extreme: "
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
        solver.weights.items(),
        key=lambda item: item[1],
        reverse=True,
    ):

        lines.append(
            f"| {ticker} | {_pct(weight)} |"
        )

    lines.extend(
        [
            "",
            "## 4. Risk Review",
            "",
            (
                "**Risk status:** "
                f"{risk_review.get('status')}"
            ),
            "",
            (
                "- Largest position: "
                f"{_pct(risk_review.get('max_single_weight'))}"
            ),
            (
                "- Top-3 concentration: "
                f"{_pct(risk_review.get('top3_concentration'))}"
            ),
            "",
        ]
    )

    findings = risk_review.get(
        "findings",
        [],
    )

    if findings:

        lines.extend(
            [
                "| Severity | Code | Finding |",
                "|---|---|---|",
            ]
        )

        for finding in findings:

            lines.append(
                "| "
                f"{finding.get('severity')} | "
                f"{finding.get('code')} | "
                f"{finding.get('message')} |"
            )

    else:

        lines.append(
            "No material risk-policy violations "
            "were detected."
        )

    lines.extend(
        [
            "",
            "## 5. Held-Out Validation",
            "",
            (
                "- CAGR: "
                f"{_pct(backtest.get('cagr'))}"
            ),
            (
                "- Annualized volatility: "
                f"{_pct(backtest.get('volatility'))}"
            ),
            (
                "- Sharpe ratio: "
                f"{_number(backtest.get('sharpe'))}"
            ),
            (
                "- Maximum drawdown: "
                f"{_pct(backtest.get('max_drawdown'))}"
            ),
            (
                "- Performance classification: "
                f"**{verdict['performance_status']}**"
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

    if walk_forward:

        lines.extend(
            [
                "",
                "## 6. Walk-Forward Validation",
                "",
                (
                    "- Status: "
                    f"{walk_forward.get('status')}"
                ),
                (
                    "- Successful windows: "
                    f"{walk_forward.get('windows', 0)}"
                ),
            ]
        )

        if (
            walk_forward.get("status")
            == "COMPLETED"
        ):

            overall = (
                walk_forward.get(
                    "overall",
                    {},
                )
            )

            lines.extend(
                [
                    (
                        "- Profitable-window ratio: "
                        f"{_pct(walk_forward.get('profitable_window_ratio'))}"
                    ),
                    (
                        "- Positive-Sharpe-window ratio: "
                        f"{_pct(walk_forward.get('positive_sharpe_window_ratio'))}"
                    ),
                    (
                        "- Combined walk-forward CAGR: "
                        f"{_pct(overall.get('cagr'))}"
                    ),
                    (
                        "- Combined walk-forward Sharpe: "
                        f"{_number(overall.get('sharpe'))}"
                    ),
                    (
                        "- Combined walk-forward drawdown: "
                        f"{_pct(overall.get('max_drawdown'))}"
                    ),
                ]
            )

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
            "## 8. Research Verdict",
            "",
            (
                "- Validation: "
                f"**{verdict['validation_status']}**"
            ),
            (
                "- Performance: "
                f"**{verdict['performance_status']}**"
            ),
            (
                "- Walk-forward: "
                f"**{verdict['walk_forward_status']}**"
            ),
            "",
            "## 9. Limitations",
            "",
            (
                "- Current-index constituents may "
                "introduce survivorship bias when "
                "used historically."
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
        ]
    )

    return "\n".join(
        lines
    )


def save_research_report(
    path: str | Path,
    content: str,
) -> Path:

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content,
        encoding="utf-8",
    )

    return path