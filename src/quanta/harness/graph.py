from __future__ import annotations

from pathlib import Path

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from quanta.data.risk_free import (
    load_risk_free_rate,
)
from quanta.harness.audit import (
    AuditLogger,
)
from quanta.harness.state import (
    ResearchState,
)
from quanta.portfolio.risk_optimizer import (
    optimize_risk_constrained_sharpe,
)
from quanta.reporting.export import (
    write_json,
)
from quanta.reporting.report import (
    build_research_report,
    save_research_report,
)
from quanta.validation.backtest import (
    backtest_static,
)
from quanta.validation.risk import (
    review_portfolio,
)
from quanta.validation.verdict import (
    determine_research_verdict,
)
from quanta.validation.walk_forward import (
    run_walk_forward_validation,
)


def build_research_graph(
    cfg,
    audit: AuditLogger,
):

    # ---------------------------------------------------------
    # RISK-FREE NODE
    # ---------------------------------------------------------

    def risk_free_node(
        state: ResearchState,
    ) -> ResearchState:

        rate, metadata = (
            load_risk_free_rate(
                csv_path=(
                    cfg.risk_free.series_csv
                ),
                date_column=(
                    cfg.risk_free.date_column
                ),
                rate_column=(
                    cfg.risk_free.rate_column
                ),
                research_date=(
                    cfg.research_date
                ),
                fallback_rate=(
                    cfg.risk_free.annual_rate
                ),
                fallback_to_static=(
                    cfg.risk_free.fallback_to_static
                ),
            )
        )

        audit.log(
            "risk_free_loaded",
            metadata,
        )

        return {
            **state,
            "risk_free_rate": rate,
            "risk_free_metadata": metadata,
        }

    # ---------------------------------------------------------
    # PLANNER
    # ---------------------------------------------------------

    def planner(
        state: ResearchState,
    ) -> ResearchState:

        plan = {
            "objective": (
                "Construct and validate a "
                "risk-constrained quantitative "
                "research portfolio."
            ),

            "rules": [
                (
                    "Use deterministic tools for "
                    "financial calculations."
                ),
                (
                    "Enforce portfolio risk limits "
                    "before held-out validation."
                ),
                (
                    "Never optimize against held-out "
                    "test performance."
                ),
                (
                    "Run walk-forward validation "
                    "after portfolio methodology is "
                    "defined."
                ),
            ],
        }

        audit.log(
            "planner_completed",
            plan,
        )

        return {
            **state,
            "plan": plan,
        }

    # ---------------------------------------------------------
    # RISK CRITIC
    # ---------------------------------------------------------

    def risk_node(
        state: ResearchState,
    ) -> ResearchState:

        phase2 = state[
            "phase2"
        ]

        solver = phase2[
            "solver2"
        ]

        train = phase2[
            "train_returns"
        ]

        review = review_portfolio(
            weights=solver.weights,
            returns=train,
            annualization_factor=(
                cfg.annualization_factor
            ),
            limits=cfg.risk_limits,
        )

        audit.log(
            "risk_review",
            review,
        )

        return {
            **state,
            "risk_review": review,
        }

    # ---------------------------------------------------------
    # ROUTER AFTER RISK
    # ---------------------------------------------------------

    def risk_router(
        state: ResearchState,
    ) -> str:

        review = state.get(
            "risk_review",
            {},
        )

        loops = state.get(
            "reoptimize_count",
            0,
        )

        if (
            review.get("status")
            == "REJECT"
            and loops
            < cfg.agentic.max_reoptimization_loops
        ):
            return "reoptimize"

        return "walk_forward"

    # ---------------------------------------------------------
    # RISK-CONSTRAINED REOPTIMIZER
    # ---------------------------------------------------------

    def reoptimize(
        state: ResearchState,
    ) -> ResearchState:

        phase2 = dict(
            state["phase2"]
        )

        train = phase2[
            "train_returns"
        ]

        test = phase2[
            "test_returns"
        ]

        final_assets = phase2[
            "final_assets"
        ]

        core = phase2[
            "core"
        ]

        risk_free_rate = state.get(
            "risk_free_rate",
            cfg.risk_free.annual_rate,
        )

        minimum_weights = {
            ticker: (
                cfg.portfolio.minimum_core_weight
            )
            for ticker in core
            if ticker in final_assets
        }

        optimized = (
            optimize_risk_constrained_sharpe(
                returns=train[
                    final_assets
                ],
                risk_free_rate=(
                    risk_free_rate
                ),
                annualization_factor=(
                    cfg.annualization_factor
                ),
                max_weight=(
                    cfg.risk_limits.max_single_weight
                ),
                max_top3_concentration=(
                    cfg.risk_limits.max_top3_concentration
                ),
                minimum_weights=(
                    minimum_weights
                ),
                restarts=(
                    cfg.portfolio.optimizer_restarts
                ),
            )
        )

        # IMPORTANT:
        # test data is used only AFTER the weights
        # have been finalized by training/risk rules.
        benchmark = None

        if (
            cfg.benchmark_ticker
            in test.columns
        ):
            benchmark = test[
                cfg.benchmark_ticker
            ]

        backtest, curve = (
            backtest_static(
                optimized.weights,
                test,
                risk_free_rate,
                cfg.annualization_factor,
                benchmark,
                cfg.portfolio.transaction_cost_bps,
            )
        )

        phase2[
            "solver2"
        ] = optimized

        phase2[
            "backtest"
        ] = backtest

        phase2[
            "backtest_curve"
        ] = curve

        output_dir = Path(
            phase2["run_dir"]
        )

        write_json(
            output_dir
            / "solver2_risk_constrained.json",
            optimized,
        )

        write_json(
            output_dir
            / "backtest_risk_constrained.json",
            backtest,
        )

        curve.to_csv(
            output_dir
            / "backtest_curve_risk_constrained.csv"
        )

        loop = (
            state.get(
                "reoptimize_count",
                0,
            )
            + 1
        )

        audit.log(
            "risk_constrained_reoptimization",
            {
                "loop": loop,
                "weights": optimized.weights,
            },
        )

        return {
            **state,
            "phase2": phase2,
            "reoptimize_count": loop,
        }

    # ---------------------------------------------------------
    # WALK-FORWARD VALIDATION
    # ---------------------------------------------------------

    def walk_forward_node(
        state: ResearchState,
    ) -> ResearchState:

        phase2 = state[
            "phase2"
        ]

        output_dir = Path(
            phase2["run_dir"]
        )

        wf_cfg = (
            cfg.validation.walk_forward
        )

        if not wf_cfg.enabled:

            summary = {
                "status": "DISABLED",
                "windows": 0,
            }

            audit.log(
                "walk_forward_skipped",
                summary,
            )

            return {
                **state,
                "walk_forward": summary,
            }

        # Use full chronological returns for repeated
        # historical train → test windows.
        train = phase2[
            "train_returns"
        ]

        test = phase2[
            "test_returns"
        ]

        full_returns = (
            train
            .combine_first(test)
            .sort_index()
        )

        final_assets = phase2[
            "final_assets"
        ]

        core = phase2[
            "core"
        ]

        minimum_weights = {
            ticker: (
                cfg.portfolio.minimum_core_weight
            )
            for ticker in core
            if ticker in final_assets
        }

        summary, windows = (
            run_walk_forward_validation(
                returns=full_returns,
                assets=final_assets,
                risk_free_rate=(
                    state.get(
                        "risk_free_rate",
                        cfg.risk_free.annual_rate,
                    )
                ),
                annualization_factor=(
                    cfg.annualization_factor
                ),
                max_weight=(
                    cfg.risk_limits.max_single_weight
                ),
                max_top3_concentration=(
                    cfg.risk_limits.max_top3_concentration
                ),
                minimum_weights=(
                    minimum_weights
                ),
                train_days=(
                    wf_cfg.train_days
                ),
                test_days=(
                    wf_cfg.test_days
                ),
                step_days=(
                    wf_cfg.step_days
                ),
                minimum_windows=(
                    wf_cfg.minimum_windows
                ),
                transaction_cost_bps=(
                    cfg.portfolio.transaction_cost_bps
                ),
                optimizer_restarts=(
                    cfg.portfolio.optimizer_restarts
                ),
            )
        )

        write_json(
            output_dir
            / "walk_forward_summary.json",
            summary,
        )

        windows.to_csv(
            output_dir
            / "walk_forward_windows.csv",
            index=False,
        )

        audit.log(
            "walk_forward_completed",
            summary,
        )

        return {
            **state,
            "walk_forward": summary,
        }

    # ---------------------------------------------------------
    # FINAL VALIDATOR
    # ---------------------------------------------------------

    def validator(
        state: ResearchState,
    ) -> ResearchState:

        phase2 = state[
            "phase2"
        ]

        test = phase2[
            "test_returns"
        ]

        verdict = (
            determine_research_verdict(
                risk_review=(
                    state.get(
                        "risk_review",
                        {},
                    )
                ),
                backtest=(
                    phase2[
                        "backtest"
                    ]
                ),
                walk_forward=(
                    state.get(
                        "walk_forward"
                    )
                ),
                test_observations=len(
                    test
                ),
                minimum_test_observations=(
                    cfg.validation.min_test_observations
                ),
            )
        )

        audit.log(
            "validation_completed",
            verdict,
        )

        return {
            **state,
            "verdict": verdict,
            "validation_status": (
                verdict[
                    "validation_status"
                ]
            ),
            "performance_status": (
                verdict[
                    "performance_status"
                ]
            ),
            "final_status": (
                verdict[
                    "display_status"
                ]
            ),
        }

    # ---------------------------------------------------------
    # REPORTER
    # ---------------------------------------------------------

    def reporter(
        state: ResearchState,
    ) -> ResearchState:

        phase2 = state[
            "phase2"
        ]

        output_dir = Path(
            phase2["run_dir"]
        )

        report = (
            build_research_report(
                run_id=state[
                    "run_id"
                ],
                cfg=cfg,
                phase2=phase2,
                risk_review=state[
                    "risk_review"
                ],
                verdict=state[
                    "verdict"
                ],
                walk_forward=state.get(
                    "walk_forward"
                ),
                risk_free_metadata=(
                    state.get(
                        "risk_free_metadata"
                    )
                ),
                audit_summary=[
                    (
                        "Research request received "
                        "and deterministic plan created."
                    ),
                    (
                        "Initial Phase-2 portfolio "
                        "reviewed by risk critic."
                    ),
                    (
                        "Risk-policy violations "
                        "trigger constrained "
                        "reoptimization when required."
                    ),
                    (
                        "Portfolio weights frozen "
                        "before held-out evaluation."
                    ),
                    (
                        "Walk-forward validation "
                        "performed independently."
                    ),
                    (
                        "Final research verdict "
                        "generated without optimizing "
                        "against test performance."
                    ),
                ],
            )
        )

        path = save_research_report(
            output_dir
            / "FINAL_REPORT.md",
            report,
        )

        audit.log(
            "report_generated",
            {
                "path": str(path)
            },
        )

        return {
            **state,
            "report_path": str(path),
        }

    # ---------------------------------------------------------
    # BUILD LANGGRAPH
    # ---------------------------------------------------------

    graph = StateGraph(
        ResearchState
    )

    graph.add_node(
        "risk_free",
        risk_free_node,
    )

    graph.add_node(
        "planner",
        planner,
    )

    graph.add_node(
        "risk",
        risk_node,
    )

    graph.add_node(
        "reoptimize",
        reoptimize,
    )

    graph.add_node(
        "walk_forward",
        walk_forward_node,
    )

    graph.add_node(
        "validate",
        validator,
    )

    graph.add_node(
        "report",
        reporter,
    )

    graph.add_edge(
        START,
        "risk_free",
    )

    graph.add_edge(
        "risk_free",
        "planner",
    )

    graph.add_edge(
        "planner",
        "risk",
    )

    graph.add_conditional_edges(
        "risk",
        risk_router,
        {
            "reoptimize": "reoptimize",
            "walk_forward": "walk_forward",
        },
    )

    graph.add_edge(
        "reoptimize",
        "risk",
    )

    graph.add_edge(
        "walk_forward",
        "validate",
    )

    graph.add_edge(
        "validate",
        "report",
    )

    graph.add_edge(
        "report",
        END,
    )

    return graph.compile()