from __future__ import annotations

from pathlib import Path

from quanta.harness.audit import (
    AuditLogger,
)
from quanta.harness.graph import (
    build_research_graph,
)


def run_phase03(
    cfg,
    phase2: dict,
    request: str = (
        "Build and validate a moderate-risk "
        "quantitative research portfolio."
    ),
) -> dict:
    """
    Execute QUANTA Phase 3.1 → 3.4.

    Phase 3.1
        Risk critic + constrained reoptimization.

    Phase 3.2
        Professional research report.

    Phase 3.3
        External risk-free / sentiment /
        factor-data support.

    Phase 3.4
        Walk-forward validation.
    """

    run_dir = Path(
        phase2["run_dir"]
    )

    run_id = (
        phase2.get("run_id")
        or run_dir.name
    )

    audit = AuditLogger(
        run_dir / "audit.jsonl"
    )

    audit.log(
        "phase03_started",
        {
            "run_id": run_id,
            "request": request,
        },
    )

    graph = build_research_graph(
        cfg,
        audit,
    )

    initial_state = {
        "run_id": run_id,
        "request": request,
        "phase2": phase2,
        "reoptimize_count": 0,
    }

    final_state = graph.invoke(
        initial_state
    )

    audit.log(
        "phase03_completed",
        {
            "final_status": (
                final_state.get(
                    "final_status"
                )
            ),
            "report_path": (
                final_state.get(
                    "report_path"
                )
            ),
        },
    )

    return {
        **phase2,

        "risk_free_rate": (
            final_state.get(
                "risk_free_rate"
            )
        ),

        "risk_free_metadata": (
            final_state.get(
                "risk_free_metadata"
            )
        ),

        "risk_review": (
            final_state.get(
                "risk_review"
            )
        ),

        "walk_forward": (
            final_state.get(
                "walk_forward"
            )
        ),

        "verdict": (
            final_state.get(
                "verdict"
            )
        ),

        "final_status": (
            final_state.get(
                "final_status"
            )
        ),

        "report_path": (
            final_state.get(
                "report_path"
            )
        ),

        "reoptimize_count": (
            final_state.get(
                "reoptimize_count",
                0,
            )
        ),

        # Important: return the potentially
        # reoptimized Phase-2 state.
        "phase2_final": (
            final_state.get(
                "phase2"
            )
        ),
    }