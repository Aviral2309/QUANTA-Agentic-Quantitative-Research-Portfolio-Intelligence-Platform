from __future__ import annotations

from typing import Any, TypedDict


class ResearchState(
    TypedDict,
    total=False,
):
    run_id: str

    request: str

    phase2: dict[str, Any]

    plan: dict[str, Any]

    risk_free_rate: float
    risk_free_metadata: dict[str, Any]

    risk_review: dict[str, Any]

    reoptimize_count: int

    walk_forward: dict[str, Any]

    validation_status: str
    performance_status: str

    verdict: dict[str, Any]

    report_path: str

    final_status: str

    llm_notes: str