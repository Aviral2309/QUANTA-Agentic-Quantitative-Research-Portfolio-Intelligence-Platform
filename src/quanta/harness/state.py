from __future__ import annotations
from typing import TypedDict, Any
class ResearchState(TypedDict, total=False):
    run_id: str
    request: str
    phase2: dict[str,Any]
    plan: dict[str,Any]
    risk_review: dict[str,Any]
    reoptimize_count: int
    validation_status: str
    report_path: str
    final_status: str
    llm_notes: dict[str,str]
