from __future__ import annotations
from pathlib import Path

def render_markdown(summary: dict, path: str|Path) -> None:
    lines=["# QUANTA Research Report","",f"**Run ID:** `{summary.get('run_id')}`",f"**Final status:** **{summary.get('final_status','UNKNOWN')}**","","## Portfolio"]
    for t,w in sorted(summary.get("weights",{}).items(),key=lambda kv:kv[1],reverse=True): lines.append(f"- {t}: {w:.2%}")
    bt=summary.get("backtest",{}) or {}; lines += ["","## Validation",f"- CAGR: {bt.get('cagr','n/a')}",f"- Sharpe: {bt.get('sharpe_ratio','n/a')}",f"- Max drawdown: {bt.get('max_drawdown','n/a')}","","## Risk findings"]
    for f in summary.get("risk_findings",[]): lines.append(f"- **{f.get('severity')} / {f.get('code')}** — {f.get('message')}")
    lines += ["","## Research disclaimer","This system is for educational and research use. It does not constitute investment advice."]
    Path(path).write_text("\n".join(lines),encoding="utf-8")
