from __future__ import annotations
from pathlib import Path
from langgraph.graph import StateGraph, START, END
from quanta.harness.state import ResearchState
from quanta.harness.audit import AuditLogger
from quanta.harness.policies import deterministic_risk_review
from quanta.portfolio.optimizer import optimize_max_sharpe
from quanta.validation.backtest import backtest_static
from quanta.reporting.report import render_markdown


def _optional_llm(cfg):
    if not cfg.agentic.use_llm: return None
    try:
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=cfg.agentic.model, temperature=0)
    except Exception:
        return None


def build_graph(cfg, run_dir: str):
    audit=AuditLogger(Path(run_dir)/"audit.jsonl"); llm=_optional_llm(cfg)

    def planner(state: ResearchState):
        plan={"objective":"review and validate the Phase-2 portfolio","stages":["risk_review","validation","report"],"research_only":True}
        notes={}
        if llm:
            try:
                msg=llm.invoke("You are QUANTA Research Planner. Return one short paragraph explaining how to independently review a quantitative portfolio. Never invent numerical results.")
                notes["planner"]=getattr(msg,"content",str(msg))
            except Exception as e: notes["planner_error"]=str(e)
        audit.log("planner",plan); return {"plan":plan,"llm_notes":notes,"reoptimize_count":state.get("reoptimize_count",0)}

    def risk_agent(state: ResearchState):
        review=deterministic_risk_review(state["phase2"],cfg.agentic.risk_limits)
        audit.log("risk_review",review.model_dump())
        return {"risk_review":review.model_dump()}

    def route_after_risk(state: ResearchState):
        review=state["risk_review"]
        if review["status"]=="REJECT" and state.get("reoptimize_count",0)<cfg.agentic.max_reoptimization_loops: return "reoptimize"
        return "validate"

    def reoptimize(state: ResearchState):
        p=state["phase2"]; count=state.get("reoptimize_count",0)+1
        assets=p["final_assets"]
        # Tighten the max single-name weight after a rejection, but never below feasibility.
        target=min(cfg.portfolio.max_weight, cfg.agentic.risk_limits.get("max_single_weight",cfg.portfolio.max_weight))*0.95
        maxw=max(target,1/len(assets)+1e-6)
        mins={t:min(cfg.portfolio.minimum_core_weight,maxw) for t in p["core"]}
        opt=optimize_max_sharpe(p["train_returns"][assets],cfg.risk_free.annual_rate,cfg.annualization_factor,maxw,mins,cfg.portfolio.optimizer_restarts)
        p["solver2"]=opt
        bench=p["test_returns"][cfg.benchmark_ticker] if cfg.benchmark_ticker in p["test_returns"].columns else None
        bt,curve=backtest_static(opt.weights,p["test_returns"],cfg.risk_free.annual_rate,cfg.annualization_factor,bench,cfg.portfolio.transaction_cost_bps)
        p["backtest"]=bt; p["backtest_curve"]=curve
        audit.log("reoptimize",{"count":count,"sharpe":opt.sharpe_ratio,"max_weight":maxw})
        return {"phase2":p,"reoptimize_count":count}

    def validator(state: ResearchState):
        p=state["phase2"]; bt=p["backtest"]
        status="VALIDATED" if bt.observations>=cfg.validation.min_test_observations and state["risk_review"]["status"]=="PASS" else "PARTIALLY_VALIDATED"
        audit.log("validation",{"status":status,"backtest":bt.model_dump()})
        return {"validation_status":status}

    def reporter(state: ResearchState):
        p=state["phase2"]; path=Path(run_dir)/"FINAL_REPORT.md"
        summary={"run_id":state["run_id"],"final_status":state["validation_status"],"weights":p["solver2"].weights,"backtest":p["backtest"].model_dump(),"risk_findings":state["risk_review"]["findings"]}
        render_markdown(summary,path); audit.log("report",{"path":str(path)})
        return {"report_path":str(path),"final_status":state["validation_status"]}

    g=StateGraph(ResearchState)
    g.add_node("planner",planner); g.add_node("risk",risk_agent); g.add_node("reoptimize",reoptimize); g.add_node("validate",validator); g.add_node("report",reporter)
    g.add_edge(START,"planner"); g.add_edge("planner","risk")
    g.add_conditional_edges("risk",route_after_risk,{"reoptimize":"reoptimize","validate":"validate"})
    g.add_edge("reoptimize","risk"); g.add_edge("validate","report"); g.add_edge("report",END)
    return g.compile()
