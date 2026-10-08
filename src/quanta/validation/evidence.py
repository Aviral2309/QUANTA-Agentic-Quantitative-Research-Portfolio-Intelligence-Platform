from __future__ import annotations
import math

def evidence_verdict(metrics,bootstrap,integrity_status,walk_forward=None):
    reasons=[]
    if integrity_status != "PASS":
        return {"signal":"INSUFFICIENT_EVIDENCE","score":0,
                "reasons":["Historical data integrity is not fully verified; research-only performance, not investment advice."],
                "observed_performance":"UNDERPERFORMED" if metrics.get("excess_cagr",0)<0 else "UNVERIFIED"}
    required=("cagr","sharpe_ratio","max_drawdown","excess_cagr","information_ratio")
    if any(not isinstance(metrics.get(k),(int,float)) or not math.isfinite(metrics[k]) for k in required):
        return {"signal":"INSUFFICIENT_EVIDENCE","score":0,"reasons":["Missing or invalid benchmark-relative metrics."]}
    score=0
    for condition,points,reason in [
        (metrics["excess_cagr"]>0,2,"Outperformed benchmark on annualized growth."),
        (metrics["information_ratio"]>0.25,1,"Positive benchmark-relative information ratio."),
        (metrics["sharpe_ratio"]>0.5,1,"Positive risk-adjusted Sharpe evidence."),
        (metrics["max_drawdown"]>=-0.25,1,"Drawdown within research threshold.")]:
        score+=points if condition else -points
        reasons.append(reason if condition else "Not met: "+reason)
    if bootstrap.get("status")=="COMPLETED":
        if bootstrap.get("cagr_low",float("-inf"))>0:score+=2;reasons.append("Bootstrap CAGR lower bound positive.")
        elif bootstrap.get("cagr_high",float("inf"))<0:score-=2;reasons.append("Bootstrap CAGR upper bound negative.")
        else:reasons.append("Bootstrap CAGR interval crosses zero.")
    if walk_forward and walk_forward.get("status")=="COMPLETED":
        if walk_forward.get("profitable_window_ratio",0)<.6:score-=1;reasons.append("Walk-forward consistency below 60%.")
    signal="STRONG" if score>=5 else "MODERATE" if score>=2 else "WEAK" if score>=0 else "AVOID"
    return {"signal":signal,"score":score,"reasons":reasons}
