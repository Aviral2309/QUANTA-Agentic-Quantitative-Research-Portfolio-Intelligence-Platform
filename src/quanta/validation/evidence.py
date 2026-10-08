from __future__ import annotations

def evidence_verdict(metrics: dict, bootstrap: dict, integrity_status: str, walk_forward: dict|None=None) -> dict:
    if integrity_status=='REJECT': return {'signal':'INSUFFICIENT_EVIDENCE','score':0,'reasons':['Data-integrity gate failed.']}
    score=0; reasons=[]
    if metrics.get('excess_cagr',0)>0: score+=2; reasons.append('Outperformed benchmark on CAGR.')
    else: score-=2; reasons.append('Did not outperform benchmark on CAGR.')
    if metrics.get('information_ratio',0)>.25: score+=1
    elif metrics.get('information_ratio',0)<0: score-=1
    if metrics.get('sharpe_ratio',0)>.5: score+=1
    elif metrics.get('sharpe_ratio',0)<0: score-=1
    if metrics.get('max_drawdown',0)<-.25: score-=1
    if bootstrap.get('status')=='COMPLETED':
        if bootstrap.get('cagr_low',-1)>0: score+=2; reasons.append('Bootstrap CAGR interval remains positive.')
        elif bootstrap.get('cagr_high',1)<0: score-=2; reasons.append('Bootstrap CAGR interval remains negative.')
    if walk_forward and walk_forward.get('status')=='COMPLETED':
        if walk_forward.get('profitable_window_ratio',0)>=.60: score+=1
        if walk_forward.get('positive_sharpe_window_ratio',0)>=.60: score+=1
    signal='STRONG' if score>=5 else 'MODERATE' if score>=2 else 'WEAK' if score>=0 else 'AVOID'
    return {'signal':signal,'score':score,'reasons':reasons}
