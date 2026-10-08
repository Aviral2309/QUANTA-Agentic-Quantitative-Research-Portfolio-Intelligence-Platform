from __future__ import annotations
from pathlib import Path
import json
import pandas as pd
from quanta.portfolio.robust_optimizer import optimize_robust
from quanta.validation.advanced_metrics import cagr, sharpe, max_drawdown, annualized_volatility, information_ratio, tracking_error, sortino
from quanta.validation.bootstrap import block_bootstrap_ci
from quanta.validation.integrity import validate_research_inputs
from quanta.validation.walk_forward35 import walk_forward_robust
from quanta.validation.evidence import evidence_verdict


def _dump(path,obj):
    def conv(x):
        if hasattr(x,'model_dump'): return x.model_dump()
        if hasattr(x,'__dict__'): return x.__dict__
        if isinstance(x,(pd.Timestamp,)): return str(x)
        raise TypeError(type(x).__name__)
    Path(path).write_text(json.dumps(obj,indent=2,default=conv),encoding='utf-8')


def run_phase35(cfg, phase2: dict) -> dict:
    out=Path(phase2['run_dir']); train=phase2['train_returns']; test=phase2['test_returns']; prices=phase2['prices']
    assets=[a for a in phase2['final_assets'] if a in train.columns and a in test.columns]
    rf_meta=phase2.get('risk_free_metadata',{'rate':cfg.risk_free.annual_rate,'source':'static_config','observation_date':None})
    p35=getattr(cfg,'phase35',None)
    strict=bool(getattr(p35,'strict_point_in_time',False)) if p35 else False
    integrity=validate_research_inputs(prices,train,test,cfg.benchmark_ticker,rf_meta,strict,
        getattr(p35,'pit_constituents_csv',None) if p35 else None,getattr(p35,'pit_fundamentals_csv',None) if p35 else None)
    if integrity.status=='REJECT':
        _dump(out/'phase35_integrity.json',integrity); raise RuntimeError('Phase 3.5 data-integrity gate REJECTED this run. See phase35_integrity.json')
    limits=getattr(cfg,'risk_limits',None)
    if limits is not None:
        top3=float(getattr(limits,'max_top3_concentration',.50))
    else:
        raw=getattr(getattr(cfg,'agentic',None),'risk_limits',{}) or {}
        top3=float(raw.get('max_top3_concentration',raw.get('max_top3_weight',.50)))
    # Train model candidates ONLY on inner training; validate on a later slice.
    cut=max(60,int(len(train)*.80)); inner_tr=train[assets].iloc[:cut]; inner_val=train[assets].iloc[cut:]
    candidates=optimize_robust(inner_tr,cfg.risk_free.annual_rate,cfg.annualization_factor,
        cfg.portfolio.max_weight,top3,
        getattr(p35,'return_shrinkage',.60) if p35 else .60,getattr(p35,'covariance_shrinkage',.25) if p35 else .25,
        getattr(p35,'l2_penalty',.02) if p35 else .02,0.0,cfg.portfolio.optimizer_restarts)
    # choose candidate using an inner chronological validation slice from TRAIN only
    rescored=[]
    for c in candidates:
        w=pd.Series(c.weights).reindex(inner_val.columns).fillna(0); r=inner_val.dot(w)
        rescored.append((sharpe(r,cfg.risk_free.annual_rate,cfg.annualization_factor),c))
    chosen_name=max(rescored,key=lambda z:z[0] if pd.notna(z[0]) else float('-inf'))[1].name
    # Refit the selected MODEL FAMILY using all training observations; never touch test.
    refitted=optimize_robust(train[assets],cfg.risk_free.annual_rate,cfg.annualization_factor,
        cfg.portfolio.max_weight,top3,
        getattr(p35,'return_shrinkage',.60) if p35 else .60,getattr(p35,'covariance_shrinkage',.25) if p35 else .25,
        getattr(p35,'l2_penalty',.02) if p35 else .02,0.0,cfg.portfolio.optimizer_restarts)
    chosen=next(c for c in refitted if c.name==chosen_name)
    w=pd.Series(chosen.weights).reindex(test.columns).fillna(0); pr=test.dot(w); turnover=float(w.abs().sum()); cost=turnover*cfg.portfolio.transaction_cost_bps/10000
    if len(pr): pr.iloc[0]-=cost
    br=test[cfg.benchmark_ticker].reindex(pr.index).dropna() if cfg.benchmark_ticker in test else pd.Series(dtype=float)
    common=pr.index.intersection(br.index); rr=pr.loc[common]; bb=br.loc[common]
    metrics={'observations':len(pr),'cagr':cagr(pr,cfg.annualization_factor),'annualized_volatility':annualized_volatility(pr,cfg.annualization_factor),
        'sharpe_ratio':sharpe(pr,cfg.risk_free.annual_rate,cfg.annualization_factor),'sortino_ratio':sortino(pr,cfg.risk_free.annual_rate,cfg.annualization_factor),
        'max_drawdown':max_drawdown(pr),'turnover':turnover,'transaction_cost':cost}
    if len(common):
        metrics.update({'benchmark_cagr':cagr(bb,cfg.annualization_factor),'excess_cagr':metrics['cagr']-cagr(bb,cfg.annualization_factor),
                        'tracking_error':tracking_error(rr,bb,cfg.annualization_factor),'information_ratio':information_ratio(rr,bb,cfg.annualization_factor)})
    boot=block_bootstrap_ci(pr,cfg.risk_free.annual_rate,cfg.annualization_factor,getattr(p35,'bootstrap_samples',1000) if p35 else 1000)
    combined=pd.concat([train,test]).sort_index()
    wf=walk_forward_robust(combined,assets,cfg.risk_free.annual_rate,cfg.annualization_factor,
        getattr(getattr(cfg.validation,'walk_forward',None),'train_days',504),getattr(getattr(cfg.validation,'walk_forward',None),'test_days',63),
        getattr(getattr(cfg.validation,'walk_forward',None),'step_days',63),cfg.portfolio.max_weight,top3,cfg.portfolio.transaction_cost_bps)
    evidence=evidence_verdict(metrics,boot,integrity.status,wf)
    # Fixed final assets were selected upstream, possibly with future information.
    # Do not advertise this run as historically investable or PIT-validated.
    evidence={'signal':'INSUFFICIENT_EVIDENCE','score':0,
        'observed_performance':'UNDERPERFORMED' if metrics.get('excess_cagr',0)<0 else 'UNVERIFIED',
        'reasons':['Final asset selection is not independently reconstructed as-of each historical date.',
                   *evidence.get('reasons',[])]}
    result={'status':'COMPLETED','integrity':integrity.model_dump(),'research_readiness':'NOT_VERIFIED','walk_forward_scope':'FIXED_ASSET_UNIVERSE_ONLY','selected_model':chosen.name,'weights':chosen.weights,'metrics':metrics,'bootstrap':boot,
            'walk_forward':{k:v for k,v in wf.items() if k!='returns'},'evidence':evidence}
    _dump(out/'phase35_integrity.json',integrity); _dump(out/'phase35_result.json',result)
    pd.DataFrame({'portfolio_return':pr}).to_csv(out/'phase35_test_returns.csv')
    if 'returns' in wf: pd.DataFrame({'walk_forward_return':wf['returns']}).to_csv(out/'phase35_walk_forward_returns.csv')
    return result
