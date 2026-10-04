from __future__ import annotations
from pathlib import Path
import pandas as pd
from quanta.core.ids import new_run_id
from quanta.data.universe import load_constituents,enrich_metadata,select_extremes
from quanta.data.validation import validate_prices
from quanta.quant.returns import calculate_returns, split_train_test
from quanta.quant.capm import estimate_capm
from quanta.quant.sml import plot_sml
from quanta.reporting.export import ensure_run_dir, write_json

def run_phase01(cfg, provider, artifacts_base="artifacts", run_id: str|None=None) -> dict:
    run_id=run_id or new_run_id(); out=ensure_run_dir(artifacts_base,run_id)
    ucfg=cfg.universe
    universe=load_constituents(ucfg.constituents_csv,ucfg.ticker_column,ucfg.name_column,ucfg.market_cap_column)
    if ucfg.auto_enrich_market_cap and universe.market_cap.isna().any(): universe=enrich_metadata(universe,provider,ucfg.metadata_workers)
    universe.to_csv(out/"universe_enriched.csv",index=False)
    selected=select_extremes(universe,ucfg.top_n,ucfg.bottom_n); selected.to_csv(out/"universe_100.csv",index=False)
    tickers=selected.ticker.tolist(); all_tickers=list(dict.fromkeys(tickers+[cfg.benchmark_ticker]))
    prices=provider.adjusted_close(all_tickers,cfg.start_date,cfg.end_date,cfg.interval)
    price_errors=validate_prices(prices[[c for c in prices.columns if c in all_tickers]],30)
    if cfg.benchmark_ticker not in prices.columns: raise RuntimeError(f"Benchmark {cfg.benchmark_ticker} missing from price response")
    returns=calculate_returns(prices); train,test=split_train_test(returns,cfg.validation_fraction)
    rows=[]; failures=[]
    for t in tickers:
        if t not in train.columns: failures.append({"ticker":t,"error":"missing price column"}); continue
        try:
            r=estimate_capm(t,train[t],train[cfg.benchmark_ticker],cfg.risk_free.annual_rate,cfg.annualization_factor,cfg.capm.market_return_method,cfg.capm.positive_alpha_threshold)
            rows.append(r.model_dump())
        except Exception as e: failures.append({"ticker":t,"error":str(e)})
    results=pd.DataFrame(rows).sort_values("capm_alpha_annual",ascending=False)
    results.to_csv(out/"capm_results.csv",index=False)
    results[results.classification=="positive_alpha"].to_csv(out/"positive_alpha.csv",index=False)
    train.to_pickle(out/"train_returns.pkl"); test.to_pickle(out/"test_returns.pkl"); prices.to_pickle(out/"prices.pkl")
    if len(results): plot_sml(results,cfg.risk_free.annual_rate,float(results.market_return_annual.median()),str(out/"sml.png"))
    write_json(out/"phase01_summary.json",{"run_id":run_id,"selected":len(selected),"capm_completed":len(results),"positive_alpha":int((results.classification=="positive_alpha").sum()) if len(results) else 0,"price_validation_warnings":price_errors,"failures":failures,"risk_free_note":cfg.risk_free.source_note})
    return {"run_id":run_id,"run_dir":str(out),"universe":selected,"metadata":universe,"prices":prices,"train_returns":train,"test_returns":test,"capm_results":results}
