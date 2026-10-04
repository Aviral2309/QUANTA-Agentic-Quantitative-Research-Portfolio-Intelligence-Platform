from __future__ import annotations
from pathlib import Path
import pandas as pd
from .provider import MarketDataProvider

def normalize_nse_ticker(value: str) -> str:
    value=str(value).strip().upper()
    return value if value.endswith(".NS") or value.startswith("^") else f"{value}.NS"

def load_constituents(path: str, ticker_col="ticker", name_col="company_name", market_cap_col="market_cap") -> pd.DataFrame:
    df=pd.read_csv(path)
    if ticker_col not in df.columns: raise ValueError(f"Missing required column: {ticker_col}")
    out=pd.DataFrame({"ticker":df[ticker_col].map(normalize_nse_ticker)})
    out["company_name"] = df[name_col] if name_col in df.columns else out["ticker"]
    out["market_cap"] = pd.to_numeric(df[market_cap_col], errors="coerce") if market_cap_col in df.columns else float("nan")
    out=out.drop_duplicates("ticker").reset_index(drop=True)
    return out

def enrich_metadata(universe: pd.DataFrame, provider: MarketDataProvider, workers: int=8) -> pd.DataFrame:
    meta=provider.company_metadata(universe["ticker"].tolist(), workers=workers)
    base=universe.drop(columns=[c for c in universe.columns if c != "ticker" and c in meta.columns], errors="ignore")
    merged=base.merge(meta, on="ticker", how="left")
    return merged

def select_extremes(df: pd.DataFrame, top_n: int=50, bottom_n: int=50) -> pd.DataFrame:
    x=df.copy(); x["market_cap"]=pd.to_numeric(x["market_cap"], errors="coerce")
    valid=x.dropna(subset=["market_cap"]); valid=valid[valid.market_cap>0]
    if len(valid) < top_n + bottom_n:
        raise ValueError(f"Need at least {top_n+bottom_n} valid market caps; found {len(valid)}")
    ranked=valid.sort_values("market_cap", ascending=False).reset_index(drop=True)
    top=ranked.head(top_n).copy(); top["universe_bucket"]="top"
    bottom=ranked.tail(bottom_n).copy(); bottom["universe_bucket"]="bottom"
    return pd.concat([top,bottom], ignore_index=True).drop_duplicates("ticker")
