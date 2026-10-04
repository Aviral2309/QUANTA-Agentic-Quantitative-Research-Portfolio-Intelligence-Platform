from __future__ import annotations
from pathlib import Path
import hashlib, json
import pandas as pd
from .provider import MarketDataProvider

class CachedProvider(MarketDataProvider):
    def __init__(self, inner: MarketDataProvider, cache_dir: str="data/cache"):
        self.inner = inner
        self.cache = Path(cache_dir); self.cache.mkdir(parents=True, exist_ok=True)
    def _key(self, prefix: str, payload: dict) -> Path:
        h=hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]
        return self.cache/f"{prefix}_{h}.pkl"
    def adjusted_close(self, tickers, start, end, interval="1d"):
        p=self._key("prices", {"tickers":sorted(tickers),"start":start,"end":end,"interval":interval})
        if p.exists(): return pd.read_pickle(p)
        df=self.inner.adjusted_close(tickers,start,end,interval); df.to_pickle(p); return df
    def company_metadata(self, tickers, workers=8):
        p=self._key("metadata", {"tickers":sorted(tickers)})
        if p.exists(): return pd.read_pickle(p)
        df=self.inner.company_metadata(tickers,workers); df.to_pickle(p); return df
