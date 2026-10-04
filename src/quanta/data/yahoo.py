from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import pandas as pd
from .provider import MarketDataProvider
log = logging.getLogger(__name__)

class YahooFinanceProvider(MarketDataProvider):
    def adjusted_close(self, tickers: list[str], start: str, end: str, interval: str="1d") -> pd.DataFrame:
        import yfinance as yf
        raw = yf.download(tickers=tickers, start=start, end=end, interval=interval, auto_adjust=True, progress=False, group_by="column", threads=True)
        if raw.empty:
            raise RuntimeError("Yahoo Finance returned no price data")
        if isinstance(raw.columns, pd.MultiIndex):
            if "Close" not in raw.columns.get_level_values(0):
                raise RuntimeError("Close field missing from Yahoo response")
            close = raw["Close"].copy()
        else:
            if "Close" not in raw.columns:
                raise RuntimeError("Close field missing from Yahoo response")
            close = raw[["Close"]].copy()
            close.columns = [tickers[0]]
        close.index = pd.to_datetime(close.index)
        close = close.sort_index()
        return close

    @staticmethod
    def _one_metadata(ticker: str) -> dict:
        import yfinance as yf
        t = yf.Ticker(ticker)
        info = {}
        try:
            info = t.info or {}
        except Exception:
            info = {}
        market_cap = info.get("marketCap")
        if market_cap is None:
            try:
                market_cap = t.fast_info.get("market_cap")
            except Exception:
                pass
        return {
            "ticker": ticker,
            "company_name": info.get("longName") or info.get("shortName") or ticker,
            "market_cap": market_cap,
            "pe_ratio": info.get("trailingPE"),
            "pb_ratio": info.get("priceToBook"),
            "ebit": info.get("ebitda") or info.get("operatingCashflow"),
            "avg_volume": info.get("averageVolume") or info.get("averageVolume10days"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
        }

    def company_metadata(self, tickers: list[str], workers: int=8) -> pd.DataFrame:
        rows = []
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = {ex.submit(self._one_metadata, t): t for t in tickers}
            for f in as_completed(futures):
                t = futures[f]
                try:
                    rows.append(f.result())
                except Exception as exc:
                    log.warning("Metadata failed for %s: %s", t, exc)
                    rows.append({"ticker": t, "company_name": t, "market_cap": None})
        return pd.DataFrame(rows)
