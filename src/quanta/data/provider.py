from __future__ import annotations
from abc import ABC, abstractmethod
from datetime import date
import pandas as pd

class MarketDataProvider(ABC):
    @abstractmethod
    def adjusted_close(self, tickers: list[str], start: str, end: str, interval: str="1d") -> pd.DataFrame: ...
    @abstractmethod
    def company_metadata(self, tickers: list[str], workers: int=8) -> pd.DataFrame: ...
