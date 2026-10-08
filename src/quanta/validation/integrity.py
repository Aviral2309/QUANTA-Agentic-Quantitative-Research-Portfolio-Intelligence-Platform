from __future__ import annotations
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd

@dataclass
class IntegrityReport:
    status: str
    checks: dict
    warnings: list[str]
    def model_dump(self): return asdict(self)


def validate_research_inputs(prices: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                             benchmark: str, rf_metadata: dict|None=None, strict_point_in_time: bool=False,
                             pit_constituents_csv: str|None=None, pit_fundamentals_csv: str|None=None) -> IntegrityReport:
    checks={}; warnings=[]
    checks['chronological_split']=bool(len(train) and len(test) and train.index.max()<test.index.min())
    checks['benchmark_present']=benchmark in prices.columns or benchmark in test.columns
    checks['price_duplicates']=not prices.index.duplicated().any()
    checks['minimum_test_rows']=len(test)>=40
    checks['risk_free_dated']=bool(rf_metadata and rf_metadata.get('observation_date'))
    checks['pit_constituents']=bool(pit_constituents_csv and Path(pit_constituents_csv).exists())
    checks['pit_fundamentals']=bool(pit_fundamentals_csv and Path(pit_fundamentals_csv).exists())
    if not checks['risk_free_dated']: warnings.append('Risk-free rate is not backed by a dated observation; static fallback is research-only.')
    if not checks['pit_constituents']: warnings.append('Historical point-in-time constituent membership unavailable; survivorship bias remains.')
    if not checks['pit_fundamentals']: warnings.append('Point-in-time fundamentals unavailable; do not claim leakage-free historical factor selection.')
    hard=['chronological_split','benchmark_present','price_duplicates','minimum_test_rows']
    if strict_point_in_time: hard += ['risk_free_dated','pit_constituents','pit_fundamentals']
    status='PASS' if all(checks[k] for k in hard) else 'REJECT'
    return IntegrityReport(status,checks,warnings)
