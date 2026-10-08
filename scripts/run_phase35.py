from __future__ import annotations
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from quanta.core.config import load_config
from quanta.data.yahoo import YahooFinanceProvider
from quanta.data.cache import CachedProvider
from quanta.pipeline.phase01 import run_phase01
from quanta.pipeline.phase02 import run_phase02
from quanta.pipeline.phase35 import run_phase35


def main():
    cfg=load_config('config/research.yaml')
    provider=CachedProvider(YahooFinanceProvider())
    p1=run_phase01(cfg,provider)
    p2=run_phase02(cfg,p1)
    result=run_phase35(cfg,p2)
    console=Console()
    table=Table(title='QUANTA Phase 3.5 Robust Portfolio')
    table.add_column('Ticker'); table.add_column('Weight',justify='right')
    for ticker,weight in sorted(result['weights'].items(),key=lambda kv:kv[1],reverse=True):
        if weight>1e-6: table.add_row(ticker,f'{weight:.2%}')
    console.print(table)
    console.print(f"Integrity: {result['integrity']['status']}")
    console.print(f"Model: {result['selected_model']}")
    console.print(f"Evidence signal: {result['evidence']['signal']} (score={result['evidence']['score']})")
    console.print(f"CAGR: {result['metrics']['cagr']:.2%}")
    console.print(f"Sharpe: {result['metrics']['sharpe_ratio']:.3f}")
    console.print(f"Max drawdown: {result['metrics']['max_drawdown']:.2%}")
    if 'excess_cagr' in result['metrics']: console.print(f"Excess CAGR vs benchmark: {result['metrics']['excess_cagr']:.2%}")
    console.print(f"Artifacts: {p2['run_dir']}")

if __name__=='__main__': main()
