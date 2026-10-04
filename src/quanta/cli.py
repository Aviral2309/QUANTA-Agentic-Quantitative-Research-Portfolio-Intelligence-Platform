from __future__ import annotations
import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from quanta.core.config import load_config
from quanta.core.logging import configure_logging
from quanta.data.yahoo import YahooFinanceProvider
from quanta.data.cache import CachedProvider
from quanta.pipeline.phase01 import run_phase01
from quanta.pipeline.phase02 import run_phase02
from quanta.pipeline.phase03 import run_phase03

app=typer.Typer(no_args_is_help=True,help="QUANTA quantitative research platform")
console=Console()

def provider(): return CachedProvider(YahooFinanceProvider())

def _summary(p2):
    t=Table(title="QUANTA Final Portfolio"); t.add_column("Ticker"); t.add_column("Weight",justify="right")
    for k,v in sorted(p2["solver2"].weights.items(),key=lambda kv:kv[1],reverse=True): t.add_row(k,f"{v:.2%}")
    console.print(t)

@app.command()
def doctor():
    """Check runtime and project prerequisites."""
    console.print(f"Python: {sys.version.split()[0]}")
    if sys.version_info[:2] not in [(3,11),(3,12)]: console.print("[red]Use Python 3.11 or 3.12.[/red]")
    else: console.print("[green]Python version OK[/green]")
    for p in ["config/research.yaml","data/raw/nifty500_constituents.csv"]: console.print(f"{p}: {'OK' if Path(p).exists() else 'MISSING'}")

@app.command()
def phase1(config: str="config/research.yaml"):
    configure_logging(); cfg=load_config(config); r=run_phase01(cfg,provider()); console.print(f"[green]Phase 1 complete[/green] -> {r['run_dir']}")

@app.command()
def phase2(config: str="config/research.yaml"):
    configure_logging(); cfg=load_config(config); p1=run_phase01(cfg,provider()); p2=run_phase02(cfg,p1); _summary(p2); console.print(f"[green]Phase 2 complete[/green] -> {p2['run_dir']}")

@app.command()
def phase3(config: str="config/research.yaml", request: str="Build and validate a moderate-risk research portfolio"):
    configure_logging(); cfg=load_config(config); p1=run_phase01(cfg,provider()); p2=run_phase02(cfg,p1); result=run_phase03(cfg,p2,request); _summary(result["phase2"]); console.print(f"[green]Phase 3 complete: {result.get('final_status')}[/green]"); console.print(f"Report: {result.get('report_path')}")

if __name__=="__main__": app()
