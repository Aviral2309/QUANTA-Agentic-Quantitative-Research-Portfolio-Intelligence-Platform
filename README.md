# QUANTA — Agentic Quantitative Research & Portfolio Intelligence Platform

**Repository scope: complete implementation through Phase 03.**

QUANTA is a research-first quantitative platform that turns a dated NIFTY 500 universe into CAPM-relative signals, constructs a constrained diversified portfolio, validates it out-of-sample, and then runs an auditable LangGraph research-review harness.

> **Core rule:** Agents reason. Tools calculate. The harness controls. Validators verify. Humans approve.

## What is implemented

### Phase 01 — Financial data + CAPM research engine
- NIFTY 500 seed/snapshot loader.
- Automatic Yahoo metadata + market-cap enrichment when caps are absent.
- Top-50 + bottom-50 market-cap sampling.
- Adjusted price ingestion and pickle cache (no PyArrow dependency).
- Data checks, returns, chronological train/test split.
- OLS beta/excess-return regression.
- CAPM required return and separate CAPM screening alpha.
- Positive-alpha candidate export and SML visualization.

### Phase 02 — Portfolio construction + validation
- Solver #1: constrained long-only max-Sharpe optimization.
- Multi-factor scoring: P/E, P/B, EBIT proxy, market cap, volatility, sentiment, volume.
- Top-5 core selection.
- Low-correlation diversifier selection.
- Solver #2 over final core + diversifier set with minimum core weights.
- Held-out static backtest, transaction-cost haircut, benchmark comparison.
- Optional MKT-RF/SMB/HML/MOM factor regression.

### Phase 03 — Agentic AI + harness engineering
- LangGraph state machine.
- Planner, deterministic Risk/Critic, re-optimization loop, Validator, Report node.
- Optional LLM planner commentary; calculations remain deterministic.
- Bounded retries/re-optimization.
- Audit JSONL and final research report.
- Explicit deterministic tool registry / permission surface.

## Repository map

```text
QUANTA_Phase_01_03/
├── config/research.yaml
├── data/
│   ├── raw/nifty500_constituents.csv
│   ├── raw/news_sentiment.example.csv
│   ├── raw/india_factors.example.csv
│   ├── cache/
│   └── processed/
├── docs/
│   ├── SYSTEM_DESIGN.md
│   ├── METHODOLOGY.md
│   └── RUNBOOK.md
├── src/quanta/
│   ├── core/
│   ├── data/
│   ├── domain/
│   ├── quant/
│   ├── portfolio/
│   ├── validation/
│   ├── pipeline/
│   ├── harness/
│   ├── reporting/
│   ├── tools/
│   └── cli.py
├── tests/
├── scripts/
├── pyproject.toml
└── README.md
```

## Installation — use Python 3.12

Python 3.14 caused the earlier PyArrow build problem. This repo deliberately removes the PyArrow requirement and formally supports Python 3.11–3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
quanta doctor
pytest
```

## Run the project

```powershell
quanta phase1 --config config/research.yaml
quanta phase2 --config config/research.yaml
quanta phase3 --config config/research.yaml
```

Phase 2 and Phase 3 intentionally execute earlier phases first so each run has one internally consistent data snapshot and run ID.

## Optional LLM mode

Phase 3 works without an API key. To add LLM planner commentary:

```powershell
pip install -e ".[dev,llm]"
```

Set `OPENAI_API_KEY` in your environment and change:

```yaml
agentic:
  use_llm: true
```

The LLM does **not** calculate beta, CAPM, covariance, weights, Sharpe, risk metrics, or backtest results.

## Required input data

`data/raw/nifty500_constituents.csv` already contains the supplied 500-symbol seed list. Its missing market caps are automatically enriched by the Yahoo prototype provider on the first live run.

For rigorous historical research, replace it with a **dated point-in-time constituent snapshot**. Using the current constituent list to simulate an old investment decision can introduce survivorship/look-ahead bias.

Optional files:
- Copy `news_sentiment.example.csv` to `news_sentiment.csv` and populate a reproducible sentiment score per ticker.
- Copy `india_factors.example.csv` to `india_factors.csv` with dated daily factor returns. If absent, factor validation is reported as `SKIPPED` rather than faked.

## Outputs

Every run gets:

```text
artifacts/QNT-.../
├── universe_enriched.csv
├── universe_100.csv
├── prices.pkl
├── train_returns.pkl
├── test_returns.pkl
├── capm_results.csv
├── positive_alpha.csv
├── sml.png
├── phase01_summary.json
├── solver1.json
├── factor_scores.csv
├── diversifiers.csv
├── solver2.json
├── backtest.json
├── backtest_curve.csv
├── factor_regression.json
├── audit.jsonl
└── FINAL_REPORT.md
```

## Research limitations

- Top/bottom-50 sampling is intentionally bimodal and non-representative of the middle 400 stocks.
- CAPM screening alpha is not synonymous with intrinsic undervaluation.
- Free Yahoo data is suitable for prototyping, not for authoritative production or regulated research.
- The default 6.5% risk-free rate is a runnable placeholder; formal analysis should use a dated Indian T-bill/G-Sec series aligned to the return frequency.
- Fundamentals from free endpoints can be missing or point-in-time inconsistent. Formal backtests require point-in-time fundamental data.
- Static held-out backtesting is implemented; walk-forward/rebalancing simulation is a natural Phase-4/5 extension.
- No brokerage execution exists. QUANTA is research-only.

## Tests

Tests cover returns, train/test chronology, CAPM beta recovery, universe selection, factor directionality, optimization constraints and risk policy behavior.

```powershell
pytest
```

## Documentation

Start with `docs/SYSTEM_DESIGN.md`, then `docs/METHODOLOGY.md`, then `docs/RUNBOOK.md`.

## Disclaimer

Educational/research software only. Nothing produced by QUANTA is investment advice.
