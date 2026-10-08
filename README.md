# QUANTA
## Agentic Quantitative Research & Portfolio Intelligence Platform

> **Status (8 October 2026):** Phases 1–3.4 have completed an end-to-end user run. Phase 3.5 is an available **experimental overlay**, not yet independently verified end-to-end on the user's environment. Phase 4 (web application) has not started.

**Research principle:** *Agents reason. Tools calculate. The harness controls. Validators verify. Humans approve.*

QUANTA is a Python-based, research-first quantitative investment analysis system. It screens a deliberately selected subset of NIFTY 500 companies using CAPM-relative signals, builds constrained portfolios, checks risk policies, and evaluates results on chronological historical data. The project is **not** an automated trading bot, investment adviser, or a system that can guarantee future returns.

---

## 1. Objectives

- Automate a reproducible pipeline from data ingestion through research report generation.
- Keep quantitative calculations in deterministic, testable Python functions rather than an LLM.
- Separate **risk-policy compliance**, **research validation**, and **investment performance**.
- Prevent held-out performance from being used to tune the portfolio retrospectively.
- Disclose data provenance, look-ahead risk, survivorship bias, uncertainty, and model limitations.

## 2. Current development status

| Stage | Scope | Status |
|---|---|---|
| Phase 1 | Universe, adjusted prices, returns, CAPM/SML | Implemented |
| Phase 2 | Factor scoring, core/diversifier selection, portfolio optimization | Implemented |
| Phase 3.1 | Risk review and bounded risk-constrained reoptimization | Implemented |
| Phase 3.2 | Research reporting and verdicts | Implemented |
| Phase 3.3 | External risk-free/factor/sentiment input interfaces | Implemented; actual data coverage incomplete |
| Phase 3.4 | Walk-forward validation | Implemented; selected-asset-set approximation |
| Phase 3.5 | Robust estimation, portfolio alternatives, integrity gate, cost-aware evaluation, uncertainty | Experimental overlay; integration/validation pending |
| Phase 4 | FastAPI, PostgreSQL, React dashboard | Planned |
| Phase 5 | Deployment, monitoring, production hardening | Planned |

**Verified run:** `QNT-20261005T194746Z-BD3EE9` completed with `VALIDATED_WITH_WEAK_PERFORMANCE`. That status means the pipeline completed its checks, **not** that the investment strategy was profitable or institutionally validated.

## 3. Research methodology

### 3.1 Universe

1. Load a NIFTY 500 constituent snapshot.
2. Enrich market capitalization and other available metadata.
3. Rank companies by market cap.
4. Select the top 50 and bottom 50 companies.

**Warning:** This is a deliberately non-representative 100-stock sample. If today's constituents or market caps are used for historical periods, survivorship and look-ahead biases remain. Do not call this a historical point-in-time universe unless the historical snapshots are truly available and used at each rebalance.

### 3.2 CAPM and SML screening

For each stock, estimate market beta by regressing stock excess returns on benchmark excess returns:

\[
R_{i,t}-R_{f,t}=\alpha_i+\beta_i(R_{m,t}-R_{f,t})+\epsilon_{i,t}.
\]

CAPM required annual return:

\[
E[R_i]_{CAPM}=R_f+\beta_i(E[R_m]-R_f).
\]

Screening alpha:

\[
\alpha_{screen}=R_{observed,annual}-E[R_i]_{CAPM}.
\]

The **CAPM screening alpha** and the **regression intercept alpha** are distinct quantities. Positive screening alpha is a model-relative signal, **not proof that a stock is intrinsically undervalued**.

### 3.3 Portfolio construction

- Filter or rank CAPM candidates.
- Run initial constrained long-only maximum-Sharpe optimization.
- Score valuation (P/E, P/B), business scale/quality proxies (EBIT, market cap), volatility, liquidity (average volume), and available news sentiment.
- Select a core group (typically five names).
- Select diversifiers based on low correlation with the core group.
- Reoptimize the combined candidate set under risk constraints.

The original optimization uses SciPy SLSQP; configuration includes per-stock maximum weight, minimum core weights where feasible, and transaction-cost assumptions. A displayed 0% weight is not an active holding.

### 3.4 Harness and risk review

The LangGraph workflow is conceptually:

```text
Research request
   → Planner
   → Risk critic
   → [Bounded risk-constrained reoptimization if needed]
   → Freeze weights
   → Held-out validation and walk-forward analysis
   → Evidence/verdict
   → Auditable Markdown report
```

Examples of risk policies: largest position, top-three concentration, average correlation, volatility, and drawdown. Passing configured limits does not imply the portfolio is safe or suitable for any investor.

### 3.5 Research validation

- Use chronological train/test splitting.
- Keep the final test period out of optimization and model selection.
- Report CAGR, annualized volatility, Sharpe ratio, drawdown, benchmark comparisons, and optional factor-regression evidence.
- Report negative performance without changing the portfolio to make the test result positive.

**Phase 3.4 limitation:** Its walk-forward procedure reoptimizes a preselected asset set. It does **not** reconstruct historical constituents, market caps, fundamentals, sentiment, CAPM candidates, and selected assets at each date.

## 4. Phase 3.5 experimental overlay

The overlay adds these files to the existing repository:

```text
src/quanta/
  quant/robust.py
  portfolio/robust_optimizer.py
  validation/integrity.py
  validation/advanced_metrics.py
  validation/bootstrap.py
  validation/walk_forward35.py
  validation/evidence.py
  pipeline/phase35.py
scripts/run_phase35.py
tests/test_phase35.py
```

Features included in the overlay:

- Return-estimate shrinkage and covariance shrinkage.
- Alternative robust portfolio candidates and regularization.
- Integrity checks for chronology, benchmark presence, duplicate dates, minimum observations, and availability of dated risk-free / historical source files.
- Test-period transaction-cost haircut and turnover reporting.
- Benchmark-relative tracking error and information ratio.
- Walk-forward evaluation on the selected asset set.
- Block-bootstrap uncertainty estimates.
- Evidence signal: `STRONG`, `MODERATE`, `WEAK`, `AVOID`, or `INSUFFICIENT_EVIDENCE`.

**Critical caveats — do not overstate Phase 3.5:**

1. `strict_point_in_time: true` checks that specified historical files exist; **file existence alone does not validate timestamps or ensure the full historical selection pipeline uses them**.
2. The Phase 3.5 overlay does **not** implement full point-in-time universe reconstruction.
3. The current Phase 3.5 candidate-selection code fits candidate optimizers on the complete training set before scoring an inner validation slice. That is **not a fully leakage-free nested model selection** and must be corrected before claiming rigorous independent model comparison.
4. The overlay's static test-period portfolio is not equivalent to a fully rebalanced, trade-level backtest with realistic fills, spreads, market impact, and corporate actions.
5. An available code module or passing unit test is not proof that its financial conclusions are correct.

**Recommended release gate:** Treat Phase 3.5 as *experimental* until those issues are fixed, historical source files are validated, the entire pipeline is rerun, and an independent test report is reviewed.

## 5. Example results — completed Phase 3.4 run

**Run:** `QNT-20261005T194746Z-BD3EE9`  
**Research period:** 2023-01-01 to 2026-09-30  
**Benchmark symbol:** `^CRSLDX` (provider identity and availability should be independently verified)  
**Risk-free rate:** 6.50% static fallback, **not** a dated observation  
**Universe:** top 50 + bottom 50 by market capitalization  
**Positive CAPM-screening-alpha stocks:** 59

| Portfolio weight | Stock |
|---:|---|
| 17.22% | BEL.NS |
| 16.39% | BHARTIARTL.NS |
| 16.39% | MARUTI.NS |
| 16.39% | EICHERMOT.NS |
| 10.95% | JYOTHYLAB.NS |
| 10.27% | ADANIPOWER.NS |
| 7.10% | HINDUNILVR.NS |
| 3.03% | CERA.NS |
| 2.26% | SAPPHIRE.NS |
| 0.00% | AAVAS.NS — candidate only, not an active holding |

**Estimated optimizer metrics:** annual return 33.93%, annualized volatility 15.61%, Sharpe 1.758.

**Held-out performance (230 observations):**

| Metric | Portfolio | Benchmark where available |
|---|---:|---:|
| Total return | −11.18% | −5.37% |
| CAGR | −12.18% | −6.13% |
| Annualized volatility | 15.96% | — |
| Sharpe | −1.142 | — |
| Maximum drawdown | −15.10% | — |

**Risk:** PASS; largest weight 17.22%; top-three concentration 50.00%; average pairwise correlation 0.156; nine active positions.

**Walk-forward:** six completed windows; 83.33% profitable windows; 66.67% positive-Sharpe windows; combined CAGR 9.40%; Sharpe 0.237; drawdown −15.42%. These figures must be interpreted with the selected-asset-set limitation above.

**Final verdict:** `VALIDATED_WITH_WEAK_PERFORMANCE` — a negative held-out research result, not a software error.

## 6. Repository layout

```text
QUANTA_Phase_01_03/
├── config/
│   └── research.yaml
├── data/
│   ├── raw/
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
│   ├── tools/
│   ├── reporting/
│   └── cli.py
├── scripts/
│   ├── generate_demo_data.py
│   └── run_phase35.py       # after overlay integration
├── tests/
├── artifacts/              # generated run outputs
├── pyproject.toml
├── .env.example
├── .gitignore
└── README.md
```

## 7. Tech stack

| Layer | Technology | Purpose |
|---|---|---|
| Core | Python 3.11–3.12 | Research and orchestration |
| Data | pandas, NumPy | Time-series transformations |
| Market data | yfinance + local cache | Price and metadata ingestion |
| Statistics | statsmodels, SciPy | CAPM regression and optimization |
| Visualization | matplotlib | SML and research plots |
| Schemas/config | Pydantic, YAML | Validation and configuration |
| Agent harness | LangGraph | Bounded, auditable research workflow |
| CLI | Typer, Rich | Commands and terminal output |
| Tests | pytest | Unit and integration tests |
| Planned web | FastAPI, PostgreSQL, React | Phase 4 application |

## 8. Installation — Windows PowerShell

From the repository root, use Python 3.12:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
quanta doctor
pytest
```

If PowerShell prevents activation, adjust execution policy for the current process as appropriate for your environment, or invoke `.venv\Scripts\python.exe` directly.

### Optional LLM planner commentary

```powershell
pip install -e ".[dev,llm]"
```

Configure `OPENAI_API_KEY` in your environment and set `agentic.use_llm: true` if desired. The LLM must not be trusted to calculate or override numerical portfolio/risk results. LLM mode is optional; the deterministic research pipeline is the default.

## 9. Configuration

Main file: `config/research.yaml`. Core settings include:

```yaml
project_name: QUANTA
benchmark_ticker: "^CRSLDX"
start_date: 2023-01-01
end_date: 2026-09-30
annualization_factor: 252
validation_fraction: 0.25

risk_free:
  annual_rate: 0.065
  series_csv: "data/raw/india_risk_free.csv"
  fallback_to_static: true

universe:
  constituents_csv: "data/raw/nifty500_constituents.csv"
  top_n: 50
  bottom_n: 50

portfolio:
  max_weight: 0.20
  core_count: 5
  diversifier_count: 5
  transaction_cost_bps: 10
```

**Illustrative excerpt only.** Preserve the other required fields in your existing working configuration. Risk-free series support is only useful when the series exists and is actually loaded.

For Phase 3.5, add this section:

```yaml
phase35:
  strict_point_in_time: false
  pit_constituents_csv: "data/raw/nifty500_constituents_history.csv"
  pit_fundamentals_csv: "data/raw/fundamentals_point_in_time.csv"
  return_shrinkage: 0.60
  covariance_shrinkage: 0.25
  l2_penalty: 0.02
  bootstrap_samples: 1000
```

**Integration requirement:** If your config is a Pydantic model, declare `Phase35Config` on that model or enable appropriate extra-field handling; otherwise the YAML section may be ignored or rejected. `strict_point_in_time: false` permits exploratory research **with disclosed biases**; it does not certify the data.

## 10. Running QUANTA

### Existing working commands

```powershell
quanta phase1 --config config/research.yaml
quanta phase2 --config config/research.yaml
quanta phase3 --config config/research.yaml
```

Phase 3.4 can generate an artifact such as:

```text
artifacts/QNT-20261005T194746Z-BD3EE9/FINAL_REPORT.md
```

### Experimental Phase 3.5

1. Back up the existing repository.
2. Copy the Phase-3.5 overlay's `src/`, `scripts/`, and `tests/` into the repository root, preserving the working modules.
3. Add the `phase35` config schema and YAML section.
4. Run the tests.
5. Execute the dedicated script:

```powershell
pytest
python scripts/run_phase35.py
```

The overlay script reruns Phase 1 and Phase 2 and invokes `run_phase35(cfg, p2)`. It is **not** automatically part of `quanta phase3` unless you explicitly integrate it into the CLI/harness. A complete successful local Phase-3.5 run has not yet been confirmed.

Expected Phase-3.5 output files, if execution succeeds:

```text
phase35_integrity.json
phase35_result.json
phase35_test_returns.csv
phase35_walk_forward_returns.csv
```

Do not confuse Phase-3.4's `FINAL_REPORT.md` with the Phase-3.5 JSON output; the overlay does not automatically regenerate the former.

## 11. Data sources and schemas

| Input | Purpose | Current caution |
|---|---|---|
| NIFTY 500 constituents | Research universe | Current snapshots are not historical membership |
| Yahoo adjusted prices | Return estimation | Validate symbols, gaps, dividends and splits |
| Yahoo company metadata | Cap, valuation, liquidity proxies | Current metadata is not PIT fundamentals |
| Risk-free CSV | Dated short-rate inputs | Static 6.5% fallback currently used |
| News sentiment CSV | Dated text-based research factor | No future timestamps may enter past decisions |
| India factor CSV | Factor exposure / alpha tests | Optional until a verified series is supplied |
| Historical constituents CSV | True historical membership | Must include effective dates and actual integration |
| PIT fundamentals CSV | Historical published fundamentals | Must include as-of/availability timestamps and actual integration |

**No fabricated datasets:** If a critical historical source is missing, report the limitation or reject the strict run. Merely creating a CSV filename is not sufficient.

## 12. Testing and quality gates

Run:

```powershell
pytest -v
```

Tests should cover chronological splitting, beta/CAPM, optimizer feasibility, top-three concentration, risk verdicts, portfolio weight normalization, no test leakage, transaction costs, PIT timestamp filtering, benchmark alignment, and reporting compatibility.

**Historical test evidence:** Earlier local runs reported 11 passing tests for the working Phase-3.4 code. The Phase-3.5 overlay includes additional tests; their success in isolation does not prove complete integration with your latest local repository.

**Before declaring Phase 3.5 research-grade:**

- [ ] Fix nested candidate selection so every model is fitted on the inner-training slice and scored on unseen inner-validation data.
- [ ] Refit the chosen model using only the full outer-training data.
- [ ] Never use the outer test for hyperparameter/model selection.
- [ ] Implement historical membership/market-cap/fundamental/news reconstruction at every rebalance.
- [ ] Verify dated risk-free series and benchmark provider identity.
- [ ] Validate splits, dividends, missing data, and delistings.
- [ ] Charge realistic transaction costs on every trade/rebalance.
- [ ] Verify exposure, concentration, sector, liquidity, and turnover limits.
- [ ] Compare gross/net results and simple benchmark baselines.
- [ ] Evaluate bootstrap uncertainty, sensitivity, and multiple-testing risk.
- [ ] Rerun all tests and inspect full outputs independently.

## 13. Known limitations

1. Market returns and risk-free estimates are noisy; expected return is not a forecast guarantee.
2. Historical current-member sampling can create survivorship bias.
3. Fundamental and news features require point-in-time availability controls.
4. Missing financial fields and proxies may materially distort factor scoring.
5. A constrained optimizer can overfit historical means and covariance.
6. A PASS risk verdict means configured rules passed, not that losses are impossible.
7. Walk-forward performance is not fully PIT until the universe and features are rebuilt at each rebalance.
8. Benchmark selection and price series require external verification.
9. Backtests exclude some real-world execution frictions and investor-specific constraints.
10. Statistical significance and future profitability cannot be inferred from one historical test.

## 14. Roadmap

**Next: Phase 3.5 validation hardening.** Fix the inner-validation issue, implement actual historical as-of selection, verify source datasets, add end-to-end tests and reproducibility manifests, then review the generated evidence.

**Phase 4 (planned):** FastAPI backend, PostgreSQL research-run storage, React frontend, portfolio/risk charts, SML visualization, benchmark comparison, drawdown and walk-forward charts, live workflow status, audit trace, and report downloads.

**Phase 5 (planned):** Containerization, CI/CD, secrets management, monitoring, access controls, deployment and operational testing.

## 15. Disclaimer

QUANTA is an educational and quantitative research project. Its output is **research evidence, not personalized investment advice**. It does not assess an investor's financial circumstances, risk tolerance, regulatory eligibility, or suitability, and does not execute real-money orders. Historical or simulated results do not guarantee future performance.

---

**QUANTA — research-first, auditable, and explicit about uncertainty.**
