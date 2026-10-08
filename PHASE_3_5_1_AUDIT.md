# QUANTA Phase 3.5.1 — Correctness patch

## Verified fixes
- Drawdown includes initial equity of 1.0.
- Research integrity reports `PASS_WITH_LIMITATIONS` when required PIT evidence is absent.
- File existence alone never certifies PIT membership/fundamentals.
- Strict PIT mode rejects runs lacking verified evidence.
- Model-family selection uses inner-training fit and inner-validation scoring; selected family is then refit on full training.
- Evidence output is `INSUFFICIENT_EVIDENCE` when historical asset selection is not independently reconstructed.
- Overlapping walk-forward evaluation windows are rejected.
- 17 local unit tests passed.

## Important limitations
This release is a correctness/safety patch, NOT an independently verified profitable or historically investable strategy.
Phase 2 final-assets selection can still reflect future information; walk-forward remains fixed-universe sensitivity analysis.
True PIT reconstruction requires real historical constituent, market-cap, published-fundamental, and news datasets with verified as-of timestamps.
Dated risk-free series, benchmark source provenance, turnover accounting under drifting holdings, and live-provider integration still require further audit.
Do not use the output as personalized financial advice.

## Run
`python -m pip install -e '.[dev]'`
`pytest`
`python scripts/run_phase35.py`
