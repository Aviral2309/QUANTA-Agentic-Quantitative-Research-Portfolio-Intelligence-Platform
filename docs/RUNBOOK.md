# Runbook

## Windows / PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -e ".[dev]"
quanta doctor
pytest
quanta phase1 --config config/research.yaml
quanta phase2 --config config/research.yaml
quanta phase3 --config config/research.yaml
```

To enable the optional LLM planner:

```powershell
pip install -e ".[dev,llm]"
Copy-Item .env.example .env
# set OPENAI_API_KEY in your shell/environment
# set agentic.use_llm: true in config/research.yaml
quanta phase3 --config config/research.yaml
```

## Expected artifacts

Each run creates `artifacts/<run_id>/` with enriched universe, selected 100, CAPM results, positive-alpha list, SML plot, solver outputs, factor scores, diversifiers, backtest curve, factor regression, `audit.jsonl`, and `FINAL_REPORT.md`.

## First-run caveats

Metadata enrichment of ~500 Yahoo symbols can be slow and some symbols may fail. The cache prevents repeating successful downloads. Yahoo is a prototype source; for final academic claims use an authoritative/licensed point-in-time source and archive the exact snapshot.
