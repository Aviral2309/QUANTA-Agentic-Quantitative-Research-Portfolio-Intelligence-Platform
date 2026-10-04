install:
	python -m pip install -e ".[dev]"
install-llm:
	python -m pip install -e ".[dev,llm]"
test:
	pytest
lint:
	ruff check src tests
phase1:
	quanta phase1 --config config/research.yaml
phase2:
	quanta phase2 --config config/research.yaml
phase3:
	quanta phase3 --config config/research.yaml
