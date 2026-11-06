# Atajos de desarrollo. En Windows, si no hay `make`, ejecutar los comandos a mano.
PYTHON ?= python

.PHONY: install test lint format smoke train eval report report-docx clean participation

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements-dev.txt
	pre-commit install

test:
	$(PYTHON) -m pytest -q --cov=src --cov-report=term-missing

lint:
	ruff check .
	black --check .

format:
	ruff check . --fix
	black .

smoke:
	$(PYTHON) scripts/make_synthetic_sample.py --out data/sample --n 8 --size 64
	$(PYTHON) -m src.gan.train --config configs/smoke.yaml --max-steps 2

train:
	$(PYTHON) -m src.gan.train --config configs/base.yaml

eval:
	$(PYTHON) -m src.gan.evaluate --config configs/base.yaml --split test

report:
	cd report && pandoc informe.md \
		--citeproc \
		--bibliography=referencias.bib \
		-o informe.pdf

report-docx:
	cd report && pandoc informe.md \
		--citeproc \
		--bibliography=referencias.bib \
		-o informe.docx

clean:
	rm -rf outputs .pytest_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +

participation:
	git shortlog -sne --all
