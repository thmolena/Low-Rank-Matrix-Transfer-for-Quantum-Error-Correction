PY ?= python

.PHONY: install verify test figures validate all

install:
	$(PY) -m pip install -e .

verify:
	$(PY) scripts/reproduce.py --verify

test:
	$(PY) -m pytest -q tests

figures:
	$(PY) scripts/make_figures.py --output results/figures

validate:
	$(PY) scripts/validate_release.py

all: verify test validate
