PYTHON ?= python3

.PHONY: run test

run:
	$(PYTHON) -m whsdsci.run_all

test:
	$(PYTHON) -m pytest -q
