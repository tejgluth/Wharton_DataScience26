PYTHON ?= python3

.PHONY: run test

run:
	$(PYTHON) -m whsdsci.run_phase1b_best

test:
	$(PYTHON) -m pytest -q
