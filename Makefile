PYTHON ?= python3

.PHONY: run phase1c phase1d test

run:
	$(PYTHON) -m whsdsci.run_phase1b_best

phase1c:
	$(PYTHON) -m whsdsci.run_phase1c

phase1d:
	$(PYTHON) -m whsdsci.run_phase1d_relevant

test:
	$(PYTHON) -m pytest -q
