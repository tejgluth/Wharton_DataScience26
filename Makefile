PYTHON ?= python3

.PHONY: run phase1c phase1d bundle test

run:
	$(PYTHON) -m whsdsci.run_phase1b_best

phase1c:
	$(PYTHON) -m whsdsci.run_phase1c

phase1d:
	$(PYTHON) -m whsdsci.run_phase1d_relevant

bundle:
	$(PYTHON) -m phases.build_submission_bundle

test:
	$(PYTHON) -m pytest -q
