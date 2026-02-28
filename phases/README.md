# Phases Folder Guide

## `phase1b/`
- `system.py`: frozen best-model system (config loading + stack + calibration).
- `run.py`: executes Phase 1b end-to-end and writes submission CSV.
- `artifacts/`: latest phase-specific copies of Phase 1b outputs.

## `phase1c/`
- `run.py`: builds required Phase 1c disparity vs team-strength visualization.
- `artifacts/`: latest phase-specific copies of Phase 1c outputs.

## `phase1d/`
- `phase1d_full_response.txt`: finalized written response section for Phase 1d (plain text).

## Utility
- `build_submission_bundle.py`: creates the `submission/` folder from current outputs and phase files.
