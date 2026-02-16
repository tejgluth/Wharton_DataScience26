# Wharton HSDSc Hockey - Round 1 (Simple, Phase-Oriented Repo)

This repository is organized by phase and kept minimal for competition execution.

## Top-Level Purpose
- `phases/phase1b/`: confirmed best Phase 1b model and run logic.
- `phases/phase1c/`: Phase 1c visualization pipeline.
- `phases/phase1d/`: finalized Phase 1d narrative file(s).
- `whsdsci/`: thin reusable utilities and command wrappers.
- `submission/`: final round-ready submission bundle.
- `outputs/`: generated runtime artifacts.
- `data/`: competition data and PDF guidelines.

## Phase Commands

### Phase 1b (best system)
```bash
python -m whsdsci.run_phase1b_best
```
Writes:
- `outputs/final_top10.csv`
- `outputs/submission_phase1b.csv`
- `outputs/best_method.txt`
- `outputs/phase1b_run.log`
- `phases/phase1b/artifacts/` copies

### Phase 1c (required visualization)
```bash
python -m whsdsci.run_phase1c
```
Writes:
- `outputs/phase1c/phase1c_line_disparity_vs_team_strength.png`
- `outputs/phase1c/phase1c_viz_table.csv`
- `outputs/phase1c_output.csv`
- `phases/phase1c/artifacts/` copies

### Phase 1d (written response)
Primary source file:
- `phases/phase1d/offensive_line_quality_disparity.md`

Optional output copy:
```bash
python -m whsdsci.run_phase1d_relevant
```
Writes:
- `outputs/phase1d/phase1d_offensive_line_quality_disparity.md`
- `outputs/phase1d_output.md`

## Build Round Submission Folder
After running Phase 1b and Phase 1c:
```bash
python -m phases.build_submission_bundle
```

Creates:
- `submission/phase1b/submission_phase1b.csv`
- `submission/phase1b/final_top10.csv`
- `submission/phase1c/phase1c_line_disparity_vs_team_strength.png`
- `submission/phase1c/phase1c_viz_table.csv`
- `submission/phase1d/offensive_line_quality_disparity.md`
- `submission/SUBMISSION_CHECKLIST.md`

## Tests
```bash
pytest -q
```

## System Documentation
- Detailed best-model explanation: `Phase_1B_System.md`
