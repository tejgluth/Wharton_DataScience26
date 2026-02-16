# Wharton HSDSc Hockey (Pruned Competition Runtime)

This repository is pruned to run only:
- Phase 1b confirmed best system
- Phase 1c deliverables
- Relevant Phase 1d section (Offensive Line Quality Disparity)

All outputs are written under `outputs/`.

## Run Phase 1b (Best System)
```bash
python -m whsdsci.run_phase1b_best
```

Writes:
- `outputs/final_top10.csv`
- `outputs/submission_phase1b.csv`
- `outputs/best_method.txt`
- `outputs/phase1b_run.log`

## Run Phase 1c
```bash
python -m whsdsci.run_phase1c
```

Writes:
- `outputs/phase1c/` artifacts
- `outputs/phase1c_output.csv`

## Run Phase 1d (Relevant Section)
```bash
python -m whsdsci.run_phase1d_relevant
```

Writes:
- `outputs/phase1d/` artifacts
- `outputs/phase1d_output.csv`

## Run Tests
```bash
pytest -q
```

## Full Prune Verification
```bash
python -m whsdsci.tools.prune_verify
```
