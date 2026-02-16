# Wharton HSDSc Hockey (Competition-Ready Minimal Repo)

This repo is intentionally pruned to only what is needed for submission work:
- **Phase 1b**: run confirmed best model and produce submission files.
- **Phase 1c**: generate required visualization.
- **Phase 1d (relevant section)**: provide final written response file.

## Folder Guide
- `phases/phase1b/`: Phase 1b pipeline code.
- `phases/phase1c/`: Phase 1c visualization code.
- `phases/phase1d/`: Phase 1d written response (markdown).
- `whsdsci/models/`: minimal model components required by confirmed best config.
- `data/`: official competition data + PDFs (never delete).
- `outputs/`: generated artifacts.

## Run Phase 1b (Best Model)
```bash
python -m whsdsci.run_phase1b_best
```

Outputs:
- `outputs/final_top10.csv`
- `outputs/submission_phase1b.csv`
- `outputs/best_method.txt`
- `outputs/phase1b_run.log`

## Run Phase 1c (Visualization)
```bash
python -m whsdsci.run_phase1c
```

Outputs:
- `outputs/phase1c/phase1c_line_disparity_vs_team_strength.png`
- `outputs/phase1c/phase1c_viz_table.csv`
- `outputs/phase1c_output.csv`

## Phase 1d Written Response
- Primary file: `phases/phase1d/offensive_line_quality_disparity.md`
- Optional copy to outputs:
```bash
python -m whsdsci.run_phase1d_relevant
```
Creates:
- `outputs/phase1d/phase1d_offensive_line_quality_disparity.md`
- `outputs/phase1d_output.md`

## Run Tests
```bash
pytest -q
```
