# Phase 1b Best System: One-Page Technical Analysis

## Executive Summary
The highest-accuracy system for Phase 1b (based on grouped out-of-sample EV Poisson deviance on xG totals) is an ensemble configuration from the exhaustive search:
- **Config**: `cfg_00699`
- **Combiner**: `tree_poisson`
- **Base pool**: `diverse_trio_7`
- **Meta model backend**: XGBoost Poisson objective (`count:poisson`)
- **Hyperparameters**: `max_depth=3`, `learning_rate=0.08`, `n_estimators=200`
- **Calibration**: scalar calibration on training OOF predictions
- **Primary score**: **mean CV Poisson deviance = 0.029877** (5-fold GroupKFold by `game_id`)

This model outperformed previously selected systems on the competition’s primary metric.

## What Problem This System Solves
Phase 1b asks for **schedule-independent line strength disparity**:
1. Estimate offensive strength for `first_off` and `second_off` lines using xG.
2. Correct for TOI exposure.
3. Correct for defensive matchup quality (defensive pairing + defending team).
4. Compute team disparity ratio: `Strength(first_off) / Strength(second_off)`.

The winning system is optimized for **predictive accuracy first** (competition objective), while preserving the required structure for downstream disparity estimation.

## Data and Leakage-Safe Evaluation Setup
The pipeline first converts segment-level game data into canonical long format (home offense row + away offense row), then filters to EV rows:
- `off_line in {first_off, second_off}`
- `def_pair in {first_def, second_def}`

Key safeguards:
- **Grouped CV by `game_id`** (no within-game leakage).
- Targets evaluated on **xG totals** (`xg_for`), not rates.
- Poisson-domain safety enforced globally: predictions clipped to `>= 1e-9` before deviance.
- Inner training-only OOF predictions for ensemble fitting/calibration.

## Architecture of the Winning System
### 1) Diverse base learners
`cfg_00699` uses seven complementary models:
- `POISSON_GLM_OFFSET`
- `POISSON_GLM_OFFSET_REG`
- `TWEEDIE_GLM_RATE`
- `TWO_STAGE_SHOTS_XG`
- `HURDLE_XG`
- `RIDGE_RAPM_RATE_SOFTPLUS`
- `DEFENSE_ADJ_TWO_STEP`

Why this helps: each base model captures different structure (offset Poisson count behavior, zero handling, shot-frequency/severity decomposition, regularized RAPM effects). Diversity lowers correlated error and improves stacked generalization.

### 2) Nonlinear Poisson meta-learner (`tree_poisson`)
The combiner learns on training-fold OOF base predictions (plus context features) with a Poisson loss. Conceptually:
- Inputs: transformed base predictions and context (`log_toi_hr`, `is_home`, regime cues)
- Objective: optimize expected Poisson fit to xG totals
- Output: positive mean prediction for each segment (`mu_hat`)

Using a Poisson objective at the meta layer aligns optimization with the scoring metric more directly than least-squares blending.

### 3) Scalar calibration layer
After raw ensemble prediction, the system applies a multiplicative scalar calibration learned on training OOF data to correct global under/over-prediction bias:
\[
\hat{\mu}_{cal} = s \cdot \hat{\mu}_{raw}, \quad s>0
\]
This tightened aggregate calibration (`sum(pred)/sum(true) ≈ 0.999`).

## Why It Won
From `outputs/ensemble_search_results.csv`, this system is the best among full/deep 5-fold-evaluated candidates. In practical terms, it wins because:
- It combines strong GLM-style count models with structurally different learners.
- It uses a nonlinear Poisson combiner, capturing interactions among base predictions.
- It is evaluated and tuned under strict grouped CV and no-leakage rules.
- It adds low-variance calibration without overcomplicating the stack.

## How It Produces Phase 1b Disparity Outputs
After selecting the best predictive model, line strengths are computed via **marginal standardization** (schedule-independent):
1. Build defense reference weights from EV training TOI over `def_unit`.
2. For each `off_unit`, predict against every `def_unit` on a synthetic grid (`toi_hr=1`, both home/away).
3. Average home/away predictions and weight by defense reference distribution.
4. Convert to `xG/60` line strengths and compute ratio:
\[
ratio_t = \frac{Strength(t, first\_off)+\epsilon}{Strength(t, second\_off)+\epsilon}
\]

This isolates line quality from observed schedule difficulty.

## Operational Outputs
The search and winner selection produce:
- `outputs/ensemble_search_results.csv` (all tested configs)
- `outputs/ensemble_best_config.json` (winner metadata)
- `outputs/best_method.txt`
- `outputs/final_top10.csv`
- `outputs/submission_phase1b.csv`

## Practical Takeaway
For Phase 1b’s scoring objective, the best total system is **`cfg_00699` tree-poisson stacking with scalar calibration over a diverse seven-model base pool**. It is currently the strongest model in this repo under the required grouped-CV, Poisson-safe, no-leakage protocol.
