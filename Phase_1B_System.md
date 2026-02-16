# Phase 1B System (Confirmed Best) - Full Technical Explanation

## 1) Executive Summary
The confirmed best Phase 1b system in this repository is a **tree-poisson stacked ensemble** (config id: `cfg_00699`) trained on the EV-only long-format dataset and evaluated with grouped out-of-sample validation by `game_id`.

This system was selected because it produced the strongest predictive performance on the competition’s primary target:
- **Target**: segment-level `xg_for` totals
- **Primary score**: mean Poisson deviance (lower is better)
- **Leakage control**: grouped folds by game id

The final submission file is derived from this model after schedule-independent standardization of line strengths.

---

## 2) What Phase 1B Requires and Why This System Matches It
Phase 1B is not just about fitting xG. It specifically asks for line-strength disparity under matchup and exposure adjustments.

The system therefore addresses all required elements:

1. **Offensive line quality for Line 1 and Line 2**
   - We model offensive production at segment level and then aggregate into standardized line strengths.

2. **TOI/exposure adjustment**
   - Base models either include exposure logic directly (Poisson-style with `toi_hr`) or rate modeling with TOI-aware weighting.

3. **Defensive matchup adjustment**
   - Core features include `off_unit`, `def_unit`, and `is_home`; base models and stacker both condition on opponent defensive context.

4. **Schedule-independent disparity computation**
   - Strength is not naive observed mean. It is obtained via **marginal standardization** over a reference defense distribution.

5. **Final ranking**
   - `ratio = line1_strength_xg60 / line2_strength_xg60`, ranked descending.

---

## 3) Data Pipeline and Core Transform
The model starts from official `whl_2025.csv` and converts every segment to canonical long format (home offense row + away offense row), then filters EV state:
- `off_line in {first_off, second_off}`
- `def_pair in {first_def, second_def}`

Key engineered columns used throughout:
- `off_unit = offense_team + "__" + off_line`
- `def_unit = defense_team + "__" + def_pair`
- `toi_hr`
- `is_home`
- `xg_for`
- `shots_for`

Why this matters:
- The long table ensures each offense-defense interaction is represented in a symmetric and model-friendly way.
- EV filtering ensures Phase 1B comparability and avoids special-team confounding.

---

## 4) Frozen Best Config and Base Model Pool
The stack uses exactly the confirmed base pool from `cfg_00699`:
- `POISSON_GLM_OFFSET`
- `POISSON_GLM_OFFSET_REG`
- `TWEEDIE_GLM_RATE`
- `TWO_STAGE_SHOTS_XG`
- `HURDLE_XG`
- `RIDGE_RAPM_RATE_SOFTPLUS`
- `DEFENSE_ADJ_TWO_STEP`

These bases are intentionally diverse:
- count GLM views,
- rate-regularized views,
- two-stage frequency × severity structure,
- zero-handling structure,
- fast residual defense-adjusted baseline.

That diversity is critical for stacking: each base model has different bias/variance behavior, which gives the meta model useful disagreement structure.

---

## 5) Meta Model (Tree Poisson)
### 5.1 Meta Features
For each row:
- Base predictions matrix `P` over the seven models
- `log(P)` for stability and multiplicative behavior capture
- context: `log_toi_hr`, `is_home`, and `shots_zero`

### 5.2 Objective and Learner
Meta learner is Poisson-oriented tree regression:
- preferred backend: XGBoost with `objective='count:poisson'`
- fallback backend: sklearn `HistGradientBoostingRegressor(loss='poisson')`

Hyperparameters from confirmed config:
- `max_depth=3`
- `learning_rate=0.08`
- `n_estimators=200`

### 5.3 Positivity and Stability
All base and meta totals are clipped to `>= 1e-9`.
If a meta surface collapses numerically to near-constant predictions, a guarded variation fallback uses scaled mean base prediction to preserve non-degenerate behavior.

Why this is important:
- Poisson deviance has strict domain constraints.
- Stability guards prevent flat-strength artifacts on synthetic standardization grids.

---

## 6) Calibration Layer
After raw meta prediction, the system applies calibration from training predictions:
- default from config: **scalar calibration**
- optional in framework: none / scalar / piecewise scalar / isotonic

Scalar calibration solves for a positive multiplier `s` minimizing Poisson deviance:
- `mu_calibrated = s * mu_raw`

Why this works:
- Stacking can still have global bias even with good ranking power.
- A low-complexity multiplicative correction often improves calibration ratio without overfitting.

---

## 7) Handling Real-World Fit Failures Robustly
Some high-dimensional GLM fits may fail on small subsets (especially smoke scenarios) because of numeric issues in iterative fitting.

The production runtime includes explicit fallback behavior:
- If a base model fit fails, that base model is replaced for that fit with a global-rate fallback model.

Why this is safe:
- It keeps the pipeline deterministic and non-crashing.
- In full-data production runs, primary base models fit normally.
- In tiny tests/smoke contexts, fallback prevents false operational failures while preserving positivity and shape expectations.

---

## 8) Schedule-Independent Strength Standardization
This is the most important Phase 1B-specific post-model step.

For each offensive unit `j`:
1. Build reference defense weights `w_k` from EV TOI by `def_unit`.
2. Construct synthetic grid against every defense `k` at `toi_hr=1` and `is_home in {0,1}`.
3. Predict rates and average home/away.
4. Weight by `w_k` to get standardized `strength_rate_hr(j)`.
5. Convert to `strength_xg60(j) = strength_rate_hr(j)/60`.

Then for each team `t`:
- `L1 = strength_xg60(t__first_off)`
- `L2 = strength_xg60(t__second_off)`
- `ratio = (L1+eps)/(L2+eps)`

Why this is correct:
- It removes schedule artifacts from observed opponent mix.
- It isolates intrinsic offensive-line quality under a common defense exposure distribution.

---

## 9) Evaluation Discipline
The best system is selected under strict protocol:
- grouped by `game_id`
- out-of-sample deviance objective
- positivity-safe prediction handling

Operationally, this avoids classic sports-analytics leakage:
- multiple segments from same game cannot leak across train/test split,
- highly correlated within-game context does not inflate validation.

---

## 10) Why This System Performs Well in Practice
The system works because it combines:

1. **Strong parametric count/rate baselines**
   - robust signal capture from offense, defense, home, and exposure.

2. **Heterogeneous model views**
   - each base addresses different xG generation behavior.

3. **Nonlinear Poisson stacker**
   - learns interactions and conditional weighting patterns that fixed convex blends miss.

4. **Calibration and domain safety**
   - improves reliability of totals and keeps metrics valid.

5. **Competition-aligned post-processing**
   - standardized strength computation exactly matches Phase 1B intent.

---

## 11) End-to-End Runtime in This Repo
Main command:
```bash
python -m whsdsci.run_phase1b_best
```

What it does:
1. Resolves confirmed best config from `outputs/confirmed_best_config.json` (fallback `ensemble_best_config.json`).
2. Loads and builds EV dataset.
3. Fits `TreePoissonBestModel`.
4. Computes standardized strengths and disparity ratios.
5. Writes:
   - `outputs/final_top10.csv`
   - `outputs/submission_phase1b.csv`
   - `outputs/best_method.txt`
   - `outputs/phase1b_run.log`

---

## 12) Final Notes on Reliability and Competition Fit
- The system is intentionally frozen to reduce moving parts and submission risk.
- It is optimized for the competition scoring target while preserving interpretability at the line-strength stage.
- The repo is now organized so contributors can quickly trace:
  - where the model is defined (`phases/phase1b/system.py`),
  - where Phase 1B output is produced (`phases/phase1b/run.py`),
  - and how downstream Phase 1C/1D artifacts connect to this same confirmed model.

This combination of predictive strength, leakage-safe evaluation, and schedule-independent standardization is why this system is the strongest practical choice for Phase 1B in this codebase.
