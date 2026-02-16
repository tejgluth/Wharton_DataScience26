# Phase 1d (Offensive Line Quality Disparity)

## Definition
Offensive Line Quality Disparity (OLQD) is the team-level ratio of standardized first-line offensive strength to second-line offensive strength.

## How OLQD Was Computed
- Fit the selected Phase 1b model on EV data.
- Compute schedule-standardized line strengths (`xG/60`) via marginalization over defense usage.
- Compute `OLQD = line1_strength_xg60 / line2_strength_xg60` per team.

## Why It Matters
The competition question asks whether teams with more even offensive lines perform better. OLQD provides a compact measure of top-line concentration versus balanced depth.

## Baseline vs Baseline+OLQD Ablation (Grouped CV)
```
       method  mean_poisson_deviance  std_poisson_deviance  mean_weighted_mse_rate  mean_mae_total  mean_calibration_ratio
     baseline               0.029890              0.000608                2.486730        0.045749                0.998427
baseline+olqd               0.029932              0.000606                2.490646        0.045790                0.998359
```

Per-fold deviance delta (`baseline+olqd - baseline`):
```
                                                  metric  n_folds  mean_delta  std_delta  ci95_half_width
poisson_deviance_delta_baseline_plus_olqd_minus_baseline        5    0.000042   0.000011          0.00001
```

Interpretation:
- Mean delta = 0.000042003 with CI half-width ≈ 0.000009507. Negative favors OLQD augmentation; positive favors baseline.
- In this run, OLQD-derived context was tested as an additive correction layer to isolate whether disparity signal improves out-of-sample xG prediction.

## Figures
- `figures/olqd_ratio_vs_team_strength.png`
- `figures/olqd_ablation_by_fold.png`

## Model Behavior and Error Modes
OLQD features mainly target systematic errors where baseline predictions under/over-estimate teams with unusually concentrated top-line offense. If fold deltas are near zero, disparity information may already be captured by matchup-adjusted base predictors.
