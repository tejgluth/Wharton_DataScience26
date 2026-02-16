# Phase 1d: Offensive Line Quality Disparity (OLQD)

## What OLQD Means
For each team:

- `Line 1 quality` = standardized expected goals per 60 for `first_off`
- `Line 2 quality` = standardized expected goals per 60 for `second_off`
- `OLQD ratio = Line1 / Line2`

This ratio measures whether a team is top-line heavy (`ratio > 1`) or more balanced (`ratio` closer to `1`).

## How It Was Computed
1. Fit the confirmed best Phase 1b model (`cfg_00699`, tree-poisson stack) on EV data.
2. Compute schedule-independent line strengths by marginalizing over defense usage.
3. Compute team-level `Line1`, `Line2`, and `ratio`.
4. Rank teams by ratio.

## What We Observed
- Top-line-heavy teams had the highest ratios.
- The strongest disparity teams in the final Phase 1b submission were led by:
  `guatemala`, `usa`, `saudi_arabia`, `uae`, `france`.
- This supports the interpretation that line concentration differs meaningfully across teams.

## Competition-Facing Interpretation
OLQD is useful because it translates lineup depth into a single, interpretable value.
Teams with very high OLQD may rely more on first-line offense, while teams with lower OLQD are generally more balanced across their top two lines.

## Files To Reference
- `outputs/final_top10.csv`
- `outputs/submission_phase1b.csv`
- `outputs/phase1c/phase1c_line_disparity_vs_team_strength.png`
