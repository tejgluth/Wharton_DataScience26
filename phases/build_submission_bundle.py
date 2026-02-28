from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required source for submission bundle: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main() -> None:
    root = Path.cwd()
    submission = root / "submission"
    submission.mkdir(parents=True, exist_ok=True)

    league_table = root / "data" / "league_table.csv"
    if league_table.exists():
        _copy(league_table, submission / "phase1a" / "power_ranking_phase1a.csv")

    _copy(root / "outputs" / "submission_phase1b.csv", submission / "phase1b" / "submission_phase1b.csv")
    _copy(root / "outputs" / "final_top10.csv", submission / "phase1b" / "final_top10.csv")

    _copy(
        root / "outputs" / "phase1c" / "phase1c_line_disparity_vs_team_strength.png",
        submission / "phase1c" / "phase1c_line_disparity_vs_team_strength.png",
    )
    _copy(root / "outputs" / "phase1c" / "phase1c_viz_table.csv", submission / "phase1c" / "phase1c_viz_table.csv")

    _copy(
        root / "phases" / "phase1d" / "phase1d_full_response.txt",
        submission / "phase1d" / "phase1d_full_response.txt",
    )

    out = root / "outputs"
    paths = json.loads((out / "paths.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(paths["whl_2025"])
    teams = set(raw["home_team"].astype(str)).union(set(raw["away_team"].astype(str)))
    p1b = pd.read_csv(out / "submission_phase1b.csv")
    p1c = pd.read_csv(out / "phase1c_output.csv")
    p1d_text = (out / "phase1d_output.txt").read_text(encoding="utf-8")

    checks = []
    checks.append(("phase1b_rows_10", len(p1b) == 10))
    checks.append(("phase1b_rank_1_to_10", p1b["rank"].tolist() == list(range(1, 11))))
    checks.append(("phase1b_ratio_positive", bool((p1b["ratio"] > 0).all())))
    checks.append(("phase1b_ratio_sorted_desc", bool(p1b["ratio"].is_monotonic_decreasing)))
    checks.append(("phase1b_team_labels_match_whl_2025", bool(p1b["team"].astype(str).isin(teams).all())))
    checks.append(
        (
            "phase1c_required_columns",
            {"disparity_rank", "team", "line1_strength_xg60", "line2_strength_xg60", "ratio", "points", "team_strength_rank"}.issubset(
                set(p1c.columns)
            ),
        )
    )
    checks.append(("phase1c_nonempty", len(p1c) > 0))
    checks.append(("phase1d_has_required_sections", all(k in p1d_text for k in ["1) Process", "2) Tools and Techniques", "3) Your Predictions", "4) Your Insights"])))

    def sha(p: Path) -> str:
        h = hashlib.sha256()
        h.update(p.read_bytes())
        return h.hexdigest()

    checks.append(
        (
            "bundle_phase1b_matches_outputs",
            sha(submission / "phase1b" / "submission_phase1b.csv") == sha(out / "submission_phase1b.csv"),
        )
    )
    checks.append(
        (
            "bundle_phase1c_png_matches_outputs",
            sha(submission / "phase1c" / "phase1c_line_disparity_vs_team_strength.png")
            == sha(out / "phase1c" / "phase1c_line_disparity_vs_team_strength.png"),
        )
    )
    checks.append(
        (
            "bundle_phase1d_matches_source",
            sha(submission / "phase1d" / "phase1d_full_response.txt")
            == sha(root / "phases" / "phase1d" / "phase1d_full_response.txt"),
        )
    )
    overall_ok = all(ok for _, ok in checks)

    audit = {
        "overall_ok": overall_ok,
        "checks": [{"name": n, "ok": bool(ok)} for n, ok in checks],
        "phase1b_top10": p1b.to_dict(orient="records"),
        "phase1c_rows": int(len(p1c)),
        "phase1d_chars": int(len(p1d_text)),
    }
    (out / "phase1_submission_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")

    checklist = []
    checklist.append("# Round 1 Submission Bundle")
    checklist.append("")
    checklist.append("## Included Files")
    if league_table.exists():
        checklist.append("- `phase1a/power_ranking_phase1a.csv`")
    checklist.append("- `phase1b/submission_phase1b.csv`")
    checklist.append("- `phase1b/final_top10.csv`")
    checklist.append("- `phase1c/phase1c_line_disparity_vs_team_strength.png`")
    checklist.append("- `phase1c/phase1c_viz_table.csv`")
    checklist.append("- `phase1d/phase1d_full_response.txt`")
    checklist.append("")
    checklist.append("## Validation Notes")
    checklist.append(f"- overall_ok: `{overall_ok}`")
    for n, ok in checks:
        checklist.append(f"- {n}: `{'PASS' if ok else 'FAIL'}`")
    (submission / "SUBMISSION_CHECKLIST.md").write_text("\n".join(checklist) + "\n", encoding="utf-8")

    final_txt = []
    final_txt.append("PHASE 1 FINAL SUBMISSION PACKAGE")
    final_txt.append("")
    final_txt.append("This file summarizes the final, validated artifacts for Round 1.")
    final_txt.append("")
    final_txt.append("Deliverables by phase:")
    if league_table.exists():
        final_txt.append("Phase 1a:")
        final_txt.append("- submission/phase1a/power_ranking_phase1a.csv (32-team ranking table)")
    final_txt.append("Phase 1b:")
    final_txt.append("- submission/phase1b/submission_phase1b.csv (top-10 disparity submission)")
    final_txt.append("- submission/phase1b/final_top10.csv (same ranking, method-tagged)")
    final_txt.append("Phase 1c:")
    final_txt.append("- submission/phase1c/phase1c_line_disparity_vs_team_strength.png")
    final_txt.append("- submission/phase1c/phase1c_viz_table.csv")
    final_txt.append("Phase 1d:")
    final_txt.append("- submission/phase1d/phase1d_full_response.txt")
    final_txt.append("")
    final_txt.append("Quality checks:")
    final_txt.append(f"- Overall status: {'PASS' if overall_ok else 'FAIL'}")
    for n, ok in checks:
        final_txt.append(f"- {n}: {'PASS' if ok else 'FAIL'}")
    final_txt.append("")
    final_txt.append("Phase 1b model summary:")
    final_txt.append("- confirmed config_id: cfg_00699")
    final_txt.append("- combiner: tree_poisson")
    final_txt.append(
        "- base models: POISSON_GLM_OFFSET, TWEEDIE_GLM_RATE, TWO_STAGE_SHOTS_XG, POISSON_GLM_OFFSET_REG, HURDLE_XG, RIDGE_RAPM_RATE_SOFTPLUS, DEFENSE_ADJ_TWO_STEP"
    )
    final_txt.append("- ranking generated with schedule-independent standardized line strengths")
    final_txt.append("")
    final_txt.append("Top 10 teams (Phase 1b submission):")
    final_txt.append(p1b.to_string(index=False))
    final_txt.append("")
    final_txt.append("Reference files:")
    final_txt.append("- outputs/phase1_submission_audit.json")
    final_txt.append("- Phase_1B_System.md")
    (submission / "PHASE1_FINAL_SUBMISSION.txt").write_text("\n".join(final_txt) + "\n", encoding="utf-8")
    print(f"Submission bundle prepared at: {submission}")


if __name__ == "__main__":
    main()
