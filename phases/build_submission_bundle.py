from __future__ import annotations

import shutil
from pathlib import Path


def _copy(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing required source for submission bundle: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def main() -> None:
    root = Path.cwd()
    submission = root / "submission"
    submission.mkdir(parents=True, exist_ok=True)

    _copy(root / "outputs" / "submission_phase1b.csv", submission / "phase1b" / "submission_phase1b.csv")
    _copy(root / "outputs" / "final_top10.csv", submission / "phase1b" / "final_top10.csv")

    _copy(
        root / "outputs" / "phase1c" / "phase1c_line_disparity_vs_team_strength.png",
        submission / "phase1c" / "phase1c_line_disparity_vs_team_strength.png",
    )
    _copy(root / "outputs" / "phase1c" / "phase1c_viz_table.csv", submission / "phase1c" / "phase1c_viz_table.csv")

    _copy(
        root / "phases" / "phase1d" / "offensive_line_quality_disparity.md",
        submission / "phase1d" / "offensive_line_quality_disparity.md",
    )

    checklist = []
    checklist.append("# Round 1 Submission Bundle")
    checklist.append("")
    checklist.append("## Included Files")
    checklist.append("- `phase1b/submission_phase1b.csv`")
    checklist.append("- `phase1b/final_top10.csv`")
    checklist.append("- `phase1c/phase1c_line_disparity_vs_team_strength.png`")
    checklist.append("- `phase1c/phase1c_viz_table.csv`")
    checklist.append("- `phase1d/offensive_line_quality_disparity.md`")
    checklist.append("")
    checklist.append("## Validation Notes")
    checklist.append("- Phase 1b generated from confirmed best config.")
    checklist.append("- Phase 1c image regenerated from current pipeline.")
    checklist.append("- Phase 1d is provided as finalized narrative markdown.")
    (submission / "SUBMISSION_CHECKLIST.md").write_text("\n".join(checklist) + "\n", encoding="utf-8")
    print(f"Submission bundle prepared at: {submission}")


if __name__ == "__main__":
    main()

