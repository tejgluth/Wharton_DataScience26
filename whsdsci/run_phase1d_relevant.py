from __future__ import annotations

from pathlib import Path

import pandas as pd

from analysis.olqd_report import run_olqd_report


def main() -> None:
    out_dir = Path("outputs") / "phase1d"
    result = run_olqd_report(
        config_name=None,
        out_dir=out_dir,
        seed=1,
        small=False,
    )
    src = out_dir / "olqd_team_table.csv"
    dst = Path("outputs") / "phase1d_output.csv"
    if src.exists():
        pd.read_csv(src).to_csv(dst, index=False)
    print(result)


if __name__ == "__main__":
    main()

