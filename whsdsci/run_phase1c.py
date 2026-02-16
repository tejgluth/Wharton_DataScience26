from __future__ import annotations

from pathlib import Path

import pandas as pd

from phases.phase1c.run import run_phase1c


def main() -> None:
    out_dir = Path("outputs") / "phase1c"
    result = run_phase1c(
        config_name=None,
        seed=1,
        out_dir=out_dir,
        small=False,
        features="baseline",
    )
    src = out_dir / "phase1c_viz_table.csv"
    dst = Path("outputs") / "phase1c_output.csv"
    if src.exists():
        pd.read_csv(src).to_csv(dst, index=False)
    print(result)


if __name__ == "__main__":
    main()

