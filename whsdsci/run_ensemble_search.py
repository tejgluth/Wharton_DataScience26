from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from whsdsci.build_long import build_canonical_long
from whsdsci.ensemble.search import run_ensemble_search
from whsdsci.io import discover_paths
from whsdsci.pdf_read import write_pdf_notes


RANDOM_STATE = 0


def setup_logging(outputs_dir: Path) -> logging.Logger:
    logger = logging.getLogger("whsdsci.ensemble_search")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    outputs_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(outputs_dir / "ensemble_search.log", mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def run_pytest(repo_root: Path, logger: logging.Logger) -> None:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    logger.info("Running pytest before ensemble search: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    logger.info("pytest stdout:\n%s", proc.stdout)
    if proc.stderr:
        logger.info("pytest stderr:\n%s", proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError("pytest failed; stopping ensemble search")


def main() -> None:
    repo_root = Path.cwd()
    outputs_dir = repo_root / "outputs"

    logger = setup_logging(outputs_dir)
    logger.info("Starting exhaustive ensemble search")

    np.random.seed(RANDOM_STATE)
    os.environ["PYTHONHASHSEED"] = "0"

    paths = discover_paths(repo_root=repo_root, outputs_dir=outputs_dir)
    write_pdf_notes(paths=paths, out_path=outputs_dir / "pdf_notes.txt")

    whl_path = Path(paths["whl_2025"])
    raw = pd.read_csv(whl_path)
    _, ev_df, _ = build_canonical_long(raw_df=raw, outputs_dir=outputs_dir)
    if ev_df.empty:
        raise RuntimeError("EV subset is empty; cannot run ensemble search")

    run_pytest(repo_root=repo_root, logger=logger)

    result = run_ensemble_search(
        ev_df=ev_df,
        outputs_dir=outputs_dir,
        random_state=RANDOM_STATE,
        screen_target=350,
        full_target=50,
        deep_target=10,
        deep_max_configs=800,
        deep_max_variants_per_seed=180,
        enable_stability_tiebreak=False,
        logger=logger,
    )

    top20 = result["top20"]
    best_cfg = result["best_config"]

    logger.info("Best ensemble config: %s", best_cfg)
    logger.info("Top 20 ensemble configs by deviance:\n%s", top20.to_string(index=False))

    print("Top 20 ensemble configs by mean CV Poisson deviance")
    print(top20.to_string(index=False))
    print()
    print("Winner")
    print(best_cfg)


if __name__ == "__main__":
    main()
