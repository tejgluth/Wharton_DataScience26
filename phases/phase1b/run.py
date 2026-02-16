from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from phases.common import setup_logger
from whsdsci.best_config import resolve_best_config
from whsdsci.build_long import build_canonical_long
from whsdsci.io import discover_paths
from whsdsci.models.tree_poisson_best import TreePoissonBestModel
from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


def run_phase1b_best(repo_root: Path | None = None, outputs_dir: Path | None = None) -> pd.DataFrame:
    repo = Path.cwd() if repo_root is None else Path(repo_root)
    out = repo / "outputs" if outputs_dir is None else Path(outputs_dir)
    out.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("phase1b_best", out / "phase1b_run.log")

    paths = discover_paths(repo_root=repo, outputs_dir=out)
    raw = pd.read_csv(paths["whl_2025"])
    _, ev_df, _ = build_canonical_long(raw_df=raw, outputs_dir=out)
    if ev_df.empty:
        raise RuntimeError("EV dataset is empty; cannot run Phase 1b best model.")

    cfg = resolve_best_config(outputs_dir=out)
    logger.info("Loaded best config: %s", cfg.config_id)
    logger.info("Combiner family: %s", cfg.combiner_family)
    logger.info("Base models: %s", cfg.base_models)

    model = TreePoissonBestModel(config=cfg, outputs_dir=out, random_state=1).fit(ev_df)
    strengths = compute_standardized_strengths(model=model, train_ev_df=ev_df)
    ratios = compute_disparity_ratios(strengths).copy()
    ratios["method"] = f"{cfg.combiner_family}::{cfg.config_id}"
    top10 = ratios.sort_values("ratio", ascending=False).head(10).copy()
    top10["rank"] = range(1, len(top10) + 1)
    top10 = top10[["rank", "team", "line1_strength_xg60", "line2_strength_xg60", "ratio", "method"]]

    top10.to_csv(out / "final_top10.csv", index=False)
    top10.to_csv(out / "submission_phase1b.csv", index=False)
    with (out / "best_method.txt").open("w", encoding="utf-8") as f:
        f.write(f"best_method={cfg.combiner_family}\n")
        f.write(f"config_id={cfg.config_id}\n")
        f.write(f"base_pool_id={cfg.base_pool_id}\n")
        f.write(f"calibration_type={cfg.calibration_type}\n")
        f.write(f"mean_cv_poisson_deviance={cfg.metadata.get('mean_cv_poisson_deviance', 'na')}\n")
        f.write(f"std_cv_poisson_deviance={cfg.metadata.get('std_cv_poisson_deviance', 'na')}\n")
        f.write(f"full_config={json.dumps({'base_models': cfg.base_models, 'hyperparams': cfg.hyperparams})}\n")

    logger.info("Wrote outputs/final_top10.csv and outputs/submission_phase1b.csv")
    print(top10.to_string(index=False))
    return top10


def main() -> None:
    run_phase1b_best()


if __name__ == "__main__":
    main()

