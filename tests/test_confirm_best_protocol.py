from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from whsdsci.build_long import build_canonical_long
from whsdsci.io import discover_paths
from whsdsci.run_best_system_maximizer import confirm_best_system


def test_confirm_best_protocol(tmp_path: Path):
    repo_root = Path.cwd()
    paths = discover_paths(repo_root=repo_root, outputs_dir=repo_root / "outputs")
    raw = pd.read_csv(paths["whl_2025"])
    _, ev_df, _ = build_canonical_long(raw_df=raw, outputs_dir=tmp_path)
    ev_df = ev_df.head(2500).copy()
    assert not ev_df.empty

    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    # Tiny candidate set for smoke check.
    candidates = pd.DataFrame(
        [
            {
                "config_id": "smoke_cfg_tree",
                "combiner_family": "tree_poisson",
                "base_pool_id": "smoke_pool",
                "base_models": json.dumps(
                    ["POISSON_GLM_OFFSET", "POISSON_GLM_OFFSET_REG", "TWO_STAGE_SHOTS_XG"]
                ),
                "calibration_type": "scalar",
                "hyperparams": json.dumps(
                    {"backend": "auto", "max_depth": 3, "learning_rate": 0.08, "n_estimators": 100}
                ),
                "stage": "full",
                "mean_cv_poisson_deviance": 0.2,
                "status": "OK",
            },
            {
                "config_id": "smoke_cfg_stack",
                "combiner_family": "stack_poisson",
                "base_pool_id": "smoke_pool",
                "base_models": json.dumps(
                    ["POISSON_GLM_OFFSET", "POISSON_GLM_OFFSET_REG", "TWO_STAGE_SHOTS_XG"]
                ),
                "calibration_type": "none",
                "hyperparams": json.dumps({"alpha": 1e-4, "include_ctx": True}),
                "stage": "full",
                "mean_cv_poisson_deviance": 0.21,
                "status": "OK",
            },
        ]
    )
    candidates.to_csv(out / "ensemble_search_results.csv", index=False)

    result = confirm_best_system(
        ev_df=ev_df,
        outputs_dir=out,
        random_state=0,
        recheck_seeds=[0],
        recheck_top_k=1,
    )
    assert "best_payload" in result
    payload = result["best_payload"]
    assert "config_id" in payload
    assert payload["mean_cv_poisson_deviance"] >= 0
    assert (out / "confirmed_best_config.json").exists()
    assert (out / "confirmed_best_report.md").exists()
