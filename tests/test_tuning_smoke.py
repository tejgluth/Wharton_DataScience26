from __future__ import annotations

from pathlib import Path

import pandas as pd

from whsdsci.build_long import build_canonical_long
from whsdsci.io import discover_paths
from whsdsci.models import get_model_builders
from whsdsci.tuning.maximize_tree_poisson import maximize_tree_poisson


def test_tuning_smoke(tmp_path: Path):
    repo_root = Path.cwd()
    paths = discover_paths(repo_root=repo_root, outputs_dir=repo_root / "outputs")
    raw = pd.read_csv(paths["whl_2025"])
    _, ev_df, _ = build_canonical_long(raw_df=raw, outputs_dir=tmp_path)
    ev_df = ev_df.head(3000).copy()
    assert not ev_df.empty

    out = tmp_path / "outputs"
    out.mkdir(parents=True, exist_ok=True)

    builders_all = get_model_builders(random_state=0)
    keep = [
        "POISSON_GLM_OFFSET",
        "POISSON_GLM_OFFSET_REG",
        "TWO_STAGE_SHOTS_XG",
        "DEFENSE_ADJ_TWO_STEP",
        "BASELINE_LINE_MEAN_RATE",
    ]
    builders = {k: builders_all[k] for k in keep}

    baseline = {
        "base_models": ["POISSON_GLM_OFFSET", "POISSON_GLM_OFFSET_REG", "TWO_STAGE_SHOTS_XG", "DEFENSE_ADJ_TWO_STEP", "BASELINE_LINE_MEAN_RATE"],
        "hyperparams": {"backend": "auto", "max_depth": 3, "learning_rate": 0.08, "n_estimators": 80},
    }
    result = maximize_tree_poisson(
        ev_df=ev_df,
        outputs_dir=out,
        base_builders=builders,
        baseline_config=baseline,
        random_state=0,
        stage1_trials=5,
        stage2_trials=0,
        max_time_hours=0.1,
    )
    assert "best_payload" in result
    assert (out / "maximize_tree_poisson_results.csv").exists()
    assert (out / "maximize_tree_poisson_best_config.json").exists()
    assert (out / "submission_phase1b.csv").exists()

