from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from whsdsci.build_long import build_canonical_long
from whsdsci.debug.debug_disparity import debug_best_disparity
from whsdsci.ensemble.search import Config, FittedSearchModel
from whsdsci.io import discover_paths
from whsdsci.models import get_model_builders
from whsdsci.models.poisson_glm_offset import PoissonGlmOffsetModel


def test_best_disparity_not_flat(tmp_path: Path):
    repo_root = Path.cwd()
    paths = discover_paths(repo_root=repo_root, outputs_dir=repo_root / "outputs")
    raw = pd.read_csv(paths["whl_2025"])
    _, ev_df, _ = build_canonical_long(raw_df=raw, outputs_dir=tmp_path)
    ev_df = ev_df.head(3500).copy()
    assert not ev_df.empty

    model = PoissonGlmOffsetModel(random_state=0).fit(ev_df)
    out = tmp_path / "debug_best_flatness.json"
    validity = debug_best_disparity(model=model, ev_df=ev_df, out_path=out, assert_valid=False)
    assert out.exists()
    assert validity.valid


def test_tree_ensemble_disparity_not_flat(tmp_path: Path):
    repo_root = Path.cwd()
    outputs_dir = repo_root / "outputs"
    paths = discover_paths(repo_root=repo_root, outputs_dir=outputs_dir)
    raw = pd.read_csv(paths["whl_2025"])
    _, ev_df, _ = build_canonical_long(raw_df=raw, outputs_dir=tmp_path)
    ev_df = ev_df.head(4000).copy()
    assert not ev_df.empty

    res_path = outputs_dir / "ensemble_search_results.csv"
    if not res_path.exists():
        return
    res = pd.read_csv(res_path)
    res = res[(res["status"] == "OK") & (res["combiner_family"].astype(str) == "tree_poisson")]
    if res.empty:
        return
    row = res.sort_values("mean_cv_poisson_deviance").iloc[0]
    cfg = Config(
        config_id=str(row["config_id"]),
        combiner_family=str(row["combiner_family"]),
        base_pool_id=str(row["base_pool_id"]),
        base_models=list(json.loads(row["base_models"])),
        calibration_type=str(row["calibration_type"]),
        hyperparams=dict(json.loads(row["hyperparams"])),
    )
    model = FittedSearchModel(
        random_state=0,
        base_builders=get_model_builders(random_state=0),
        config=cfg,
    ).fit(ev_df)
    validity = debug_best_disparity(
        model=model,
        ev_df=ev_df,
        out_path=tmp_path / "debug_tree_flatness.json",
        assert_valid=False,
    )
    assert validity.valid
