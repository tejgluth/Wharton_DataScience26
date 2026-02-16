from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from whsdsci.build_long import build_canonical_long
from whsdsci.debug.debug_disparity import check_phase1b_validity_tables
from whsdsci.io import discover_paths
from whsdsci.models.poisson_glm_offset import PoissonGlmOffsetModel
from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


def test_phase1b_validity_non_flat():
    repo_root = Path.cwd()
    paths = discover_paths(repo_root=repo_root, outputs_dir=repo_root / "outputs")
    raw = pd.read_csv(paths["whl_2025"])
    _, ev_df, _ = build_canonical_long(raw_df=raw, outputs_dir=repo_root / "outputs")
    ev_df = ev_df.head(4000).copy()
    assert not ev_df.empty

    model = PoissonGlmOffsetModel(random_state=0).fit(ev_df)
    strengths = compute_standardized_strengths(model=model, train_ev_df=ev_df)
    ratios = compute_disparity_ratios(strengths)
    validity = check_phase1b_validity_tables(strength_df=strengths, ratio_df=ratios)

    assert validity.valid
    assert validity.ratio_unique_1e6 >= 3
    assert validity.log_ratio_std > 1e-8
    assert np.isfinite(validity.strength_std_xg60)

