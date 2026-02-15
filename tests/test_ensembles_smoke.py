from __future__ import annotations

import numpy as np
import pandas as pd

from whsdsci.models import get_ensemble_model_builders, get_model_builders
from whsdsci.models.base import SkipModelError
from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


def _sample_ev_df(n_rows: int = 180, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    teams = [f"T{i}" for i in range(1, 9)]
    rows = []
    for i in range(n_rows):
        offense = teams[i % len(teams)]
        defense = teams[(i + 2) % len(teams)]
        off_line = "first_off" if i % 2 == 0 else "second_off"
        def_pair = "first_def" if (i // 2) % 2 == 0 else "second_def"
        toi_hr = rng.uniform(0.01, 0.09)
        base = 0.75 if off_line == "first_off" else 0.58
        xg_rate = np.clip(base + rng.normal(0, 0.08), 0.01, None)
        xg = xg_rate * toi_hr
        shots = rng.poisson(max(0.1, xg * 7.5))
        rows.append(
            {
                "game_id": f"game_{1 + (i % 20)}",
                "off_unit": f"{offense}__{off_line}",
                "def_unit": f"{defense}__{def_pair}",
                "offense_team": offense,
                "defense_team": defense,
                "off_line": off_line,
                "def_pair": def_pair,
                "is_home": i % 2,
                "toi_hr": toi_hr,
                "xg_for": xg,
                "shots_for": shots,
            }
        )
    return pd.DataFrame(rows)


def test_ensemble_models_smoke_fit_predict_and_strength():
    df = _sample_ev_df()
    base_builders_all = get_model_builders(random_state=0)
    base_names = [
        "POISSON_GLM_OFFSET_REG",
        "POISSON_GLM_OFFSET",
        "TWEEDIE_GLM_RATE",
        "RIDGE_RAPM_RATE_SOFTPLUS",
        "DEFENSE_ADJ_TWO_STEP",
    ]
    base_builders = {k: base_builders_all[k] for k in base_names if k in base_builders_all}

    ens_builders = get_ensemble_model_builders(
        random_state=0,
        base_model_builders=base_builders,
        base_model_names=list(base_builders.keys()),
    )

    n_checked = 0
    for method_name, factory in ens_builders.items():
        try:
            model = factory()
        except SkipModelError:
            continue

        model.fit(df)
        mu = model.predict_total(df)
        rate = model.predict_rate_hr(df)

        assert mu.shape[0] == len(df), method_name
        assert rate.shape[0] == len(df), method_name
        assert np.all(np.isfinite(mu)), method_name
        assert np.all(np.isfinite(rate)), method_name
        assert np.all(mu >= 1e-9), method_name
        assert np.all(rate >= 1e-12), method_name

        strengths = compute_standardized_strengths(model=model, train_ev_df=df)
        disparity = compute_disparity_ratios(strengths)
        assert set(["rank", "team", "line1_strength_xg60", "line2_strength_xg60", "ratio"]).issubset(disparity.columns)
        if not disparity.empty:
            assert np.all(np.isfinite(disparity["ratio"])), method_name
            assert np.all(disparity["ratio"] > 0), method_name
        n_checked += 1

    assert n_checked >= 4
