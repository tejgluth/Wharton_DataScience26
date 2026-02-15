from __future__ import annotations

import numpy as np
import pandas as pd

from whsdsci.eval.metrics import poisson_deviance_safe
from whsdsci.models import get_model_builders
from whsdsci.models.base import SkipModelError


def _make_sample_df(n_rows: int = 120, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    teams = [f"T{i}" for i in range(1, 9)]
    off_lines = ["first_off", "second_off"]
    def_lines = ["first_def", "second_def"]

    rows = []
    for i in range(n_rows):
        game = f"game_{1 + (i % 12)}"
        team_o = teams[i % len(teams)]
        team_d = teams[(i + 3) % len(teams)]
        off = off_lines[i % 2]
        deff = def_lines[(i // 2) % 2]
        toi_hr = rng.uniform(0.01, 0.08)
        base = 0.8 if off == "first_off" else 0.6
        xg_rate = base + rng.normal(0, 0.1)
        xg_rate = max(0.01, xg_rate)
        xg = xg_rate * toi_hr
        shots = rng.poisson(max(0.1, xg * 7.0))
        rows.append(
            {
                "game_id": game,
                "off_unit": f"{team_o}__{off}",
                "def_unit": f"{team_d}__{deff}",
                "offense_team": team_o,
                "defense_team": team_d,
                "off_line": off,
                "def_pair": deff,
                "is_home": i % 2,
                "toi_hr": toi_hr,
                "xg_for": xg,
                "shots_for": shots,
                "game_num": float(1 + (i % 12)),
            }
        )

    return pd.DataFrame(rows)


def test_models_smoke_fit_predict_positive():
    df = _make_sample_df()
    builders = get_model_builders(random_state=0)

    n_ran = 0
    for method_name, factory in builders.items():
        try:
            model = factory()
        except SkipModelError:
            continue

        model.fit(df)
        pred_rate = model.predict_rate_hr(df)
        pred_total = model.predict_total(df)

        assert np.all(np.isfinite(pred_rate)), method_name
        assert np.all(np.isfinite(pred_total)), method_name
        assert np.all(pred_rate >= 1e-12), method_name
        assert np.all(pred_total >= 1e-9), method_name

        dev = poisson_deviance_safe(df["xg_for"].to_numpy(dtype=float), pred_total)
        assert np.isfinite(dev), method_name
        n_ran += 1

    assert n_ran >= 8
