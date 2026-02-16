from __future__ import annotations

import numpy as np
import pandas as pd

from whsdsci.models.defense_two_step import DefenseAdjTwoStepModel
from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


def _sample_ev_df() -> pd.DataFrame:
    rows = []
    for g in range(1, 7):
        for team in ["A", "B", "C", "D"]:
            for off in ["first_off", "second_off"]:
                opp = "Z" if team != "Z" else "Y"
                deff = "first_def" if (g + len(team)) % 2 == 0 else "second_def"
                toi_hr = 0.05
                xg_rate = 0.8 if off == "first_off" else 0.5
                xg = xg_rate * toi_hr
                rows.append(
                    {
                        "game_id": f"game_{g}",
                        "off_unit": f"{team}__{off}",
                        "def_unit": f"{opp}__{deff}",
                        "offense_team": team,
                        "off_line": off,
                        "def_pair": deff,
                        "is_home": g % 2,
                        "toi_hr": toi_hr,
                        "xg_for": xg,
                    }
                )
    return pd.DataFrame(rows)


def test_strength_disparity_positive_and_complete():
    df = _sample_ev_df()
    model = DefenseAdjTwoStepModel(random_state=0).fit(df)
    strengths = compute_standardized_strengths(model, df)
    disparity = compute_disparity_ratios(strengths)

    teams_with_both = set(df[df["off_line"] == "first_off"]["offense_team"]).intersection(
        set(df[df["off_line"] == "second_off"]["offense_team"])
    )
    assert teams_with_both.issubset(set(disparity["team"]))
    assert np.all(np.isfinite(disparity["ratio"]))
    assert np.all(disparity["ratio"] > 0)
