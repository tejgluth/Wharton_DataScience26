from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.olqd import compute_team_olqd_table, make_olqd_feature_frame


class _DummyModel:
    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        off_line = df["off_unit"].astype(str).str.split("__", n=1).str[1]
        def_line = df["def_unit"].astype(str).str.split("__", n=1).str[1]
        base = np.where(off_line == "first_off", 2.8, 1.9)
        adj = np.where(def_line == "first_def", 0.95, 1.05)
        return np.clip(base * adj, 1e-12, None)

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        toi = np.maximum(pd.to_numeric(df["toi_hr"], errors="coerce").to_numpy(dtype=float), 1e-9)
        return np.clip(self.predict_rate_hr(df) * toi, 1e-9, None)


def _tiny_ev_df() -> pd.DataFrame:
    rows = []
    teams = ["A", "B", "C", "D"]
    for g in range(1, 9):
        for i in range(4):
            o = teams[(g + i) % len(teams)]
            d = teams[(g + i + 1) % len(teams)]
            off = "first_off" if i % 2 == 0 else "second_off"
            deff = "first_def" if (i // 2) % 2 == 0 else "second_def"
            toi_hr = 0.03 + 0.005 * ((g + i) % 3)
            rows.append(
                {
                    "game_id": f"game_{g}",
                    "offense_team": o,
                    "defense_team": d,
                    "off_line": off,
                    "def_pair": deff,
                    "off_unit": f"{o}__{off}",
                    "def_unit": f"{d}__{deff}",
                    "is_home": i % 2,
                    "toi_hr": toi_hr,
                    "log_toi_hr": np.log(max(toi_hr, 1e-9)),
                    "xg_for": 0.05 + 0.01 * i,
                    "shots_for": 1 + i,
                }
            )
    return pd.DataFrame(rows)


def test_olqd_feature_computation_tiny_synthetic():
    df = _tiny_ev_df()
    model = _DummyModel()
    team_olqd = compute_team_olqd_table(model=model, train_ev_df=df)

    assert not team_olqd.empty
    assert set(["team", "olqd_ratio", "olqd_log_ratio", "olqd_gap_xg60"]).issubset(team_olqd.columns)
    assert np.all(np.isfinite(team_olqd["olqd_ratio"]))
    assert np.all(team_olqd["olqd_ratio"] > 0)

    mu_base = np.full(len(df), 0.08, dtype=float)
    feats = make_olqd_feature_frame(df=df, mu_base_total=mu_base, team_olqd=team_olqd)
    expected = {
        "log_mu_base",
        "log_toi_hr",
        "is_home",
        "off_olqd_ratio",
        "def_olqd_ratio",
        "olqd_ratio_diff",
        "off_line_strength_xg60",
        "off_olqd_gap_xg60",
    }
    assert expected.issubset(feats.columns)
    assert np.all(np.isfinite(feats.to_numpy(dtype=float)))
