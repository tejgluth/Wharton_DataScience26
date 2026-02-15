from __future__ import annotations

from pathlib import Path

import pandas as pd

from whsdsci.build_long import build_canonical_long


def test_long_table_two_rows_per_retained_original(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "game_id": ["game_1", "game_2"],
            "record_id": ["record_1", "record_2"],
            "home_team": ["A", "C"],
            "away_team": ["B", "D"],
            "home_off_line": ["first_off", "first_off"],
            "away_off_line": ["second_off", "second_off"],
            "home_def_pairing": ["first_def", "first_def"],
            "away_def_pairing": ["second_def", "second_def"],
            "toi": [120.0, 0.0],
            "home_xg": [0.2, 0.1],
            "away_xg": [0.1, 0.2],
            "home_shots": [2, 1],
            "away_shots": [1, 2],
            "home_goals": [0, 0],
            "away_goals": [0, 1],
        }
    )

    long_df, ev_df, _ = build_canonical_long(raw, outputs_dir=tmp_path)

    # Only first source row retained (toi > 0), and each retained row becomes two long rows.
    assert len(long_df) == 2

    required_cols = {
        "offense_team",
        "defense_team",
        "off_line",
        "def_pair",
        "xg_for",
        "toi_sec",
        "toi_min",
        "toi_hr",
        "log_toi_hr",
        "xg_rate_hr",
        "xg_diff",
        "xg_diff_rate_hr",
        "off_unit",
        "def_unit",
        "is_ev",
        "state",
        "is_pp",
        "is_empty_net",
        "game_num",
    }
    assert required_cols.issubset(set(long_df.columns))

    assert ev_df["is_ev"].all()
    assert set(ev_df["off_line"].unique()).issubset({"first_off", "second_off"})
    assert set(ev_df["def_pair"].unique()).issubset({"first_def", "second_def"})
