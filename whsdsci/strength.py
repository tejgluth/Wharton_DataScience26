from __future__ import annotations

import numpy as np
import pandas as pd


def _split_off_unit(off_unit: str) -> tuple[str, str]:
    if "__" not in off_unit:
        return off_unit, ""
    team, line = off_unit.split("__", 1)
    return team, line


def compute_standardized_strengths(
    model,
    train_ev_df: pd.DataFrame,
    off_units: list[str] | None = None,
) -> pd.DataFrame:
    ref = (
        train_ev_df.groupby("def_unit", as_index=False)["toi_hr"]
        .sum()
        .rename(columns={"toi_hr": "w"})
    )
    if ref.empty:
        return pd.DataFrame(columns=["off_unit", "team", "off_line", "strength_rate_hr", "strength_xg60"])

    ref["w"] = ref["w"] / ref["w"].sum()
    def_units = ref["def_unit"].astype(str).tolist()
    weights = ref["w"].to_numpy(dtype=float)

    if off_units is None:
        off_units = sorted(train_ev_df["off_unit"].astype(str).unique())

    n_off = len(off_units)
    n_def = len(def_units)
    off_rep = np.repeat(np.asarray(off_units), n_def)
    def_rep = np.tile(np.asarray(def_units), n_off)

    grid_home0 = pd.DataFrame(
        {
            "off_unit": off_rep,
            "def_unit": def_rep,
            "is_home": np.zeros(n_off * n_def, dtype=int),
            "toi_hr": np.ones(n_off * n_def, dtype=float),
            "game_id": ["synthetic"] * (n_off * n_def),
        }
    )
    grid_home1 = grid_home0.copy()
    grid_home1["is_home"] = 1

    r0 = np.asarray(model.predict_rate_hr(grid_home0), dtype=float).reshape(n_off, n_def)
    r1 = np.asarray(model.predict_rate_hr(grid_home1), dtype=float).reshape(n_off, n_def)
    r_avg = 0.5 * (r0 + r1)
    strengths = r_avg @ weights

    rows = []
    for i, off in enumerate(off_units):
        team, line = _split_off_unit(off)
        strength_rate_hr = float(strengths[i])
        rows.append(
            {
                "off_unit": off,
                "team": team,
                "off_line": line,
                "strength_rate_hr": strength_rate_hr,
                "strength_xg60": strength_rate_hr / 60.0,
            }
        )

    return pd.DataFrame(rows)


def compute_disparity_ratios(strength_df: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    if strength_df.empty:
        return pd.DataFrame(columns=["team", "line1_strength_xg60", "line2_strength_xg60", "ratio"])

    l1 = strength_df[strength_df["off_line"] == "first_off"][
        ["team", "strength_xg60"]
    ].rename(columns={"strength_xg60": "line1_strength_xg60"})
    l2 = strength_df[strength_df["off_line"] == "second_off"][
        ["team", "strength_xg60"]
    ].rename(columns={"strength_xg60": "line2_strength_xg60"})

    merged = l1.merge(l2, on="team", how="inner")
    merged["ratio"] = (merged["line1_strength_xg60"] + eps) / (merged["line2_strength_xg60"] + eps)
    merged = merged.replace([np.inf, -np.inf], np.nan).dropna(subset=["ratio"])
    merged = merged.sort_values("ratio", ascending=False).reset_index(drop=True)
    merged["rank"] = np.arange(1, len(merged) + 1)
    return merged[["rank", "team", "line1_strength_xg60", "line2_strength_xg60", "ratio"]]
