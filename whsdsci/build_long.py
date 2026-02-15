from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)

REQUIRED_MIN_COLS = [
    "game_id",
    "record_id",
    "home_team",
    "away_team",
    "home_off_line",
    "away_off_line",
    "home_def_pairing",
    "away_def_pairing",
    "toi",
    "home_xg",
    "away_xg",
    "home_shots",
    "away_shots",
    "home_goals",
    "away_goals",
]


CLIP_NONNEG_COLS = [
    "home_xg",
    "away_xg",
    "home_shots",
    "away_shots",
    "home_goals",
    "away_goals",
    "toi",
]


def _ensure_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out


def _clip_nonnegative(df: pd.DataFrame, cols: list[str]) -> dict[str, int]:
    clip_counts: dict[str, int] = {}
    for c in cols:
        if c not in df.columns:
            continue
        vals = pd.to_numeric(df[c], errors="coerce")
        n_neg = int((vals < 0).sum())
        if n_neg > 0:
            LOGGER.warning("Clipping %s negatives to zero in column %s", n_neg, c)
        df[c] = np.clip(vals, 0, None)
        clip_counts[c] = n_neg
    return clip_counts


def _parse_game_num(game_id: Any) -> float:
    if pd.isna(game_id):
        return np.nan
    m = re.match(r"^game_(\d+)$", str(game_id))
    if not m:
        return np.nan
    return float(m.group(1))


def build_canonical_long(
    raw_df: pd.DataFrame,
    outputs_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    df = _ensure_columns(raw_df, REQUIRED_MIN_COLS)
    clip_counts = _clip_nonnegative(df, CLIP_NONNEG_COLS)

    home = df.copy()
    home["offense_team"] = df["home_team"]
    home["defense_team"] = df["away_team"]
    home["off_line"] = df["home_off_line"]
    home["def_pair"] = df["away_def_pairing"]
    home["xg_for"] = df["home_xg"]
    home["shots_for"] = df["home_shots"]
    home["goals_for"] = df["home_goals"]
    home["xg_against"] = df["away_xg"]
    home["shots_against"] = df["away_shots"]
    home["goals_against"] = df["away_goals"]
    home["toi_sec"] = df["toi"]
    home["is_home"] = 1

    away = df.copy()
    away["offense_team"] = df["away_team"]
    away["defense_team"] = df["home_team"]
    away["off_line"] = df["away_off_line"]
    away["def_pair"] = df["home_def_pairing"]
    away["xg_for"] = df["away_xg"]
    away["shots_for"] = df["away_shots"]
    away["goals_for"] = df["away_goals"]
    away["xg_against"] = df["home_xg"]
    away["shots_against"] = df["home_shots"]
    away["goals_against"] = df["home_goals"]
    away["toi_sec"] = df["toi"]
    away["is_home"] = 0

    long_df = pd.concat([home, away], ignore_index=True)

    before_drop = len(long_df)
    long_df = long_df[pd.to_numeric(long_df["toi_sec"], errors="coerce") > 0].copy()
    drop_toi = before_drop - len(long_df)

    required_key_cols = ["offense_team", "defense_team", "off_line", "def_pair"]
    before_drop2 = len(long_df)
    long_df = long_df.dropna(subset=required_key_cols).copy()
    drop_missing = before_drop2 - len(long_df)

    for c in [
        "xg_for",
        "shots_for",
        "goals_for",
        "xg_against",
        "shots_against",
        "goals_against",
        "toi_sec",
    ]:
        long_df[c] = np.clip(pd.to_numeric(long_df[c], errors="coerce"), 0, None)

    toi_hr_safe = np.maximum(long_df["toi_sec"].to_numpy(dtype=float) / 3600.0, 1e-9)
    long_df["toi_min"] = long_df["toi_sec"] / 60.0
    long_df["toi_hr"] = long_df["toi_sec"] / 3600.0
    long_df["log_toi_hr"] = np.log(toi_hr_safe)
    long_df["xg_rate_hr"] = long_df["xg_for"] / toi_hr_safe
    long_df["xg_diff"] = long_df["xg_for"] - long_df["xg_against"]
    long_df["xg_diff_rate_hr"] = long_df["xg_diff"] / toi_hr_safe
    long_df["off_unit"] = long_df["offense_team"].astype(str) + "__" + long_df["off_line"].astype(str)
    long_df["def_unit"] = long_df["defense_team"].astype(str) + "__" + long_df["def_pair"].astype(str)

    ev_off = {"first_off", "second_off"}
    ev_def = {"first_def", "second_def"}
    long_df["is_ev"] = long_df["off_line"].isin(ev_off) & long_df["def_pair"].isin(ev_def)
    long_df["state"] = np.where(long_df["is_ev"], "EV", "NON_EV")
    long_df["is_pp"] = long_df["off_line"].isin({"PP_up", "PP_kill_dwn"}) | long_df["def_pair"].isin(
        {"PP_up", "PP_kill_dwn"}
    )
    long_df["is_empty_net"] = (long_df["off_line"] == "empty_net_line") | (
        long_df["def_pair"] == "empty_net_line"
    )

    long_df["game_num"] = long_df["game_id"].map(_parse_game_num)

    ev_df = long_df[long_df["is_ev"]].copy()

    off_counts = long_df["off_line"].value_counts(dropna=False).to_dict()
    def_counts = long_df["def_pair"].value_counts(dropna=False).to_dict()

    toi_stats = long_df["toi_sec"].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).to_dict()
    xg_zero_frac = float((long_df["xg_for"] <= 0).mean()) if len(long_df) else float("nan")

    teams = sorted(ev_df["offense_team"].astype(str).unique())
    first_teams = set(ev_df.loc[ev_df["off_line"] == "first_off", "offense_team"].astype(str).unique())
    second_teams = set(ev_df.loc[ev_df["off_line"] == "second_off", "offense_team"].astype(str).unique())
    missing_line_teams = sorted([t for t in teams if (t not in first_teams or t not in second_teams)])

    profile: dict[str, Any] = {
        "rows_raw": int(len(raw_df)),
        "rows_long": int(len(long_df)),
        "rows_ev": int(len(ev_df)),
        "rows_non_ev": int((~long_df["is_ev"]).sum()),
        "dropped_toi_nonpositive": int(drop_toi),
        "dropped_missing_keys": int(drop_missing),
        "clip_negative_counts": clip_counts,
        "off_line_counts": off_counts,
        "def_pair_counts": def_counts,
        "toi_sec_summary": {k: float(v) for k, v in toi_stats.items()},
        "xg_zeros_fraction": xg_zero_frac,
        "unique_off_unit": int(long_df["off_unit"].nunique()),
        "unique_def_unit": int(long_df["def_unit"].nunique()),
        "teams_missing_first_or_second_off_in_ev": missing_line_teams,
    }

    outputs_dir.mkdir(parents=True, exist_ok=True)
    long_df.to_parquet(outputs_dir / "whl_long.parquet", index=False)
    ev_df.to_parquet(outputs_dir / "whl_long_ev.parquet", index=False)
    with (outputs_dir / "data_profile.json").open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    return long_df, ev_df, profile
