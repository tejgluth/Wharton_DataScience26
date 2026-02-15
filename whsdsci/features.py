from __future__ import annotations

import numpy as np
import pandas as pd


def filter_ev(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["is_ev"]].copy()


def ensure_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["off_unit", "def_unit"]:
        out[col] = out[col].astype(str)
    out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").fillna(0).astype(int)
    out["toi_hr"] = np.maximum(pd.to_numeric(out["toi_hr"], errors="coerce"), 1e-9)
    out["xg_for"] = np.clip(pd.to_numeric(out["xg_for"], errors="coerce"), 0, None)
    out["xg_rate_hr"] = out["xg_for"] / out["toi_hr"]
    out["game_id"] = out["game_id"].astype(str)
    return out


def has_time_split(df: pd.DataFrame) -> bool:
    vals = pd.to_numeric(df.get("game_num"), errors="coerce")
    return vals.notna().sum() > 0


def build_time_split_mask(df: pd.DataFrame, train_frac: float = 0.7) -> tuple[np.ndarray, np.ndarray] | None:
    nums = pd.to_numeric(df.get("game_num"), errors="coerce")
    temp = pd.DataFrame({"game_id": df["game_id"].astype(str), "game_num": nums})
    game_order = temp.dropna().drop_duplicates("game_id").sort_values("game_num")
    if game_order.empty:
        return None
    n_train = max(1, int(len(game_order) * train_frac))
    train_games = set(game_order.iloc[:n_train]["game_id"])
    test_games = set(game_order.iloc[n_train:]["game_id"])
    if not test_games:
        return None
    train_mask = df["game_id"].astype(str).isin(train_games).to_numpy()
    test_mask = df["game_id"].astype(str).isin(test_games).to_numpy()
    return train_mask, test_mask
