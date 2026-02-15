from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold


@dataclass
class Split:
    split_type: str
    fold_id: str
    train_idx: np.ndarray
    test_idx: np.ndarray


def make_group_kfold_splits(df: pd.DataFrame, n_splits: int = 5) -> list[Split]:
    groups = df["game_id"].astype(str).to_numpy()
    unique_games = np.unique(groups)
    use_splits = int(min(n_splits, len(unique_games)))
    if use_splits < 2:
        raise ValueError("Need at least 2 unique games for GroupKFold")
    gkf = GroupKFold(n_splits=use_splits)
    rows = np.arange(len(df))
    out: list[Split] = []
    for i, (tr, te) in enumerate(gkf.split(rows, groups=groups), start=1):
        out.append(Split(split_type="groupkfold", fold_id=f"fold_{i}", train_idx=tr, test_idx=te))
    return out


def make_time_split(df: pd.DataFrame, train_frac: float = 0.7) -> Split | None:
    if "game_num" not in df.columns:
        return None
    temp = (
        df[["game_id", "game_num"]]
        .assign(game_id=lambda x: x["game_id"].astype(str), game_num=lambda x: pd.to_numeric(x["game_num"], errors="coerce"))
        .dropna(subset=["game_num"]) 
        .drop_duplicates(subset=["game_id"]) 
        .sort_values("game_num")
    )
    if temp.empty:
        return None
    n_train = max(1, int(len(temp) * train_frac))
    if n_train >= len(temp):
        return None
    train_games = set(temp.iloc[:n_train]["game_id"])
    test_games = set(temp.iloc[n_train:]["game_id"])
    game_ids = df["game_id"].astype(str)
    train_idx = np.where(game_ids.isin(train_games).to_numpy())[0]
    test_idx = np.where(game_ids.isin(test_games).to_numpy())[0]
    if len(test_idx) == 0:
        return None
    return Split(split_type="time_split", fold_id="time_70_30", train_idx=train_idx, test_idx=test_idx)
