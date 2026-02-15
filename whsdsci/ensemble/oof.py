from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

EPS_TOTAL = 1e-9


def clip_total_positive(arr: np.ndarray) -> np.ndarray:
    out = np.asarray(arr, dtype=float)
    out = np.nan_to_num(out, nan=EPS_TOTAL, posinf=1e9, neginf=EPS_TOTAL)
    out = np.clip(out, EPS_TOTAL, None)
    return out


def dataset_key(df: pd.DataFrame) -> str:
    games = sorted(df["game_id"].astype(str).unique().tolist())
    payload = "|".join(games).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def _resolve_splits(df: pd.DataFrame, n_splits: int) -> int:
    n_games = int(df["game_id"].astype(str).nunique())
    return int(max(2, min(n_splits, n_games)))


def build_oof_predictions(
    df: pd.DataFrame,
    model_builders: dict[str, Callable[[], object]],
    model_names: list[str],
    n_splits: int = 5,
) -> pd.DataFrame:
    if not model_names:
        raise ValueError("No model names provided for OOF prediction building")

    rows = np.arange(len(df))
    groups = df["game_id"].astype(str).to_numpy()
    use_splits = _resolve_splits(df, n_splits)
    splitter = GroupKFold(n_splits=use_splits)

    pred = np.full((len(df), len(model_names)), np.nan, dtype=float)

    for tr, te in splitter.split(rows, groups=groups):
        dtr = df.iloc[tr].reset_index(drop=True)
        dte = df.iloc[te].reset_index(drop=True)

        for j, method in enumerate(model_names):
            if method not in model_builders:
                raise KeyError(f"OOF builder missing for method: {method}")
            model = model_builders[method]()
            try:
                model.fit(dtr)
                mu = model.predict_total(dte)
            except SkipModelError:
                raise
            except Exception as exc:
                raise RuntimeError(f"OOF fit/predict failed for {method}: {exc}") from exc
            pred[te, j] = clip_total_positive(mu)

    if np.isnan(pred).any():
        missing = int(np.isnan(pred).sum())
        raise RuntimeError(f"OOF predictions contain missing values: {missing}")

    out = pd.DataFrame(
        {
            "row_id": df.index.to_numpy(),
            "game_id": df["game_id"].astype(str).to_numpy(),
            "toi_hr": np.maximum(pd.to_numeric(df["toi_hr"], errors="coerce").to_numpy(dtype=float), 1e-9),
            "y_true_total": np.clip(pd.to_numeric(df["xg_for"], errors="coerce").to_numpy(dtype=float), 0, None),
        }
    )

    for j, method in enumerate(model_names):
        out[f"mu_pred_total_{method}"] = clip_total_positive(pred[:, j])

    return out


def select_diverse_models(
    oof_df: pd.DataFrame,
    ranked_model_names: list[str],
    max_models: int = 6,
    corr_threshold: float = 0.995,
    required_models: list[str] | None = None,
) -> list[str]:
    required = required_models or []
    pred_cols = {m: f"mu_pred_total_{m}" for m in ranked_model_names if f"mu_pred_total_{m}" in oof_df.columns}

    selected: list[str] = []

    for req in required:
        if req in pred_cols and req not in selected:
            selected.append(req)

    for method in ranked_model_names:
        if method not in pred_cols:
            continue
        if method in selected:
            continue
        if len(selected) >= max_models:
            break

        keep = True
        v = oof_df[pred_cols[method]].to_numpy(dtype=float)
        for s in selected:
            vs = oof_df[pred_cols[s]].to_numpy(dtype=float)
            if np.std(v) <= 0 or np.std(vs) <= 0:
                corr = 1.0
            else:
                corr = float(np.corrcoef(v, vs)[0, 1])
            if np.isfinite(corr) and abs(corr) > corr_threshold:
                keep = False
                break
        if keep:
            selected.append(method)

    for req in required:
        if req in pred_cols and req not in selected:
            selected.insert(0, req)

    return selected[:max_models]
