from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from whsdsci.ensemble.search import Config, FittedSearchModel
from whsdsci.eval.cv import make_group_kfold_splits
from whsdsci.eval.metrics import calibration_ratio, mae_total, poisson_deviance_safe, weighted_mse_rate
from whsdsci.models import get_model_builders
from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


EPS = 1e-9


@dataclass
class OlqdAblationResult:
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    delta_summary: pd.DataFrame
    oof_predictions: pd.DataFrame
    team_olqd_full: pd.DataFrame


def _infer_team_from_unit(series: pd.Series) -> pd.Series:
    return series.astype(str).str.split("__", n=1).str[0]


def _get_offense_team(df: pd.DataFrame) -> pd.Series:
    if "offense_team" in df.columns:
        return df["offense_team"].astype(str)
    if "off_unit" in df.columns:
        return _infer_team_from_unit(df["off_unit"])
    return pd.Series("", index=df.index, dtype=object)


def _get_defense_team(df: pd.DataFrame) -> pd.Series:
    if "defense_team" in df.columns:
        return df["defense_team"].astype(str)
    if "def_unit" in df.columns:
        return _infer_team_from_unit(df["def_unit"])
    return pd.Series("", index=df.index, dtype=object)


def compute_team_olqd_table(model, train_ev_df: pd.DataFrame) -> pd.DataFrame:
    strengths = compute_standardized_strengths(model=model, train_ev_df=train_ev_df)
    ratios = compute_disparity_ratios(strengths)
    if strengths.empty or ratios.empty:
        return pd.DataFrame(
            columns=[
                "team",
                "line1_strength_xg60",
                "line2_strength_xg60",
                "olqd_ratio",
                "olqd_log_ratio",
                "olqd_gap_xg60",
            ]
        )
    out = ratios.rename(columns={"ratio": "olqd_ratio"}).copy()
    out["olqd_log_ratio"] = np.log(np.clip(pd.to_numeric(out["olqd_ratio"], errors="coerce"), 1e-12, None))
    out["olqd_gap_xg60"] = (
        pd.to_numeric(out["line1_strength_xg60"], errors="coerce").fillna(0.0)
        - pd.to_numeric(out["line2_strength_xg60"], errors="coerce").fillna(0.0)
    )
    return out[
        [
            "team",
            "line1_strength_xg60",
            "line2_strength_xg60",
            "olqd_ratio",
            "olqd_log_ratio",
            "olqd_gap_xg60",
        ]
    ].copy()


def make_olqd_feature_frame(
    df: pd.DataFrame,
    mu_base_total: np.ndarray,
    team_olqd: pd.DataFrame,
) -> pd.DataFrame:
    off_team = _get_offense_team(df)
    def_team = _get_defense_team(df)
    off_line = df["off_line"].astype(str) if "off_line" in df.columns else pd.Series("", index=df.index, dtype=object)

    olqd = team_olqd.set_index("team") if not team_olqd.empty else pd.DataFrame()
    off_ratio = off_team.map(olqd["olqd_ratio"] if "olqd_ratio" in olqd.columns else pd.Series(dtype=float))
    def_ratio = def_team.map(olqd["olqd_ratio"] if "olqd_ratio" in olqd.columns else pd.Series(dtype=float))
    off_line1 = off_team.map(olqd["line1_strength_xg60"] if "line1_strength_xg60" in olqd.columns else pd.Series(dtype=float))
    off_line2 = off_team.map(olqd["line2_strength_xg60"] if "line2_strength_xg60" in olqd.columns else pd.Series(dtype=float))
    off_gap = off_team.map(olqd["olqd_gap_xg60"] if "olqd_gap_xg60" in olqd.columns else pd.Series(dtype=float))

    off_line1_arr = pd.to_numeric(off_line1, errors="coerce").to_numpy(dtype=float)
    off_line2_arr = pd.to_numeric(off_line2, errors="coerce").to_numpy(dtype=float)
    off_current = np.where(off_line.to_numpy() == "first_off", off_line1_arr, off_line2_arr)
    off_avg = 0.5 * (
        off_line1_arr + off_line2_arr
    )
    off_current = np.where(np.isnan(off_current), off_avg, off_current)

    x = pd.DataFrame(
        {
            "log_mu_base": np.log(np.clip(np.asarray(mu_base_total, dtype=float), EPS, None)),
            "log_toi_hr": pd.to_numeric(df.get("log_toi_hr"), errors="coerce").fillna(0.0),
            "is_home": pd.to_numeric(df.get("is_home"), errors="coerce").fillna(0.0),
            "off_olqd_ratio": pd.to_numeric(off_ratio, errors="coerce"),
            "def_olqd_ratio": pd.to_numeric(def_ratio, errors="coerce"),
            "olqd_ratio_diff": pd.to_numeric(off_ratio, errors="coerce") - pd.to_numeric(def_ratio, errors="coerce"),
            "off_line_strength_xg60": off_current,
            "off_olqd_gap_xg60": pd.to_numeric(off_gap, errors="coerce"),
        },
        index=df.index,
    )
    for c in x.columns:
        med = float(pd.to_numeric(x[c], errors="coerce").median()) if x[c].notna().any() else 0.0
        x[c] = pd.to_numeric(x[c], errors="coerce").fillna(med)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return x


def _select_alpha(
    x_train: np.ndarray,
    y_train: np.ndarray,
    groups: np.ndarray,
    alphas: list[float] | None = None,
) -> float:
    alpha_grid = alphas or [1e-6, 1e-4, 1e-3, 1e-2]
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return float(alpha_grid[0])
    n_splits = min(3, len(uniq))
    if n_splits < 2:
        return float(alpha_grid[0])
    from sklearn.model_selection import GroupKFold

    gkf = GroupKFold(n_splits=n_splits)
    best_alpha = float(alpha_grid[0])
    best_score = float("inf")
    for alpha in alpha_grid:
        scores = []
        for tr, va in gkf.split(np.arange(len(y_train)), groups=groups):
            m = PoissonRegressor(alpha=float(alpha), fit_intercept=True, max_iter=1000)
            m.fit(x_train[tr], y_train[tr])
            pred = np.clip(m.predict(x_train[va]), EPS, None)
            scores.append(poisson_deviance_safe(y_train[va], pred))
        score = float(np.mean(scores))
        if score < best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha


def _fold_predictions(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cfg: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    pred_train_base, pred_test_base, pred_test_olqd, team_olqd = fit_predict_baseline_plus_olqd(
        train_df=train_df,
        infer_df=test_df,
        cfg=cfg,
        seed=seed,
    )
    _ = pred_train_base
    return pred_test_base, pred_test_olqd, team_olqd


def fit_predict_baseline_plus_olqd(
    train_df: pd.DataFrame,
    infer_df: pd.DataFrame,
    cfg: Config,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Fit baseline + OLQD meta-correction on train_df, predict on infer_df."""
    builders = get_model_builders(random_state=seed)
    baseline_model = FittedSearchModel(random_state=seed, base_builders=builders, config=cfg)
    baseline_model.fit(train_df)

    pred_train_base = np.clip(baseline_model.predict_total(train_df), EPS, None)
    pred_infer_base = np.clip(baseline_model.predict_total(infer_df), EPS, None)
    team_olqd = compute_team_olqd_table(model=baseline_model, train_ev_df=train_df)

    x_train = make_olqd_feature_frame(train_df, pred_train_base, team_olqd)
    x_infer = make_olqd_feature_frame(infer_df, pred_infer_base, team_olqd)
    y_train = np.clip(pd.to_numeric(train_df["xg_for"], errors="coerce").to_numpy(dtype=float), 0, None)
    groups = train_df["game_id"].astype(str).to_numpy()
    alpha = _select_alpha(x_train.to_numpy(dtype=float), y_train, groups=groups)
    meta = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=1000)
    meta.fit(x_train.to_numpy(dtype=float), y_train)
    pred_infer_olqd = np.clip(meta.predict(x_infer.to_numpy(dtype=float)), EPS, None)
    return pred_train_base, pred_infer_base, pred_infer_olqd, team_olqd


def _metric_row(method: str, fold_id: str, df: pd.DataFrame, pred_total: np.ndarray) -> dict[str, Any]:
    y = np.clip(pd.to_numeric(df["xg_for"], errors="coerce").to_numpy(dtype=float), 0, None)
    toi = np.maximum(pd.to_numeric(df["toi_hr"], errors="coerce").to_numpy(dtype=float), EPS)
    pred = np.clip(np.asarray(pred_total, dtype=float), EPS, None)
    return {
        "method": method,
        "fold_id": fold_id,
        "poisson_deviance": float(poisson_deviance_safe(y, pred)),
        "weighted_mse_rate": float(weighted_mse_rate(y / toi, pred / toi, toi)),
        "mae_total": float(mae_total(y, pred)),
        "calibration_ratio": float(calibration_ratio(y, pred)),
        "n_test": int(len(df)),
    }


def evaluate_baseline_vs_olqd(
    ev_df: pd.DataFrame,
    cfg: Config,
    seed: int = 1,
    n_splits: int = 5,
) -> OlqdAblationResult:
    work = ev_df.reset_index(drop=False).rename(columns={"index": "row_id"}).copy()
    splits = make_group_kfold_splits(work, n_splits=n_splits)

    fold_rows: list[dict[str, Any]] = []
    oof_rows: list[pd.DataFrame] = []
    last_team_olqd = pd.DataFrame()
    for split in splits:
        tr = work.iloc[split.train_idx].reset_index(drop=True)
        te = work.iloc[split.test_idx].reset_index(drop=True)

        pred_base, pred_olqd, team_olqd = _fold_predictions(train_df=tr, test_df=te, cfg=cfg, seed=seed)
        last_team_olqd = team_olqd
        fold_rows.append(_metric_row("baseline", split.fold_id, te, pred_base))
        fold_rows.append(_metric_row("baseline+olqd", split.fold_id, te, pred_olqd))

        oof_rows.append(
            pd.DataFrame(
                {
                    "row_id": te["row_id"].to_numpy(),
                    "game_id": te["game_id"].astype(str).to_numpy(),
                    "y_true_total": np.clip(pd.to_numeric(te["xg_for"], errors="coerce").to_numpy(dtype=float), 0, None),
                    "toi_hr": np.maximum(pd.to_numeric(te["toi_hr"], errors="coerce").to_numpy(dtype=float), EPS),
                    "pred_baseline_total": pred_base,
                    "pred_baseline_olqd_total": pred_olqd,
                }
            )
        )

    fold_metrics = pd.DataFrame(fold_rows)
    summary = (
        fold_metrics.groupby("method", as_index=False)
        .agg(
            mean_poisson_deviance=("poisson_deviance", "mean"),
            std_poisson_deviance=("poisson_deviance", "std"),
            mean_weighted_mse_rate=("weighted_mse_rate", "mean"),
            mean_mae_total=("mae_total", "mean"),
            mean_calibration_ratio=("calibration_ratio", "mean"),
        )
        .sort_values("mean_poisson_deviance")
        .reset_index(drop=True)
    )

    piv = fold_metrics.pivot(index="fold_id", columns="method", values="poisson_deviance").dropna()
    if {"baseline", "baseline+olqd"}.issubset(set(piv.columns)):
        delta = piv["baseline+olqd"] - piv["baseline"]
        n = int(len(delta))
        mean_delta = float(delta.mean())
        sd_delta = float(delta.std(ddof=1)) if n > 1 else 0.0
        se = sd_delta / np.sqrt(n) if n > 0 else np.nan
        ci95 = 1.96 * se if np.isfinite(se) else np.nan
    else:
        n = 0
        mean_delta = np.nan
        sd_delta = np.nan
        ci95 = np.nan
    delta_summary = pd.DataFrame(
        [
            {
                "metric": "poisson_deviance_delta_baseline_plus_olqd_minus_baseline",
                "n_folds": n,
                "mean_delta": mean_delta,
                "std_delta": sd_delta,
                "ci95_half_width": ci95,
            }
        ]
    )

    oof = pd.concat(oof_rows, ignore_index=True) if oof_rows else pd.DataFrame()
    return OlqdAblationResult(
        fold_metrics=fold_metrics,
        summary=summary,
        delta_summary=delta_summary,
        oof_predictions=oof,
        team_olqd_full=last_team_olqd,
    )
