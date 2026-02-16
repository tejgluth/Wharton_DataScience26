from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_extraction import FeatureHasher

from whsdsci.debug.debug_disparity import (
    Phase1bValidityResult,
    check_phase1b_validity_tables,
    debug_best_disparity,
)
from whsdsci.ensemble.calibration import fit_calibrator
from whsdsci.ensemble.oof import clip_total_positive
from whsdsci.ensemble.search import _build_cache_for_models
from whsdsci.eval.bootstrap import bootstrap_rank_stability
from whsdsci.eval.metrics import calibration_ratio, poisson_deviance_safe, weighted_mse_rate
from whsdsci.models.base import BaseModel
from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


LOGGER = logging.getLogger(__name__)
EPS = 1e-9


@dataclass
class TreePoissonTrialConfig:
    trial_id: str
    stage: str
    base_pool_id: str
    base_models: list[str]
    feature_set: str
    calibration_type: str
    cal_bins: int
    hyperparams: dict[str, Any]
    seed_bag: int
    random_seed: int


def _parse_team(values: pd.Series, unit_values: pd.Series, side: str) -> np.ndarray:
    if side in values.index:
        out = values[side].astype(str)
    else:
        out = unit_values.astype(str).str.split("__", n=1).str[0]
    return out.to_numpy(dtype=object)


def _safe_series(df: pd.DataFrame, col: str, fill: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(fill)
    return pd.Series(fill, index=df.index, dtype=float)


def _safe_label_series(df: pd.DataFrame, col: str, fill: str = "") -> pd.Series:
    if col in df.columns:
        return df[col].astype(str).fillna(fill)
    return pd.Series(fill, index=df.index, dtype=object)


def _context_from_df(df: pd.DataFrame) -> dict[str, np.ndarray]:
    toi_hr = np.maximum(_safe_series(df, "toi_hr", 1.0).to_numpy(dtype=float), EPS)
    log_toi_hr = np.log(toi_hr)
    is_home = _safe_series(df, "is_home", 0.0).to_numpy(dtype=float)

    off_unit = _safe_label_series(df, "off_unit", "")
    def_unit = _safe_label_series(df, "def_unit", "")
    off_line = _safe_label_series(df, "off_line", "")
    def_pair = _safe_label_series(df, "def_pair", "")
    if (off_line == "").all():
        off_line = off_unit.str.split("__", n=1).str[-1].fillna("")
    if (def_pair == "").all():
        def_pair = def_unit.str.split("__", n=1).str[-1].fillna("")
    off_is_first = (off_line == "first_off").astype(float).to_numpy(dtype=float)
    def_is_first = (def_pair == "first_def").astype(float).to_numpy(dtype=float)

    offense_team = _safe_label_series(df, "offense_team", "")
    defense_team = _safe_label_series(df, "defense_team", "")
    if (offense_team == "").all():
        offense_team = off_unit.str.split("__", n=1).str[0].fillna("")
    if (defense_team == "").all():
        defense_team = def_unit.str.split("__", n=1).str[0].fillna("")

    return {
        "toi_hr": toi_hr,
        "log_toi_hr": log_toi_hr,
        "is_home": is_home,
        "off_is_first": off_is_first,
        "def_is_first": def_is_first,
        "offense_team": offense_team.to_numpy(dtype=object),
        "defense_team": defense_team.to_numpy(dtype=object),
    }


def _hash_tokens(tokens: list[list[str]], n_features: int) -> np.ndarray:
    fh = FeatureHasher(n_features=n_features, input_type="string", alternate_sign=False)
    mat = fh.transform(tokens)
    return np.asarray(mat.toarray(), dtype=float)


def _build_meta_features(
    P: np.ndarray,
    df: pd.DataFrame,
    feature_set: str,
    model_names: list[str],
    ref_model: str = "POISSON_GLM_OFFSET_REG",
) -> np.ndarray:
    mu = np.clip(np.asarray(P, dtype=float), EPS, None)
    log_mu = np.log(mu)
    ctx = _context_from_df(df)
    m = mu.shape[1]

    if ref_model in model_names:
        ref_idx = model_names.index(ref_model)
    else:
        ref_idx = 0
    mu_ref = np.clip(mu[:, ref_idx], EPS, None)
    log_ref = np.log(mu_ref)

    if feature_set == "F0_BASE_ONLY":
        X = mu
    elif feature_set == "F1_LOG_BASE":
        X = log_mu
    elif feature_set == "F2_MIXED":
        X = np.column_stack([mu, log_mu])
    elif feature_set == "F3_DELTAS":
        X = np.column_stack([mu - mu_ref[:, None], log_mu - log_ref[:, None]])
    elif feature_set == "F4_RATIOS":
        X = np.column_stack([mu / mu_ref[:, None], log_mu - log_ref[:, None]])
    elif feature_set == "F5_CONTEXT_MIN":
        X = np.column_stack([mu, log_mu, ctx["log_toi_hr"], ctx["is_home"]])
    elif feature_set == "F6_CONTEXT_EV_STATE":
        X = np.column_stack(
            [
                mu,
                log_mu,
                ctx["log_toi_hr"],
                ctx["is_home"],
                ctx["off_is_first"],
                ctx["def_is_first"],
            ]
        )
    elif feature_set == "F7_TEAM_HASH_CONTEXT":
        base = np.column_stack(
            [
                mu,
                log_mu,
                ctx["log_toi_hr"],
                ctx["is_home"],
                ctx["off_is_first"],
                ctx["def_is_first"],
            ]
        )
        off_tokens = [[f"off={x}"] for x in ctx["offense_team"]]
        def_tokens = [[f"def={x}"] for x in ctx["defense_team"]]
        Xh_off = _hash_tokens(off_tokens, n_features=16)
        Xh_def = _hash_tokens(def_tokens, n_features=16)
        X = np.column_stack([base, Xh_off, Xh_def])
    elif feature_set == "F8_INTERACTIONS_LIGHT":
        base = np.column_stack(
            [
                mu,
                log_mu,
                ctx["log_toi_hr"],
                ctx["is_home"],
                ctx["off_is_first"],
                ctx["def_is_first"],
            ]
        )
        var_idx = np.argsort(-np.var(log_mu, axis=0))[: min(3, m)]
        ints = []
        for j in var_idx:
            ints.append(ctx["log_toi_hr"] * log_mu[:, j])
            ints.append(ctx["is_home"] * log_mu[:, j])
        X = np.column_stack([base, *ints]) if ints else base
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=50.0, neginf=-50.0)
    return X


def _try_xgboost():
    try:
        from xgboost import XGBRegressor

        return XGBRegressor
    except Exception:
        return None


def _fit_single_meta_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    hp: dict[str, Any],
    seed: int,
):
    XGBRegressor = _try_xgboost()
    max_depth = int(hp.get("max_depth", 3))
    learning_rate = float(hp.get("learning_rate", 0.08))
    n_estimators = int(hp.get("n_estimators", 200))
    min_child_weight = float(hp.get("min_child_weight", 1.0))
    subsample = float(hp.get("subsample", 0.9))
    colsample_bytree = float(hp.get("colsample_bytree", 0.9))
    gamma = float(hp.get("gamma", 0.0))
    reg_lambda = float(hp.get("reg_lambda", 1.0))
    reg_alpha = float(hp.get("reg_alpha", 0.0))
    max_delta_step = float(hp.get("max_delta_step", 0.7))

    if XGBRegressor is not None:
        model = XGBRegressor(
            objective="count:poisson",
            tree_method=str(hp.get("tree_method", "hist")),
            max_depth=max_depth,
            learning_rate=learning_rate,
            n_estimators=n_estimators,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            gamma=gamma,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            max_delta_step=max_delta_step,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(X_train, y_train)
        backend = "xgboost"
    else:
        model = HistGradientBoostingRegressor(
            loss="poisson",
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=n_estimators,
            min_samples_leaf=max(1, int(np.ceil(min_child_weight))),
            l2_regularization=reg_lambda,
            random_state=seed,
        )
        model.fit(X_train, y_train)
        backend = "sklearn_histgbr"
    return model, backend


def _seed_list(seed_bag: int, base_seed: int) -> list[int]:
    k = max(1, int(seed_bag))
    return [int(base_seed + i) for i in range(k)]


def _fit_predict_bag(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_pred: np.ndarray,
    hp: dict[str, Any],
    seed_bag: int,
    base_seed: int,
):
    seeds = _seed_list(seed_bag=seed_bag, base_seed=base_seed)
    preds_train = []
    preds_pred = []
    backend = "unknown"
    for seed in seeds:
        model, backend = _fit_single_meta_model(X_train=X_train, y_train=y_train, hp=hp, seed=seed)
        preds_train.append(clip_total_positive(model.predict(X_train)))
        preds_pred.append(clip_total_positive(model.predict(X_pred)))
    p_train = np.mean(np.column_stack(preds_train), axis=1)
    p_pred = np.mean(np.column_stack(preds_pred), axis=1)
    return clip_total_positive(p_train), clip_total_positive(p_pred), backend


def _evaluate_trial_cv(
    trial: TreePoissonTrialConfig,
    fold_caches: list[dict[str, Any]],
) -> dict[str, Any]:
    devs = []
    wmses = []
    cals = []
    backends = []
    t0 = time.perf_counter()

    for fc in fold_caches:
        models = [m for m in trial.base_models if m in fc["inner_oof"] and m in fc["outer_test"]]
        if len(models) < 2:
            raise RuntimeError("insufficient base models in fold cache")

        P_train = np.column_stack([fc["inner_oof"][m] for m in models])
        P_test = np.column_stack([fc["outer_test"][m] for m in models])
        y_train = np.clip(np.asarray(fc["y_train"], dtype=float), 0, None)
        y_test = np.clip(np.asarray(fc["y_test"], dtype=float), 0, None)
        toi_test = np.maximum(np.asarray(fc["toi_test"], dtype=float), EPS)

        X_train = _build_meta_features(P=P_train, df=fc["train_df"], feature_set=trial.feature_set, model_names=models)
        X_test = _build_meta_features(P=P_test, df=fc["test_df"], feature_set=trial.feature_set, model_names=models)

        mu_train_raw, mu_test_raw, backend = _fit_predict_bag(
            X_train=X_train,
            y_train=y_train,
            X_pred=X_test,
            hp=trial.hyperparams,
            seed_bag=trial.seed_bag,
            base_seed=trial.random_seed,
        )
        backends.append(backend)

        ctx_train = _context_from_df(fc["train_df"])
        ctx_test = _context_from_df(fc["test_df"])
        cal = fit_calibrator(
            y_true=y_train,
            mu_raw=mu_train_raw,
            calibration_type=trial.calibration_type,
            log_toi_hr=ctx_train["log_toi_hr"],
            n_bins=trial.cal_bins,
        )
        mu_test = cal.predict(mu_test_raw, log_toi_hr=ctx_test["log_toi_hr"])
        mu_test = clip_total_positive(mu_test)

        dev = float(poisson_deviance_safe(y_test, mu_test))
        y_rate = y_test / toi_test
        p_rate = mu_test / toi_test
        wmse = float(weighted_mse_rate(y_rate, p_rate, toi_test))
        calr = float(calibration_ratio(y_test, mu_test))
        devs.append(dev)
        wmses.append(wmse)
        cals.append(calr)

    return {
        "mean_cv_poisson_deviance": float(np.mean(devs)),
        "std_cv_poisson_deviance": float(np.std(devs, ddof=0)),
        "mean_weighted_mse_rate": float(np.mean(wmses)),
        "calibration_ratio": float(np.mean(cals)),
        "runtime_seconds": float(time.perf_counter() - t0),
        "backend": backends[0] if backends else "unknown",
    }


def _build_synthetic_grid(ev_df: pd.DataFrame):
    ref = (
        ev_df.groupby("def_unit", as_index=False)["toi_hr"]
        .sum()
        .rename(columns={"toi_hr": "w"})
    )
    ref["w"] = ref["w"] / np.maximum(ref["w"].sum(), EPS)
    def_units = ref["def_unit"].astype(str).tolist()
    weights = ref["w"].to_numpy(dtype=float)
    off_units = sorted(ev_df["off_unit"].astype(str).unique())

    n_off = len(off_units)
    n_def = len(def_units)
    off_rep = np.repeat(np.asarray(off_units), n_def)
    def_rep = np.tile(np.asarray(def_units), n_off)
    syn0 = pd.DataFrame(
        {
            "off_unit": off_rep,
            "def_unit": def_rep,
            "is_home": np.zeros(n_off * n_def, dtype=int),
            "toi_hr": np.ones(n_off * n_def, dtype=float),
            "game_id": ["synthetic"] * (n_off * n_def),
        }
    )
    syn1 = syn0.copy()
    syn1["is_home"] = 1
    return off_units, def_units, weights, syn0, syn1


def _prefit_base_predictions(
    ev_df: pd.DataFrame,
    base_builders: dict[str, Any],
    model_names: list[str],
) -> dict[str, Any]:
    off_units, def_units, weights, syn0, syn1 = _build_synthetic_grid(ev_df)
    model_cache: dict[str, dict[str, Any]] = {}
    for m in model_names:
        model = base_builders[m]()
        model.fit(ev_df)
        model_cache[m] = {
            "model": model,
            "full_total": clip_total_positive(model.predict_total(ev_df)),
            "syn0_total": clip_total_positive(model.predict_total(syn0)),
            "syn1_total": clip_total_positive(model.predict_total(syn1)),
        }
    return {
        "off_units": off_units,
        "def_units": def_units,
        "weights": weights,
        "syn0": syn0,
        "syn1": syn1,
        "model_cache": model_cache,
    }


def _compute_strengths_from_rates(
    off_units: list[str],
    def_units: list[str],
    weights: np.ndarray,
    rate_home0: np.ndarray,
    rate_home1: np.ndarray,
) -> pd.DataFrame:
    n_off = len(off_units)
    n_def = len(def_units)
    r0 = np.asarray(rate_home0, dtype=float).reshape(n_off, n_def)
    r1 = np.asarray(rate_home1, dtype=float).reshape(n_off, n_def)
    r = 0.5 * (r0 + r1)
    s = r @ weights
    rows = []
    for i, off in enumerate(off_units):
        if "__" in off:
            team, off_line = off.split("__", 1)
        else:
            team, off_line = off, ""
        sr = float(s[i])
        rows.append(
            {
                "off_unit": off,
                "team": team,
                "off_line": off_line,
                "strength_rate_hr": sr,
                "strength_xg60": sr / 60.0,
            }
        )
    return pd.DataFrame(rows)


def _evaluate_trial_validity(
    trial: TreePoissonTrialConfig,
    ev_df: pd.DataFrame,
    prefit: dict[str, Any],
) -> tuple[Phase1bValidityResult, pd.DataFrame, pd.DataFrame]:
    models = [m for m in trial.base_models if m in prefit["model_cache"]]
    if len(models) < 2:
        raise RuntimeError("insufficient base models for validity evaluation")

    y_full = np.clip(pd.to_numeric(ev_df["xg_for"], errors="coerce").to_numpy(dtype=float), 0, None)
    P_full = np.column_stack([prefit["model_cache"][m]["full_total"] for m in models])
    P_syn0 = np.column_stack([prefit["model_cache"][m]["syn0_total"] for m in models])
    P_syn1 = np.column_stack([prefit["model_cache"][m]["syn1_total"] for m in models])

    X_full = _build_meta_features(P=P_full, df=ev_df, feature_set=trial.feature_set, model_names=models)
    X_syn0 = _build_meta_features(P=P_syn0, df=prefit["syn0"], feature_set=trial.feature_set, model_names=models)
    X_syn1 = _build_meta_features(P=P_syn1, df=prefit["syn1"], feature_set=trial.feature_set, model_names=models)

    mu_train_raw, mu_syn0_raw, _ = _fit_predict_bag(
        X_train=X_full,
        y_train=y_full,
        X_pred=X_syn0,
        hp=trial.hyperparams,
        seed_bag=trial.seed_bag,
        base_seed=trial.random_seed,
    )
    _, mu_syn1_raw, _ = _fit_predict_bag(
        X_train=X_full,
        y_train=y_full,
        X_pred=X_syn1,
        hp=trial.hyperparams,
        seed_bag=trial.seed_bag,
        base_seed=trial.random_seed,
    )

    ctx_full = _context_from_df(ev_df)
    cal = fit_calibrator(
        y_true=y_full,
        mu_raw=mu_train_raw,
        calibration_type=trial.calibration_type,
        log_toi_hr=ctx_full["log_toi_hr"],
        n_bins=trial.cal_bins,
    )
    log_toi_syn = np.zeros(len(mu_syn0_raw), dtype=float)
    mu_syn0 = cal.predict(mu_syn0_raw, log_toi_hr=log_toi_syn)
    mu_syn1 = cal.predict(mu_syn1_raw, log_toi_hr=log_toi_syn)
    rate0 = np.clip(mu_syn0, EPS, None)  # toi_hr=1
    rate1 = np.clip(mu_syn1, EPS, None)  # toi_hr=1

    strength_df = _compute_strengths_from_rates(
        off_units=prefit["off_units"],
        def_units=prefit["def_units"],
        weights=np.asarray(prefit["weights"], dtype=float),
        rate_home0=rate0,
        rate_home1=rate1,
    )
    ratio_df = compute_disparity_ratios(strength_df)
    validity = check_phase1b_validity_tables(strength_df=strength_df, ratio_df=ratio_df)
    return validity, strength_df, ratio_df


class TreePoissonStackedModel(BaseModel):
    name = "TREE_POISSON_STACKED_TUNED"

    def __init__(
        self,
        random_state: int,
        base_builders: dict[str, Any],
        trial_config: TreePoissonTrialConfig,
    ):
        super().__init__(random_state=random_state)
        self.base_builders = base_builders
        self.trial_config = trial_config

    def fit(self, df: pd.DataFrame):
        self.models_ = {}
        for m in self.trial_config.base_models:
            model = self.base_builders[m]()
            model.fit(df)
            self.models_[m] = model

        self.model_names_ = [m for m in self.trial_config.base_models if m in self.models_]
        y = np.clip(pd.to_numeric(df["xg_for"], errors="coerce").to_numpy(dtype=float), 0, None)
        P = np.column_stack([clip_total_positive(self.models_[m].predict_total(df)) for m in self.model_names_])
        X = _build_meta_features(P=P, df=df, feature_set=self.trial_config.feature_set, model_names=self.model_names_)
        self.meta_models_ = []
        self.backend_ = "unknown"
        preds_train = []
        for seed in _seed_list(self.trial_config.seed_bag, self.trial_config.random_seed):
            mm, backend = _fit_single_meta_model(X_train=X, y_train=y, hp=self.trial_config.hyperparams, seed=seed)
            self.meta_models_.append(mm)
            self.backend_ = backend
            preds_train.append(clip_total_positive(mm.predict(X)))
        mu_train_raw = clip_total_positive(np.mean(np.column_stack(preds_train), axis=1))
        ctx = _context_from_df(df)
        self.calibrator_ = fit_calibrator(
            y_true=y,
            mu_raw=mu_train_raw,
            calibration_type=self.trial_config.calibration_type,
            log_toi_hr=ctx["log_toi_hr"],
            n_bins=self.trial_config.cal_bins,
        )
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        P = np.column_stack([clip_total_positive(self.models_[m].predict_total(df)) for m in self.model_names_])
        X = _build_meta_features(P=P, df=df, feature_set=self.trial_config.feature_set, model_names=self.model_names_)
        preds = [clip_total_positive(mm.predict(X)) for mm in self.meta_models_]
        mu_raw = clip_total_positive(np.mean(np.column_stack(preds), axis=1))
        ctx = _context_from_df(df)
        return clip_total_positive(self.calibrator_.predict(mu_raw, log_toi_hr=ctx["log_toi_hr"]))

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        mu = self.predict_total(df)
        toi = np.maximum(pd.to_numeric(df["toi_hr"], errors="coerce").to_numpy(dtype=float), EPS)
        return np.clip(mu / toi, 1e-12, None)


def _loguniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _sample_hp_exploit(rng: np.random.Generator, base_hp: dict[str, Any]) -> dict[str, Any]:
    md = int(base_hp.get("max_depth", 3))
    ne = int(base_hp.get("n_estimators", 200))
    lr = float(base_hp.get("learning_rate", 0.08))
    hp = {
        "max_depth": int(np.clip(md + rng.integers(-1, 2), 2, 5)),
        "learning_rate": float(np.clip(rng.normal(lr, 0.02), 0.01, 0.2)),
        "n_estimators": int(rng.choice([max(100, ne // 2), ne, min(1000, ne * 2)])),
        "min_child_weight": float(np.clip(_loguniform(rng, 0.1, 10.0), 0.1, 30.0)),
        "subsample": float(rng.uniform(0.7, 1.0)),
        "colsample_bytree": float(rng.uniform(0.7, 1.0)),
        "gamma": float(rng.choice([0.0, 0.5, 1.0, 2.0])),
        "reg_lambda": float(_loguniform(rng, 1e-3, 20.0)),
        "reg_alpha": float(_loguniform(rng, 1e-6, 2.0)),
        "max_delta_step": float(rng.choice([0.3, 0.5, 0.7, 1.0])),
        "tree_method": "hist",
    }
    return hp


def _sample_hp_explore(rng: np.random.Generator) -> dict[str, Any]:
    return {
        "max_depth": int(rng.choice([2, 3, 4, 5])),
        "learning_rate": _loguniform(rng, 0.01, 0.2),
        "n_estimators": int(rng.choice([100, 200, 400, 700, 1000])),
        "min_child_weight": _loguniform(rng, 0.1, 30.0),
        "subsample": float(rng.uniform(0.5, 1.0)),
        "colsample_bytree": float(rng.uniform(0.5, 1.0)),
        "gamma": float(rng.choice([0.0, 0.5, 1.0, 2.0, 5.0])),
        "reg_lambda": _loguniform(rng, 1e-3, 100.0),
        "reg_alpha": _loguniform(rng, 1e-6, 10.0),
        "max_delta_step": float(rng.choice([0.1, 0.3, 0.5, 0.7, 1.0, 2.0])),
        "tree_method": "hist",
    }


def _pool_id(models: list[str]) -> str:
    return "|".join(sorted(models))


def _generate_base_pools(
    baseline_models: list[str],
    all_models_ranked: list[str],
) -> list[tuple[str, list[str]]]:
    pools: list[list[str]] = []
    base = [m for m in baseline_models if m in all_models_ranked]
    if not base:
        base = all_models_ranked[: min(7, len(all_models_ranked))]
    base = base[:9]
    if len(base) < 5 and len(all_models_ranked) >= 5:
        for m in all_models_ranked:
            if m not in base:
                base.append(m)
            if len(base) >= 5:
                break
    pools.append(base)

    # One-model ablations.
    if len(base) > 5:
        for i in range(len(base)):
            ab = [m for j, m in enumerate(base) if i != j]
            if 5 <= len(ab) <= 9:
                pools.append(ab)

    # Additions and swaps.
    extras = [m for m in all_models_ranked if m not in base]
    for m in extras[:4]:
        ad = base + [m]
        if len(ad) <= 9:
            pools.append(ad)
    for i, old in enumerate(base[:3]):
        for m in extras[:4]:
            sw = base.copy()
            sw[i] = m
            sw = list(dict.fromkeys(sw))
            if 5 <= len(sw) <= 9:
                pools.append(sw)

    dedupe: dict[str, list[str]] = {}
    for p in pools:
        p2 = [x for x in p if x in all_models_ranked]
        key = _pool_id(p2)
        if 5 <= len(p2) <= 9:
            dedupe[key] = p2
    out = sorted([(k, v) for k, v in dedupe.items()], key=lambda x: (len(x[1]), x[0]))
    return out


def _choose_best_by_tie_rules(df: pd.DataFrame) -> pd.DataFrame:
    x = df.sort_values(["mean_cv_poisson_deviance", "std_cv_poisson_deviance"], ascending=[True, True]).reset_index(drop=True)
    if len(x) <= 1:
        return x.head(1)
    a = x.iloc[0]
    b = x.iloc[1]
    da = float(a["mean_cv_poisson_deviance"])
    db = float(b["mean_cv_poisson_deviance"])
    if da > 0 and ((db - da) / da) > 0.002:
        return x.head(1)
    sa = float(a["std_cv_poisson_deviance"])
    sb = float(b["std_cv_poisson_deviance"])
    if sb < sa:
        return x.iloc[[1]]
    return x.head(1)


def maximize_tree_poisson(
    ev_df: pd.DataFrame,
    outputs_dir: Path,
    base_builders: dict[str, Any],
    baseline_config: dict[str, Any],
    random_state: int = 0,
    stage1_trials: int = 500,
    stage2_trials: int = 1500,
    max_time_hours: float = 6.0,
    enable_bootstrap_tiebreak: bool = False,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or LOGGER
    rng = np.random.default_rng(random_state)

    metrics_path = outputs_dir / "metrics_summary.csv"
    ranked = list(base_builders.keys())
    if metrics_path.exists():
        try:
            ms = pd.read_csv(metrics_path)
            col = "cv_poisson_deviance_mean" if "cv_poisson_deviance_mean" in ms.columns else None
            if col is not None:
                ms = ms[(ms["status"] == "OK") & (~ms["method"].astype(str).str.startswith("ENSEMBLE_"))]
                ms = ms.sort_values(col)
                ranked2 = [m for m in ms["method"].astype(str).tolist() if m in base_builders]
                if ranked2:
                    ranked = ranked2
        except Exception:
            pass

    baseline_models = baseline_config.get("base_models", [])
    pools = _generate_base_pools(baseline_models=baseline_models, all_models_ranked=ranked)
    pool_map = {pid: models for pid, models in pools}
    all_models = sorted({m for _, mm in pools for m in mm})
    if len(all_models) < 2:
        raise RuntimeError("Not enough models for tree-poisson maximization")

    log.info("Maximize tree-poisson pools=%s models=%s", len(pools), all_models)

    fold_caches, valid_models = _build_cache_for_models(
        ev_df=ev_df,
        base_builders=base_builders,
        model_names=all_models,
        outer_splits=5,
        inner_splits=3,
        cache_dir=outputs_dir / "ensemble_oof_cache" / "bestmax_all",
    )
    valid_models = sorted(valid_models)
    if len(valid_models) < 2:
        raise RuntimeError("No valid models for max search")
    pool_map = {k: [m for m in v if m in valid_models] for k, v in pool_map.items()}
    pool_map = {k: v for k, v in pool_map.items() if len(v) >= 2}

    prefit = _prefit_base_predictions(ev_df=ev_df, base_builders=base_builders, model_names=valid_models)

    feature_sets = [
        "F0_BASE_ONLY",
        "F1_LOG_BASE",
        "F2_MIXED",
        "F3_DELTAS",
        "F4_RATIOS",
        "F5_CONTEXT_MIN",
        "F6_CONTEXT_EV_STATE",
        "F7_TEAM_HASH_CONTEXT",
        "F8_INTERACTIONS_LIGHT",
    ]
    calibrations = ["none", "scalar", "piecewise_scalar", "isotonic"]
    cal_bins_opts = [3, 5, 7]

    baseline_hp = dict(baseline_config.get("hyperparams", {}))
    if "learning_rate" not in baseline_hp:
        baseline_hp["learning_rate"] = 0.08
    if "max_depth" not in baseline_hp:
        baseline_hp["max_depth"] = 3
    if "n_estimators" not in baseline_hp:
        baseline_hp["n_estimators"] = 200

    rows: list[dict[str, Any]] = []
    best_valid = float("inf")
    no_improve = 0
    start = time.perf_counter()
    tid = 0

    def time_exceeded() -> bool:
        return (time.perf_counter() - start) >= max_time_hours * 3600.0

    def run_trial(stage: str, trial: TreePoissonTrialConfig):
        nonlocal best_valid, no_improve
        t0 = time.perf_counter()
        out_row = {
            "trial_id": trial.trial_id,
            "stage": stage,
            "base_pool_id": trial.base_pool_id,
            "base_models": json.dumps(trial.base_models),
            "feature_set": trial.feature_set,
            "calibration_type": trial.calibration_type,
            "cal_bins": int(trial.cal_bins),
            "seed_bag": int(trial.seed_bag),
            "random_seed": int(trial.random_seed),
            "hyperparams": json.dumps(trial.hyperparams, sort_keys=True),
        }
        try:
            summary = _evaluate_trial_cv(trial=trial, fold_caches=fold_caches)
            validity, _, _ = _evaluate_trial_validity(trial=trial, ev_df=ev_df, prefit=prefit)
            out_row.update(summary)
            out_row.update(
                {
                    "phase1b_valid": bool(validity.valid),
                    "validity_reason": validity.reason,
                    "strength_std_xg60": validity.strength_std_xg60,
                    "log_ratio_std": validity.log_ratio_std,
                    "ratio_unique_1e6": validity.ratio_unique_1e6,
                    "ratio_one_share_1e6": validity.ratio_one_share_1e6,
                    "status": "OK",
                    "notes": "",
                }
            )
            if validity.valid:
                d = float(summary["mean_cv_poisson_deviance"])
                if d < (best_valid - 1e-5):
                    best_valid = d
                    no_improve = 0
                else:
                    no_improve += 1
            else:
                no_improve += 1
        except Exception as exc:
            out_row.update(
                {
                    "mean_cv_poisson_deviance": np.nan,
                    "std_cv_poisson_deviance": np.nan,
                    "mean_weighted_mse_rate": np.nan,
                    "calibration_ratio": np.nan,
                    "runtime_seconds": np.nan,
                    "backend": "unknown",
                    "phase1b_valid": False,
                    "validity_reason": "error",
                    "strength_std_xg60": np.nan,
                    "log_ratio_std": np.nan,
                    "ratio_unique_1e6": np.nan,
                    "ratio_one_share_1e6": np.nan,
                    "status": "FAILED",
                    "notes": str(exc)[:300],
                }
            )
            no_improve += 1
        out_row["trial_runtime_seconds"] = float(time.perf_counter() - t0)
        rows.append(out_row)
        if len(rows) % 25 == 0:
            valid_now = [r for r in rows if r.get("status") == "OK" and bool(r.get("phase1b_valid")) and np.isfinite(r.get("mean_cv_poisson_deviance", np.nan))]
            if valid_now:
                best_now = min(float(r["mean_cv_poisson_deviance"]) for r in valid_now)
                log.info("Maximize progress rows=%s best_valid_deviance=%.9f", len(rows), best_now)
            else:
                log.info("Maximize progress rows=%s no valid trials yet", len(rows))

    # Stage 1 exploit.
    for _ in range(stage1_trials):
        if time_exceeded() or no_improve >= 400:
            break
        pid, models = pools[rng.integers(0, len(pools))]
        fs = rng.choice(feature_sets[:7])  # exploit around lower-complexity feature sets first
        cal = rng.choice(calibrations)
        cal_bins = int(rng.choice(cal_bins_opts))
        hp = _sample_hp_exploit(rng, baseline_hp)
        trial = TreePoissonTrialConfig(
            trial_id=f"mx_{tid:05d}",
            stage="stage1_exploit",
            base_pool_id=pid,
            base_models=models,
            feature_set=str(fs),
            calibration_type=str(cal),
            cal_bins=cal_bins,
            hyperparams=hp,
            seed_bag=1,
            random_seed=int(rng.integers(0, 10000)),
        )
        tid += 1
        run_trial("stage1_exploit", trial)

    # Stage 2 explore.
    for _ in range(stage2_trials):
        if time_exceeded() or no_improve >= 400:
            break
        pid, models = pools[rng.integers(0, len(pools))]
        fs = rng.choice(feature_sets)
        cal = rng.choice(calibrations)
        cal_bins = int(rng.choice(cal_bins_opts))
        hp = _sample_hp_explore(rng)
        trial = TreePoissonTrialConfig(
            trial_id=f"mx_{tid:05d}",
            stage="stage2_explore",
            base_pool_id=pid,
            base_models=models,
            feature_set=str(fs),
            calibration_type=str(cal),
            cal_bins=cal_bins,
            hyperparams=hp,
            seed_bag=1,
            random_seed=int(rng.integers(0, 10000)),
        )
        tid += 1
        run_trial("stage2_explore", trial)

    # Stage 3 refine top-20 with 3-seed recheck + seed bagging.
    df_rows = pd.DataFrame(rows)
    valid_df = df_rows[(df_rows["status"] == "OK") & (df_rows["phase1b_valid"] == True)].copy()  # noqa: E712
    top20 = valid_df.sort_values("mean_cv_poisson_deviance").head(20)

    refine_rows = []
    for _, r in top20.iterrows():
        if time_exceeded():
            break
        base_models = json.loads(r["base_models"])
        hp = json.loads(r["hyperparams"])
        for seed in [0, 1, 2]:
            if time_exceeded():
                break
            tr = TreePoissonTrialConfig(
                trial_id=f"{r['trial_id']}_s{seed}",
                stage="stage3_recheck_seed",
                base_pool_id=r["base_pool_id"],
                base_models=base_models,
                feature_set=r["feature_set"],
                calibration_type=r["calibration_type"],
                cal_bins=int(r["cal_bins"]),
                hyperparams=hp,
                seed_bag=1,
                random_seed=seed,
            )
            t0 = time.perf_counter()
            try:
                sm = _evaluate_trial_cv(trial=tr, fold_caches=fold_caches)
                vd, _, _ = _evaluate_trial_validity(trial=tr, ev_df=ev_df, prefit=prefit)
                rr = {
                    "trial_id": tr.trial_id,
                    "parent_trial_id": r["trial_id"],
                    "stage": tr.stage,
                    "base_pool_id": tr.base_pool_id,
                    "base_models": json.dumps(tr.base_models),
                    "feature_set": tr.feature_set,
                    "calibration_type": tr.calibration_type,
                    "cal_bins": tr.cal_bins,
                    "seed_bag": tr.seed_bag,
                    "random_seed": seed,
                    "hyperparams": json.dumps(hp, sort_keys=True),
                    **sm,
                    "phase1b_valid": vd.valid,
                    "validity_reason": vd.reason,
                    "strength_std_xg60": vd.strength_std_xg60,
                    "log_ratio_std": vd.log_ratio_std,
                    "ratio_unique_1e6": vd.ratio_unique_1e6,
                    "ratio_one_share_1e6": vd.ratio_one_share_1e6,
                    "status": "OK",
                    "notes": "",
                    "trial_runtime_seconds": float(time.perf_counter() - t0),
                }
            except Exception as exc:
                rr = {
                    "trial_id": tr.trial_id,
                    "parent_trial_id": r["trial_id"],
                    "stage": tr.stage,
                    "base_pool_id": tr.base_pool_id,
                    "base_models": json.dumps(tr.base_models),
                    "feature_set": tr.feature_set,
                    "calibration_type": tr.calibration_type,
                    "cal_bins": tr.cal_bins,
                    "seed_bag": tr.seed_bag,
                    "random_seed": seed,
                    "hyperparams": json.dumps(hp, sort_keys=True),
                    "mean_cv_poisson_deviance": np.nan,
                    "std_cv_poisson_deviance": np.nan,
                    "mean_weighted_mse_rate": np.nan,
                    "calibration_ratio": np.nan,
                    "runtime_seconds": np.nan,
                    "backend": "unknown",
                    "phase1b_valid": False,
                    "validity_reason": "error",
                    "strength_std_xg60": np.nan,
                    "log_ratio_std": np.nan,
                    "ratio_unique_1e6": np.nan,
                    "ratio_one_share_1e6": np.nan,
                    "status": "FAILED",
                    "notes": str(exc)[:300],
                    "trial_runtime_seconds": float(time.perf_counter() - t0),
                }
            refine_rows.append(rr)

    if refine_rows:
        rows.extend(refine_rows)

    df_rows = pd.DataFrame(rows)
    recheck = df_rows[(df_rows["stage"] == "stage3_recheck_seed") & (df_rows["status"] == "OK") & (df_rows["phase1b_valid"] == True)]  # noqa: E712
    if not recheck.empty:
        agg = (
            recheck.groupby("parent_trial_id", as_index=False)
            .agg(
                mean_cv_poisson_deviance=("mean_cv_poisson_deviance", "mean"),
                std_cv_poisson_deviance=("mean_cv_poisson_deviance", "std"),
            )
            .sort_values("mean_cv_poisson_deviance")
        )
        top10_parent = agg.head(10)["parent_trial_id"].astype(str).tolist()
    else:
        top10_parent = valid_df.sort_values("mean_cv_poisson_deviance").head(10)["trial_id"].astype(str).tolist()

    # Seed bagging for top-10.
    bag_rows = []
    base_rows = {str(r["trial_id"]): r for _, r in df_rows.iterrows() if str(r["trial_id"]) in top10_parent}
    for pid in top10_parent:
        if time_exceeded():
            break
        if pid not in base_rows:
            continue
        r = base_rows[pid]
        base_models = json.loads(r["base_models"])
        hp = json.loads(r["hyperparams"])
        for bag_k in [3, 5]:
            if time_exceeded():
                break
            tr = TreePoissonTrialConfig(
                trial_id=f"{pid}_bag{bag_k}",
                stage="stage3_seed_bag",
                base_pool_id=r["base_pool_id"],
                base_models=base_models,
                feature_set=r["feature_set"],
                calibration_type=r["calibration_type"],
                cal_bins=int(r["cal_bins"]),
                hyperparams=hp,
                seed_bag=bag_k,
                random_seed=0,
            )
            t0 = time.perf_counter()
            try:
                sm = _evaluate_trial_cv(trial=tr, fold_caches=fold_caches)
                vd, _, _ = _evaluate_trial_validity(trial=tr, ev_df=ev_df, prefit=prefit)
                rr = {
                    "trial_id": tr.trial_id,
                    "parent_trial_id": pid,
                    "stage": tr.stage,
                    "base_pool_id": tr.base_pool_id,
                    "base_models": json.dumps(tr.base_models),
                    "feature_set": tr.feature_set,
                    "calibration_type": tr.calibration_type,
                    "cal_bins": tr.cal_bins,
                    "seed_bag": tr.seed_bag,
                    "random_seed": tr.random_seed,
                    "hyperparams": json.dumps(hp, sort_keys=True),
                    **sm,
                    "phase1b_valid": vd.valid,
                    "validity_reason": vd.reason,
                    "strength_std_xg60": vd.strength_std_xg60,
                    "log_ratio_std": vd.log_ratio_std,
                    "ratio_unique_1e6": vd.ratio_unique_1e6,
                    "ratio_one_share_1e6": vd.ratio_one_share_1e6,
                    "status": "OK",
                    "notes": "",
                    "trial_runtime_seconds": float(time.perf_counter() - t0),
                }
            except Exception as exc:
                rr = {
                    "trial_id": tr.trial_id,
                    "parent_trial_id": pid,
                    "stage": tr.stage,
                    "base_pool_id": tr.base_pool_id,
                    "base_models": json.dumps(tr.base_models),
                    "feature_set": tr.feature_set,
                    "calibration_type": tr.calibration_type,
                    "cal_bins": tr.cal_bins,
                    "seed_bag": tr.seed_bag,
                    "random_seed": tr.random_seed,
                    "hyperparams": json.dumps(hp, sort_keys=True),
                    "mean_cv_poisson_deviance": np.nan,
                    "std_cv_poisson_deviance": np.nan,
                    "mean_weighted_mse_rate": np.nan,
                    "calibration_ratio": np.nan,
                    "runtime_seconds": np.nan,
                    "backend": "unknown",
                    "phase1b_valid": False,
                    "validity_reason": "error",
                    "strength_std_xg60": np.nan,
                    "log_ratio_std": np.nan,
                    "ratio_unique_1e6": np.nan,
                    "ratio_one_share_1e6": np.nan,
                    "status": "FAILED",
                    "notes": str(exc)[:300],
                    "trial_runtime_seconds": float(time.perf_counter() - t0),
                }
            bag_rows.append(rr)
    if bag_rows:
        rows.extend(bag_rows)

    results = pd.DataFrame(rows)
    results.to_csv(outputs_dir / "maximize_tree_poisson_results.csv", index=False)

    valid = results[(results["status"] == "OK") & (results["phase1b_valid"] == True) & results["mean_cv_poisson_deviance"].notna()]  # noqa: E712
    if valid.empty:
        raise RuntimeError("No Phase1b-valid trial found during maximization")
    best_df = _choose_best_by_tie_rules(valid)
    best_row = best_df.iloc[0]

    # Optional stability tie-break if still exactly tied after rules.
    ties = valid[np.isclose(valid["mean_cv_poisson_deviance"], float(best_row["mean_cv_poisson_deviance"]), rtol=0.002, atol=0.0)]
    if enable_bootstrap_tiebreak and len(ties) > 1:
        tie_eval = []
        for _, tr in ties.head(2).iterrows():
            cfg = TreePoissonTrialConfig(
                trial_id=str(tr["trial_id"]),
                stage="final_tie_eval",
                base_pool_id=str(tr["base_pool_id"]),
                base_models=json.loads(tr["base_models"]),
                feature_set=str(tr["feature_set"]),
                calibration_type=str(tr["calibration_type"]),
                cal_bins=int(tr["cal_bins"]),
                hyperparams=json.loads(tr["hyperparams"]),
                seed_bag=int(tr["seed_bag"]),
                random_seed=int(tr["random_seed"]),
            )
            model = TreePoissonStackedModel(random_state=0, base_builders=base_builders, trial_config=cfg).fit(ev_df)
            top10 = compute_disparity_ratios(compute_standardized_strengths(model, ev_df)).head(10)
            teams = top10["team"].astype(str).tolist()
            boot = bootstrap_rank_stability(
                model_factory=lambda c=cfg: TreePoissonStackedModel(
                    random_state=0,
                    base_builders=base_builders,
                    trial_config=c,
                ),
                ev_df=ev_df,
                full_top10_teams=teams,
                n_boot=200,
                random_state=0,
                show_progress=False,
            )
            tie_eval.append((float(tr["mean_cv_poisson_deviance"]), float(tr["std_cv_poisson_deviance"]), boot.stability_score, tr))
        if tie_eval:
            tie_eval = sorted(tie_eval, key=lambda x: (x[0], x[1], x[2]))
            best_row = tie_eval[0][3]

    best_trial = TreePoissonTrialConfig(
        trial_id=str(best_row["trial_id"]),
        stage="final",
        base_pool_id=str(best_row["base_pool_id"]),
        base_models=json.loads(best_row["base_models"]),
        feature_set=str(best_row["feature_set"]),
        calibration_type=str(best_row["calibration_type"]),
        cal_bins=int(best_row["cal_bins"]),
        hyperparams=json.loads(best_row["hyperparams"]),
        seed_bag=int(best_row["seed_bag"]),
        random_seed=int(best_row["random_seed"]),
    )

    best_model = TreePoissonStackedModel(
        random_state=random_state,
        base_builders=base_builders,
        trial_config=best_trial,
    ).fit(ev_df)
    validity = debug_best_disparity(
        model=best_model,
        ev_df=ev_df,
        out_path=outputs_dir / "debug_best_flatness.json",
        assert_valid=True,
    )
    strengths = compute_standardized_strengths(model=best_model, train_ev_df=ev_df)
    top10 = compute_disparity_ratios(strengths).head(10).copy()
    top10["method"] = "TREE_POISSON_MAXIMIZED"
    top10 = top10[["rank", "team", "line1_strength_xg60", "line2_strength_xg60", "ratio", "method"]]
    top10.to_csv(outputs_dir / "final_top10.csv", index=False)
    top10.to_csv(outputs_dir / "submission_phase1b.csv", index=False)

    best_payload = {
        "trial_id": best_trial.trial_id,
        "family": "tree_poisson",
        "base_pool_id": best_trial.base_pool_id,
        "base_models": best_trial.base_models,
        "feature_set": best_trial.feature_set,
        "calibration_type": best_trial.calibration_type,
        "cal_bins": best_trial.cal_bins,
        "seed_bag": best_trial.seed_bag,
        "random_seed": best_trial.random_seed,
        "hyperparams": best_trial.hyperparams,
        "mean_cv_poisson_deviance": float(best_row["mean_cv_poisson_deviance"]),
        "std_cv_poisson_deviance": float(best_row["std_cv_poisson_deviance"]),
        "phase1b_validity": asdict(validity),
        "runtime_seconds_total": float(time.perf_counter() - start),
    }
    (outputs_dir / "maximize_tree_poisson_best_config.json").write_text(
        json.dumps(best_payload, indent=2),
        encoding="utf-8",
    )

    report = []
    report.append("# Maximize Tree-Poisson Report")
    report.append("")
    report.append(f"- trials_run: {len(results)}")
    report.append(f"- phase1b_valid_trials: {int(valid.shape[0])}")
    report.append(f"- best_trial: {best_trial.trial_id}")
    report.append(f"- best_mean_cv_poisson_deviance: {best_payload['mean_cv_poisson_deviance']}")
    report.append(f"- best_std_cv_poisson_deviance: {best_payload['std_cv_poisson_deviance']}")
    report.append(f"- feature_set: {best_trial.feature_set}")
    report.append(f"- base_pool_id: {best_trial.base_pool_id}")
    report.append(f"- base_models: {', '.join(best_trial.base_models)}")
    report.append(f"- calibration: {best_trial.calibration_type} (bins={best_trial.cal_bins})")
    report.append(f"- seed_bag: {best_trial.seed_bag}")
    report.append("")
    report.append("## Top 20 Valid Trials")
    report.append("```")
    report.append(
        valid.sort_values("mean_cv_poisson_deviance")
        .head(20)[
            [
                "trial_id",
                "stage",
                "feature_set",
                "base_pool_id",
                "calibration_type",
                "seed_bag",
                "mean_cv_poisson_deviance",
                "std_cv_poisson_deviance",
                "validity_reason",
            ]
        ]
        .to_string(index=False)
    )
    report.append("```")
    report.append("")
    report.append("## Final Top 10")
    report.append("```")
    report.append(top10.to_string(index=False))
    report.append("```")
    (outputs_dir / "maximize_tree_poisson_best_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    (outputs_dir / "best_method.txt").write_text(
        f"best_method=TREE_POISSON_MAXIMIZED\n"
        f"trial_id={best_trial.trial_id}\n"
        f"mean_cv_poisson_deviance={best_payload['mean_cv_poisson_deviance']}\n",
        encoding="utf-8",
    )
    (outputs_dir / "ensemble_best_config.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")

    return {
        "best_payload": best_payload,
        "results": results,
        "top10": top10,
    }
