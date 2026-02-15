from __future__ import annotations

import json
import logging
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor, Ridge
from sklearn.model_selection import GroupKFold

from whsdsci.ensemble.calibration import fit_calibrator
from whsdsci.ensemble.oof import build_oof_predictions, clip_total_positive
from whsdsci.eval.bootstrap import bootstrap_rank_stability
from whsdsci.eval.metrics import calibration_ratio, mae_total, poisson_deviance_safe, weighted_mse_rate
from whsdsci.models import get_model_builders
from whsdsci.models.base import BaseModel
from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


EPS = 1e-9
LOGGER = logging.getLogger(__name__)


@dataclass
class Config:
    config_id: str
    combiner_family: str
    base_pool_id: str
    base_models: list[str]
    calibration_type: str
    hyperparams: dict[str, Any]


class FittedSearchModel(BaseModel):
    name = "ENSEMBLE_SEARCH_WINNER"

    def __init__(
        self,
        random_state: int,
        base_builders: dict[str, Callable[[], BaseModel]],
        config: Config,
    ):
        super().__init__(random_state=random_state)
        self.base_builders = base_builders
        self.config = config

    def fit(self, df: pd.DataFrame):
        self.base_models_ = {}
        for m in self.config.base_models:
            model = self.base_builders[m]()
            model.fit(df)
            self.base_models_[m] = model

        P = self._predict_base(df, self.config.base_models)
        y = np.clip(pd.to_numeric(df["xg_for"], errors="coerce").to_numpy(dtype=float), 0, None)
        shots_vec = pd.to_numeric(df["shots_for"], errors="coerce") if "shots_for" in df.columns else pd.Series(0.0, index=df.index)
        self.default_shots_for_ = float(np.nanmedian(np.clip(shots_vec.fillna(0.0).to_numpy(dtype=float), 0, None)))
        ctx = _context_from_df(df, default_shots_for=self.default_shots_for_)
        comb = _fit_combiner(self.config.combiner_family, self.config.hyperparams, P, y, ctx, self.config.base_models)
        self.combiner_ = comb
        mu_train_raw = _predict_combiner(comb, P, ctx)
        self.calibrator_ = fit_calibrator(
            y_true=y,
            mu_raw=mu_train_raw,
            calibration_type=self.config.calibration_type,
            log_toi_hr=ctx["log_toi_hr"],
            n_bins=int(self.config.hyperparams.get("cal_bins", 4)),
        )
        return self

    def _predict_base(self, df: pd.DataFrame, models: list[str]) -> np.ndarray:
        mats = []
        for m in models:
            mu = self.base_models_[m].predict_total(df)
            mats.append(clip_total_positive(mu))
        return np.column_stack(mats)

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        P = self._predict_base(df, self.config.base_models)
        ctx = _context_from_df(df, default_shots_for=getattr(self, "default_shots_for_", 0.0))
        mu_raw = _predict_combiner(self.combiner_, P, ctx)
        mu = self.calibrator_.predict(mu_raw, log_toi_hr=ctx["log_toi_hr"])
        return clip_total_positive(mu)

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        mu = self.predict_total(df)
        toi = np.maximum(pd.to_numeric(df["toi_hr"], errors="coerce").to_numpy(dtype=float), EPS)
        return np.clip(mu / toi, 1e-12, None)


def _context_from_df(df: pd.DataFrame, default_shots_for: float = 0.0) -> dict[str, np.ndarray]:
    toi_raw = df["toi_hr"] if "toi_hr" in df.columns else pd.Series(1.0, index=df.index)
    toi = np.maximum(pd.to_numeric(toi_raw, errors="coerce"), EPS)
    log_toi = np.log(toi).to_numpy(dtype=float)
    home_raw = df["is_home"] if "is_home" in df.columns else pd.Series(0.0, index=df.index)
    shots_raw = df["shots_for"] if "shots_for" in df.columns else pd.Series(float(default_shots_for), index=df.index)
    is_home = pd.to_numeric(home_raw, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    shots = pd.to_numeric(shots_raw, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    shots_zero = (shots <= 0).astype(float)
    return {
        "log_toi_hr": log_toi,
        "is_home": is_home,
        "shots_for": shots,
        "shots_zero": shots_zero,
    }


def _softmax(z: np.ndarray, temp: float = 1.0) -> np.ndarray:
    t = max(float(temp), 1e-6)
    x = np.asarray(z, dtype=float) / t
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(np.clip(x, -60, 60))
    return ex / np.maximum(ex.sum(axis=1, keepdims=True), EPS)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    x = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60, 60)))


def _fit_convex_weights(P: np.ndarray, y: np.ndarray, lam: float = 0.0) -> np.ndarray:
    m = P.shape[1]
    x0 = np.full(m, 1.0 / m, dtype=float)

    def objective(w):
        ww = np.clip(np.asarray(w, dtype=float), 0, 1)
        ww = ww / max(float(np.sum(ww)), EPS)
        mu = clip_total_positive(P @ ww)
        return float(poisson_deviance_safe(y, mu) + lam * np.sum(ww * ww))

    cons = [{"type": "eq", "fun": lambda w: np.sum(np.clip(w, 0, 1)) - 1.0}]
    bnds = [(0.0, 1.0)] * m
    res = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 500})
    if not res.success:
        return x0
    w = np.clip(np.asarray(res.x, dtype=float), 0, 1)
    return w / max(float(np.sum(w)), EPS)


def _fit_logspace(P: np.ndarray, y: np.ndarray, lam: float = 0.0) -> tuple[np.ndarray, float]:
    L = np.log(np.clip(P, EPS, None))
    m = L.shape[1]
    x0 = np.concatenate([np.full(m, 1.0 / m, dtype=float), np.array([0.0])])

    def objective(x):
        a = np.clip(np.asarray(x[:-1], dtype=float), 0, 1)
        a = a / max(float(np.sum(a)), EPS)
        b = float(x[-1])
        mu = clip_total_positive(np.exp(b + L @ a))
        return float(poisson_deviance_safe(y, mu) + lam * np.sum(a * a))

    cons = [{"type": "eq", "fun": lambda x: np.sum(np.clip(x[:-1], 0, 1)) - 1.0}]
    bnds = [(0.0, 1.0)] * m + [(-5.0, 5.0)]
    res = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons, options={"maxiter": 500})
    x = res.x if res.success else x0
    a = np.clip(np.asarray(x[:-1], dtype=float), 0, 1)
    a = a / max(float(np.sum(a)), EPS)
    b = float(x[-1])
    return a, b


def _power_mean(P: np.ndarray, p: float, trim_frac: float = 0.0) -> np.ndarray:
    X = np.clip(np.asarray(P, dtype=float), EPS, None)
    if trim_frac > 0:
        lo = np.quantile(X, trim_frac, axis=1, keepdims=True)
        hi = np.quantile(X, 1.0 - trim_frac, axis=1, keepdims=True)
        keep = (X >= lo) & (X <= hi)
        keep = np.where(keep.any(axis=1, keepdims=True), keep, np.ones_like(keep, dtype=bool))
        X = np.where(keep, X, np.nan)
    if p == 0:
        out = np.exp(np.nanmean(np.log(np.clip(X, EPS, None)), axis=1))
    else:
        out = np.power(np.nanmean(np.power(np.clip(X, EPS, None), p), axis=1), 1.0 / p)
    return clip_total_positive(np.nan_to_num(out, nan=EPS, posinf=1e9, neginf=EPS))


def _fit_moe(
    P: np.ndarray,
    y: np.ndarray,
    ctx: dict[str, np.ndarray],
    n_experts: int,
    variant: str,
    temperature: float,
    l2: float,
) -> dict[str, Any]:
    n_experts = max(2, min(int(n_experts), P.shape[1]))

    devs = [poisson_deviance_safe(y, clip_total_positive(P[:, i])) for i in range(P.shape[1])]
    expert_idx = np.argsort(devs)[:n_experts]
    Pe = P[:, expert_idx]

    base_feats = [np.ones(len(y), dtype=float), ctx["log_toi_hr"], ctx["is_home"]]
    if variant in {"B", "C"}:
        base_feats.append(ctx["shots_zero"])
    if variant == "C":
        base_feats.extend([np.log(np.clip(Pe[:, i], EPS, None)) for i in range(Pe.shape[1])])
    F = np.column_stack(base_feats)

    if n_experts == 2:
        p = F.shape[1]
        theta0 = np.zeros(p, dtype=float)

        def objective(theta):
            w = _sigmoid(F @ theta)
            mu = clip_total_positive(w * Pe[:, 0] + (1.0 - w) * Pe[:, 1])
            return float(poisson_deviance_safe(y, mu) + l2 * np.sum(theta * theta))

        res = minimize(objective, theta0, method="L-BFGS-B", options={"maxiter": 500})
        theta = res.x if res.success else theta0
        return {
            "kind": "moe2",
            "expert_idx": expert_idx,
            "theta": theta,
            "variant": variant,
            "temperature": temperature,
        }

    p = F.shape[1]
    theta0 = np.zeros((p, n_experts), dtype=float)

    def objective(theta_vec):
        Th = theta_vec.reshape(p, n_experts)
        logits = F @ Th
        w = _softmax(logits, temp=temperature)
        mu = clip_total_positive(np.sum(w * Pe, axis=1))
        return float(poisson_deviance_safe(y, mu) + l2 * np.sum(theta_vec * theta_vec))

    res = minimize(objective, theta0.ravel(), method="L-BFGS-B", options={"maxiter": 500})
    theta = res.x.reshape(p, n_experts) if res.success else theta0

    return {
        "kind": "moek",
        "expert_idx": expert_idx,
        "theta": theta,
        "variant": variant,
        "temperature": temperature,
    }


def _predict_moe(model: dict[str, Any], P: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    idx = np.asarray(model["expert_idx"], dtype=int)
    Pe = P[:, idx]

    variant = model.get("variant", "A")
    base_feats = [np.ones(P.shape[0], dtype=float), ctx["log_toi_hr"], ctx["is_home"]]
    if variant in {"B", "C"}:
        base_feats.append(ctx["shots_zero"])
    if variant == "C":
        base_feats.extend([np.log(np.clip(Pe[:, i], EPS, None)) for i in range(Pe.shape[1])])
    F = np.column_stack(base_feats)

    if model["kind"] == "moe2":
        theta = np.asarray(model["theta"], dtype=float)
        w = _sigmoid(F @ theta)
        mu = w * Pe[:, 0] + (1.0 - w) * Pe[:, 1]
        return clip_total_positive(mu)

    Th = np.asarray(model["theta"], dtype=float)
    logits = F @ Th
    w = _softmax(logits, temp=float(model.get("temperature", 1.0)))
    mu = np.sum(w * Pe, axis=1)
    return clip_total_positive(mu)


def _fit_regime_weights(P: np.ndarray, y: np.ndarray, wtype: str) -> np.ndarray:
    if wtype == "nnls":
        w, _ = nnls(P, y)
        if float(np.sum(w)) <= 0:
            w = np.ones(P.shape[1], dtype=float)
        return w
    return _fit_convex_weights(P, y, lam=0.0)


def _fit_combiner(
    family: str,
    hp: dict[str, Any],
    P_train: np.ndarray,
    y_train: np.ndarray,
    ctx_train: dict[str, np.ndarray],
    pool_models: list[str],
) -> dict[str, Any]:
    family = str(family)

    if family == "mean":
        return {"family": family}

    if family == "convex":
        lam = float(hp.get("lambda", 0.0))
        w = _fit_convex_weights(P_train, y_train, lam=lam)
        return {"family": family, "w": w, "lambda": lam}

    if family == "nnls":
        w, _ = nnls(P_train, y_train)
        if float(np.sum(w)) <= 0:
            w = np.ones(P_train.shape[1], dtype=float)
        return {"family": family, "w": w}

    if family == "logspace":
        lam = float(hp.get("lambda", 0.0))
        a, b = _fit_logspace(P_train, y_train, lam=lam)
        return {"family": family, "a": a, "b": float(b), "lambda": lam}

    if family == "power_mean":
        return {"family": family, "p": float(hp.get("p", 1.0)), "trim": float(hp.get("trim", 0.0))}

    if family == "median":
        return {"family": family, "trim": float(hp.get("trim", 0.0))}

    if family == "stack_poisson":
        alpha = float(hp.get("alpha", 1e-4))
        include_ctx = bool(hp.get("include_ctx", True))
        X = np.log(np.clip(P_train, EPS, None))
        if include_ctx:
            X = np.column_stack([X, ctx_train["log_toi_hr"], ctx_train["is_home"], ctx_train["shots_zero"]])
        model = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=1000)
        model.fit(X, y_train)
        return {"family": family, "model": model, "alpha": alpha, "include_ctx": include_ctx}

    if family == "tree_poisson":
        backend = hp.get("backend", "auto")
        max_depth = int(hp.get("max_depth", 3))
        lr = float(hp.get("learning_rate", 0.05))
        n_estimators = int(hp.get("n_estimators", 300))

        X = np.column_stack(
            [
                np.log(np.clip(P_train, EPS, None)),
                ctx_train["log_toi_hr"],
                ctx_train["is_home"],
                ctx_train["shots_zero"],
            ]
        )

        model_obj = None
        used_backend = "sklearn_histgbr"
        if backend in {"auto", "xgboost"}:
            try:
                from xgboost import XGBRegressor

                model_obj = XGBRegressor(
                    objective="count:poisson",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=lr,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=0,
                    tree_method="hist",
                    n_jobs=1,
                )
                model_obj.fit(X, y_train)
                used_backend = "xgboost"
            except Exception:
                model_obj = None

        if model_obj is None:
            model_obj = HistGradientBoostingRegressor(
                loss="poisson",
                max_depth=max_depth,
                learning_rate=lr,
                max_iter=n_estimators,
                random_state=0,
            )
            model_obj.fit(X, y_train)

        return {
            "family": family,
            "model": model_obj,
            "backend": used_backend,
            "max_depth": max_depth,
            "learning_rate": lr,
            "n_estimators": n_estimators,
        }

    if family == "residual_corr":
        alpha = float(hp.get("alpha", 1.0))
        if "POISSON_GLM_OFFSET_REG" in pool_models:
            bidx = pool_models.index("POISSON_GLM_OFFSET_REG")
        else:
            devs = [poisson_deviance_safe(y_train, clip_total_positive(P_train[:, i])) for i in range(P_train.shape[1])]
            bidx = int(np.argmin(devs))
        mu0 = np.clip(P_train[:, bidx], EPS, None)
        t = np.log((y_train + EPS) / (mu0 + EPS))
        X = np.column_stack(
            [
                np.log(np.clip(P_train, EPS, None)),
                ctx_train["log_toi_hr"],
                ctx_train["is_home"],
                ctx_train["shots_zero"],
            ]
        )
        model = Ridge(alpha=alpha, random_state=0)
        model.fit(X, t)
        return {"family": family, "model": model, "baseline_idx": bidx, "alpha": alpha}

    if family == "regime_blend":
        n_bins = int(hp.get("n_bins", 4))
        use_shots_zero = bool(hp.get("use_shots_zero", True))
        wtype = str(hp.get("weight_type", "convex"))

        lt = ctx_train["log_toi_hr"]
        edges = np.quantile(lt, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            edges = np.array([lt.min() - 1e-9, lt.max() + 1e-9])
        bid = np.digitize(lt, edges[1:-1], right=False)

        if use_shots_zero:
            rid = bid.astype(int) * 10 + ctx_train["shots_zero"].astype(int)
        else:
            rid = bid.astype(int)

        global_w = _fit_regime_weights(P_train, y_train, wtype=wtype)
        regime_w: dict[int, np.ndarray] = {}
        for k in np.unique(rid):
            m = rid == k
            if int(m.sum()) < 30:
                regime_w[int(k)] = global_w
            else:
                regime_w[int(k)] = _fit_regime_weights(P_train[m], y_train[m], wtype=wtype)

        return {
            "family": family,
            "bin_edges": edges,
            "use_shots_zero": use_shots_zero,
            "weight_type": wtype,
            "regime_weights": regime_w,
            "global_w": global_w,
        }

    if family == "moe":
        n_experts = int(hp.get("n_experts", 2))
        variant = str(hp.get("variant", "A"))
        temp = float(hp.get("temperature", 1.0))
        l2 = float(hp.get("l2", 0.0))
        model = _fit_moe(P_train, y_train, ctx_train, n_experts=n_experts, variant=variant, temperature=temp, l2=l2)
        return {"family": family, "model": model, "n_experts": n_experts, "variant": variant, "temperature": temp, "l2": l2}

    raise ValueError(f"Unknown combiner family: {family}")


def _predict_combiner(model: dict[str, Any], P: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    family = model["family"]

    if family == "mean":
        return clip_total_positive(np.mean(P, axis=1))

    if family == "convex":
        return clip_total_positive(P @ np.asarray(model["w"], dtype=float))

    if family == "nnls":
        return clip_total_positive(P @ np.asarray(model["w"], dtype=float))

    if family == "logspace":
        a = np.asarray(model["a"], dtype=float)
        b = float(model["b"])
        return clip_total_positive(np.exp(b + np.log(np.clip(P, EPS, None)) @ a))

    if family == "power_mean":
        return _power_mean(P, p=float(model.get("p", 1.0)), trim_frac=float(model.get("trim", 0.0)))

    if family == "median":
        trim = float(model.get("trim", 0.0))
        if trim > 0:
            lo = np.quantile(P, trim, axis=1, keepdims=True)
            hi = np.quantile(P, 1.0 - trim, axis=1, keepdims=True)
            keep = (P >= lo) & (P <= hi)
            keep = np.where(keep.any(axis=1, keepdims=True), keep, np.ones_like(keep, dtype=bool))
            X = np.where(keep, P, np.nan)
            mu = np.nanmedian(X, axis=1)
        else:
            mu = np.median(P, axis=1)
        return clip_total_positive(np.nan_to_num(mu, nan=EPS, posinf=1e9, neginf=EPS))

    if family == "stack_poisson":
        X = np.log(np.clip(P, EPS, None))
        if bool(model.get("include_ctx", True)):
            X = np.column_stack([X, ctx["log_toi_hr"], ctx["is_home"], ctx["shots_zero"]])
        mu = model["model"].predict(X)
        return clip_total_positive(mu)

    if family == "tree_poisson":
        X = np.column_stack([np.log(np.clip(P, EPS, None)), ctx["log_toi_hr"], ctx["is_home"], ctx["shots_zero"]])
        mu = model["model"].predict(X)
        return clip_total_positive(mu)

    if family == "residual_corr":
        bidx = int(model["baseline_idx"])
        mu0 = np.clip(P[:, bidx], EPS, None)
        X = np.column_stack([np.log(np.clip(P, EPS, None)), ctx["log_toi_hr"], ctx["is_home"], ctx["shots_zero"]])
        corr = np.exp(model["model"].predict(X))
        return clip_total_positive(mu0 * corr)

    if family == "regime_blend":
        edges = np.asarray(model["bin_edges"], dtype=float)
        bid = np.digitize(ctx["log_toi_hr"], edges[1:-1], right=False)
        if bool(model.get("use_shots_zero", True)):
            rid = bid.astype(int) * 10 + ctx["shots_zero"].astype(int)
        else:
            rid = bid.astype(int)
        rw = model["regime_weights"]
        gw = np.asarray(model["global_w"], dtype=float)
        out = np.zeros(P.shape[0], dtype=float)
        for i, k in enumerate(rid):
            w = np.asarray(rw.get(int(k), gw), dtype=float)
            out[i] = float(np.dot(P[i], w))
        return clip_total_positive(out)

    if family == "moe":
        return _predict_moe(model["model"], P, ctx)

    raise ValueError(f"Unknown combiner family: {family}")


def _evaluate_predictions(y_true: np.ndarray, mu_pred: np.ndarray, toi_hr: np.ndarray) -> dict[str, float]:
    y = np.clip(np.asarray(y_true, dtype=float), 0, None)
    mu = clip_total_positive(mu_pred)
    toi = np.maximum(np.asarray(toi_hr, dtype=float), EPS)
    y_rate = y / toi
    p_rate = mu / toi
    return {
        "poisson_deviance": float(poisson_deviance_safe(y, mu)),
        "weighted_mse_rate": float(weighted_mse_rate(y_rate, p_rate, toi)),
        "mae_total": float(mae_total(y, mu)),
        "calibration_ratio": float(calibration_ratio(y, mu)),
    }


def _is_glm_like(name: str) -> bool:
    nm = name.upper()
    return ("POISSON" in nm) or ("TWEEDIE" in nm) or ("GLM" in nm)


def _build_base_pool_sets(
    ranked_models: list[str],
    global_oof: pd.DataFrame,
    random_state: int,
    max_random_per_size: int = 50,
) -> list[tuple[str, list[str], str]]:
    rng = np.random.default_rng(random_state)
    pools: list[tuple[str, list[str], str]] = []
    seen: set[tuple[str, ...]] = set()

    def add_pool(pid: str, models: list[str], ptype: str):
        models2 = [m for m in models if m in ranked_models]
        if len(models2) < 2:
            return
        key = tuple(sorted(models2))
        if key in seen:
            return
        seen.add(key)
        pools.append((pid, models2, ptype))

    # A) TopK pools
    for k in [3, 4, 5, 6, 8, 10]:
        if len(ranked_models) >= k:
            add_pool(f"topk_{k}", ranked_models[:k], "topk")

    # B) Forced pools
    if "POISSON_GLM_OFFSET_REG" in ranked_models:
        for k in [4, 6, 8]:
            if len(ranked_models) >= k:
                forced = ["POISSON_GLM_OFFSET_REG"] + [m for m in ranked_models if m != "POISSON_GLM_OFFSET_REG"][: k - 1]
                add_pool(f"forced_offsetreg_{k}", forced, "forced")

    diverse = [m for m in ["POISSON_GLM_OFFSET", "TWEEDIE_GLM_RATE", "TWO_STAGE_SHOTS_XG"] if m in ranked_models]
    if len(diverse) >= 2:
        for k in [3, 5, 7]:
            extra = [m for m in ranked_models if m not in diverse][: max(0, k - len(diverse))]
            add_pool(f"diverse_trio_{k}", diverse + extra, "forced")

    # C) Diversity-pruned greedy pools
    corr = None
    cols = [f"mu_pred_total_{m}" for m in ranked_models if f"mu_pred_total_{m}" in global_oof.columns]
    if len(cols) >= 2:
        corr = global_oof[cols].corr().to_numpy(dtype=float)

    if corr is not None:
        for k in [3, 4, 5, 6, 8]:
            if len(ranked_models) < k:
                continue
            selected = []
            for m in ranked_models:
                if not selected:
                    selected.append(m)
                    continue
                ok = True
                i = ranked_models.index(m)
                for s in selected:
                    j = ranked_models.index(s)
                    c = corr[i, j]
                    if np.isfinite(c) and abs(c) > 0.995:
                        ok = False
                        break
                if ok:
                    selected.append(m)
                if len(selected) >= k:
                    break
            if len(selected) >= 2:
                add_pool(f"diverse_greedy_{k}", selected, "diverse")

    # D) Random subset pools
    n_models = len(ranked_models)
    for s in [3, 4, 5, 6, 7, 8]:
        if n_models < s:
            continue
        n_try = 0
        n_ok = 0
        while n_ok < max_random_per_size and n_try < max_random_per_size * 20:
            n_try += 1
            pick = rng.choice(ranked_models, size=s, replace=False).tolist()
            if not any(_is_glm_like(m) for m in pick):
                continue
            # Reject near-perfect collinearity in random pool.
            bad = False
            for i in range(len(pick)):
                for j in range(i + 1, len(pick)):
                    ci = f"mu_pred_total_{pick[i]}"
                    cj = f"mu_pred_total_{pick[j]}"
                    if ci not in global_oof.columns or cj not in global_oof.columns:
                        continue
                    v1 = global_oof[ci].to_numpy(dtype=float)
                    v2 = global_oof[cj].to_numpy(dtype=float)
                    if np.std(v1) <= 0 or np.std(v2) <= 0:
                        c = 1.0
                    else:
                        c = float(np.corrcoef(v1, v2)[0, 1])
                    if np.isfinite(c) and abs(c) > 0.9999:
                        bad = True
                        break
                if bad:
                    break
            if bad:
                continue
            add_pool(f"rand_s{s}_{n_ok}", pick, "random")
            n_ok += 1

    return pools


def _generate_config_candidates(
    pools: list[tuple[str, list[str], str]],
    random_state: int,
    max_configs: int = 1200,
) -> list[Config]:
    rng = np.random.default_rng(random_state)

    family_defs: dict[str, list[dict[str, Any]]] = {
        "mean": [{"calibration": c} for c in ["none", "scalar", "piecewise_scalar", "isotonic"]],
        "convex": [
            {"lambda": lam, "calibration": c}
            for lam in [0.0, 1e-4, 1e-3, 1e-2]
            for c in ["none", "scalar", "piecewise_scalar", "isotonic"]
        ],
        "nnls": [{"calibration": c} for c in ["none", "scalar", "piecewise_scalar", "isotonic"]],
        "logspace": [
            {"lambda": lam, "calibration": c}
            for lam in [0.0, 1e-4, 1e-3, 1e-2]
            for c in ["none", "scalar", "isotonic"]
        ],
        "power_mean": [
            {"p": p, "trim": tr, "calibration": c}
            for p in [-2, -1, -0.5, 0, 0.5, 1, 2]
            for tr in [0.0, 0.1]
            for c in ["none", "scalar"]
        ],
        "median": [{"trim": tr, "calibration": c} for tr in [0.0, 0.1] for c in ["none", "scalar"]],
        "stack_poisson": [
            {"alpha": a, "include_ctx": ic, "calibration": c}
            for a in [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]
            for ic in [True, False]
            for c in ["none", "scalar", "isotonic"]
        ],
        "tree_poisson": [
            {
                "backend": b,
                "max_depth": d,
                "learning_rate": lr,
                "n_estimators": ne,
                "calibration": c,
            }
            for b in ["auto", "xgboost"]
            for d in [2, 3, 4]
            for lr in [0.03, 0.08]
            for ne in [200, 400]
            for c in ["none", "scalar"]
        ],
        "residual_corr": [
            {"alpha": a, "calibration": c}
            for a in [0.1, 1.0, 10.0, 30.0]
            for c in ["none", "scalar", "isotonic"]
        ],
        "regime_blend": [
            {"n_bins": nb, "use_shots_zero": sz, "weight_type": wt, "calibration": c}
            for nb in [3, 5]
            for sz in [False, True]
            for wt in ["convex", "nnls"]
            for c in ["none", "scalar"]
        ],
        "moe": [
            {
                "n_experts": ne,
                "variant": var,
                "temperature": temp,
                "l2": l2,
                "calibration": c,
            }
            for ne in [2, 3, 4]
            for var in ["A", "B", "C"]
            for temp in [0.7, 1.0, 1.5]
            for l2 in [0.0, 1e-3, 1e-2]
            for c in ["none", "scalar"]
        ],
    }

    cfgs: list[Config] = []
    cid = 0

    # Deterministic rich coverage for non-random pools.
    for pool_id, models, ptype in pools:
        fam_keys = list(family_defs.keys()) if ptype != "random" else ["nnls", "logspace", "power_mean", "median", "regime_blend"]
        for fam in fam_keys:
            options = family_defs[fam]
            if ptype == "random":
                take = min(3, len(options))
                pick_idx = rng.choice(np.arange(len(options)), size=take, replace=False)
                subset = [options[i] for i in pick_idx]
            else:
                subset = options
            for hp0 in subset:
                hp = dict(hp0)
                cal = str(hp.pop("calibration", "none"))
                # Guard expert count vs pool size.
                if fam == "moe" and int(hp.get("n_experts", 2)) > len(models):
                    continue
                cfgs.append(
                    Config(
                        config_id=f"cfg_{cid:05d}",
                        combiner_family=fam,
                        base_pool_id=pool_id,
                        base_models=models,
                        calibration_type=cal,
                        hyperparams=hp,
                    )
                )
                cid += 1

    if len(cfgs) > max_configs:
        keep_idx = rng.choice(np.arange(len(cfgs)), size=max_configs, replace=False)
        keep_idx = np.sort(keep_idx)
        cfgs = [cfgs[i] for i in keep_idx]
        for i, c in enumerate(cfgs):
            c.config_id = f"cfg_{i:05d}"

    return cfgs


def _config_runtime_note(exc: Exception) -> str:
    txt = str(exc)
    if len(txt) > 300:
        txt = txt[:300] + "..."
    return txt


def _build_cache_for_models(
    ev_df: pd.DataFrame,
    base_builders: dict[str, Callable[[], BaseModel]],
    model_names: list[str],
    outer_splits: int,
    inner_splits: int,
    cache_dir: Path,
) -> tuple[list[dict[str, Any]], list[str]]:
    cache_dir.mkdir(parents=True, exist_ok=True)

    groups = ev_df["game_id"].astype(str).to_numpy()
    n_outer = max(2, min(outer_splits, ev_df["game_id"].astype(str).nunique()))
    outer = GroupKFold(n_splits=n_outer)
    rows = np.arange(len(ev_df))

    fold_caches: list[dict[str, Any]] = []
    good_models = set(model_names)

    for fidx, (tr, te) in enumerate(outer.split(rows, groups=groups), start=1):
        tr_df = ev_df.iloc[tr].reset_index(drop=True)
        te_df = ev_df.iloc[te].reset_index(drop=True)

        fc: dict[str, Any] = {
            "fold_id": f"outer_{fidx}",
            "train_df": tr_df,
            "test_df": te_df,
            "y_train": np.clip(pd.to_numeric(tr_df["xg_for"], errors="coerce").to_numpy(dtype=float), 0, None),
            "y_test": np.clip(pd.to_numeric(te_df["xg_for"], errors="coerce").to_numpy(dtype=float), 0, None),
            "toi_train": np.maximum(pd.to_numeric(tr_df["toi_hr"], errors="coerce").to_numpy(dtype=float), EPS),
            "toi_test": np.maximum(pd.to_numeric(te_df["toi_hr"], errors="coerce").to_numpy(dtype=float), EPS),
            "ctx_train": _context_from_df(tr_df),
            "ctx_test": _context_from_df(te_df),
            "inner_oof": {},
            "outer_test": {},
        }

        # Reuse cached fold predictions when available.
        train_cache = cache_dir / f"{fc['fold_id']}_train.parquet"
        test_cache = cache_dir / f"{fc['fold_id']}_test.parquet"
        loaded_from_cache = False
        if train_cache.exists() and test_cache.exists():
            try:
                tr_tbl = pd.read_parquet(train_cache)
                te_tbl = pd.read_parquet(test_cache)
                if len(tr_tbl) == len(tr_df) and len(te_tbl) == len(te_df):
                    cached_models = []
                    for m in model_names:
                        c = f"pred_{m}"
                        if c in tr_tbl.columns and c in te_tbl.columns:
                            cached_models.append(m)
                            fc["inner_oof"][m] = clip_total_positive(tr_tbl[c].to_numpy(dtype=float))
                            fc["outer_test"][m] = clip_total_positive(te_tbl[c].to_numpy(dtype=float))
                        else:
                            good_models.discard(m)
                    if len(cached_models) >= 2:
                        loaded_from_cache = True
            except Exception:
                loaded_from_cache = False

        if loaded_from_cache:
            fold_caches.append(fc)
            continue

        for m in model_names:
            if m not in good_models:
                continue
            try:
                mdl = base_builders[m]()
                mdl.fit(tr_df)
                fc["outer_test"][m] = clip_total_positive(mdl.predict_total(te_df))

                g_inner = tr_df["game_id"].astype(str).to_numpy()
                n_inner = max(2, min(inner_splits, tr_df["game_id"].astype(str).nunique()))
                inner = GroupKFold(n_splits=n_inner)
                idx_inner = np.arange(len(tr_df))
                oof = np.full(len(tr_df), np.nan, dtype=float)
                for itr, iva in inner.split(idx_inner, groups=g_inner):
                    dtr = tr_df.iloc[itr].reset_index(drop=True)
                    dva = tr_df.iloc[iva].reset_index(drop=True)
                    m2 = base_builders[m]()
                    m2.fit(dtr)
                    oof[iva] = clip_total_positive(m2.predict_total(dva))
                if np.isnan(oof).any():
                    raise RuntimeError(f"inner OOF has NaN for model {m}")
                fc["inner_oof"][m] = clip_total_positive(oof)
            except Exception as exc:
                good_models.discard(m)
                fc["inner_oof"].pop(m, None)
                fc["outer_test"].pop(m, None)
                LOGGER.warning("Cache build failed for %s on %s: %s", m, fc["fold_id"], exc)

        # persist lightweight fold cache tables
        train_tbl = pd.DataFrame({
            "y_true": fc["y_train"],
            "toi_hr": fc["toi_train"],
        })
        test_tbl = pd.DataFrame({
            "y_true": fc["y_test"],
            "toi_hr": fc["toi_test"],
        })
        for m, v in fc["inner_oof"].items():
            train_tbl[f"pred_{m}"] = v
        for m, v in fc["outer_test"].items():
            test_tbl[f"pred_{m}"] = v

        train_tbl.to_parquet(cache_dir / f"{fc['fold_id']}_train.parquet", index=False)
        test_tbl.to_parquet(cache_dir / f"{fc['fold_id']}_test.parquet", index=False)

        fold_caches.append(fc)

    # prune models not good in all folds
    valid_models = sorted(list(good_models))
    for fc in fold_caches:
        fc["inner_oof"] = {k: v for k, v in fc["inner_oof"].items() if k in valid_models}
        fc["outer_test"] = {k: v for k, v in fc["outer_test"].items() if k in valid_models}

    return fold_caches, valid_models


def _eval_config_on_fold(cfg: Config, fc: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    models = [m for m in cfg.base_models if m in fc["inner_oof"] and m in fc["outer_test"]]
    if len(models) < 2:
        raise RuntimeError("Not enough valid base models for config on fold")

    P_train = np.column_stack([fc["inner_oof"][m] for m in models])
    P_test = np.column_stack([fc["outer_test"][m] for m in models])

    y_train = fc["y_train"]
    y_test = fc["y_test"]

    comb = _fit_combiner(cfg.combiner_family, cfg.hyperparams, P_train, y_train, fc["ctx_train"], models)
    mu_train_raw = _predict_combiner(comb, P_train, fc["ctx_train"])
    mu_test_raw = _predict_combiner(comb, P_test, fc["ctx_test"])

    cal = fit_calibrator(
        y_true=y_train,
        mu_raw=mu_train_raw,
        calibration_type=cfg.calibration_type,
        log_toi_hr=fc["ctx_train"]["log_toi_hr"],
        n_bins=int(cfg.hyperparams.get("cal_bins", 4)),
    )
    mu_test = cal.predict(mu_test_raw, log_toi_hr=fc["ctx_test"]["log_toi_hr"])

    metrics = _evaluate_predictions(y_test, mu_test, fc["toi_test"])

    details = {
        "combiner": {k: (v if isinstance(v, (float, int, str, bool, list, dict)) else str(type(v))) for k, v in comb.items() if k not in {"model"}},
        "calibration": cal.kind,
    }
    return metrics, details


def _evaluate_config(cfg: Config, fold_caches: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_fold = []
    t0 = time.perf_counter()
    for fc in fold_caches:
        m, d = _eval_config_on_fold(cfg, fc)
        per_fold.append({"fold_id": fc["fold_id"], **m, "details": d})
    runtime = time.perf_counter() - t0

    devs = np.array([r["poisson_deviance"] for r in per_fold], dtype=float)
    wmse = np.array([r["weighted_mse_rate"] for r in per_fold], dtype=float)
    cals = np.array([r["calibration_ratio"] for r in per_fold], dtype=float)

    summary = {
        "mean_cv_poisson_deviance": float(np.mean(devs)),
        "std_cv_poisson_deviance": float(np.std(devs, ddof=0)),
        "mean_weighted_mse_rate": float(np.mean(wmse)),
        "calibration_ratio": float(np.mean(cals)),
        "runtime_seconds": float(runtime),
    }
    return summary, per_fold


def _screen_config(cfg: Config, fold_caches: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    # Use first fold for cheap screening.
    fc = fold_caches[0]
    t0 = time.perf_counter()
    m, _ = _eval_config_on_fold(cfg, fc)
    runtime = time.perf_counter() - t0
    return float(m["poisson_deviance"]), {"runtime_seconds": float(runtime)}


def _config_row(cfg: Config) -> dict[str, Any]:
    return {
        "config_id": cfg.config_id,
        "combiner_family": cfg.combiner_family,
        "base_pool_id": cfg.base_pool_id,
        "base_models": json.dumps(cfg.base_models),
        "calibration_type": cfg.calibration_type,
        "hyperparams": json.dumps(cfg.hyperparams, sort_keys=True),
    }


def _expand_for_deep_tuning(
    cfg: Config,
    random_state: int = 0,
    max_variants_per_seed: int = 180,
) -> list[Config]:
    variants: list[Config] = [cfg]
    rng = np.random.default_rng(random_state)

    fam = cfg.combiner_family
    hp = dict(cfg.hyperparams)
    cid_base = cfg.config_id

    def add_variant(hupd: dict[str, Any], suffix: str):
        nh = dict(hp)
        nh.update(hupd)
        variants.append(
            Config(
                config_id=f"{cid_base}_d{suffix}",
                combiner_family=fam,
                base_pool_id=cfg.base_pool_id,
                base_models=cfg.base_models,
                calibration_type=cfg.calibration_type,
                hyperparams=nh,
            )
        )

    if fam in {"convex", "logspace"}:
        for lam in [0.0, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]:
            add_variant({"lambda": lam}, f"lam{lam}")

    if fam == "stack_poisson":
        for a in [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
            for ic in [True, False]:
                add_variant({"alpha": a, "include_ctx": ic}, f"a{a}_c{int(ic)}")

    if fam == "power_mean":
        for p in [-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3]:
            for tr in [0.0, 0.05, 0.1, 0.2]:
                add_variant({"p": p, "trim": tr}, f"p{p}_t{tr}")

    if fam == "tree_poisson":
        for d in [2, 3, 4, 5]:
            for lr in [0.02, 0.05, 0.1]:
                for ne in [200, 400, 700]:
                    add_variant({"max_depth": d, "learning_rate": lr, "n_estimators": ne}, f"d{d}_lr{lr}_n{ne}")

    if fam == "residual_corr":
        for a in [0.01, 0.1, 1.0, 10.0, 30.0, 100.0]:
            add_variant({"alpha": a}, f"a{a}")

    if fam == "regime_blend":
        for nb in [3, 4, 5, 6]:
            for sz in [False, True]:
                for wt in ["convex", "nnls"]:
                    add_variant({"n_bins": nb, "use_shots_zero": sz, "weight_type": wt}, f"b{nb}_s{int(sz)}_{wt}")

    if fam == "moe":
        for ne in [2, 3, 4]:
            for var in ["A", "B", "C"]:
                for temp in [0.5, 0.7, 1.0, 1.3, 1.7]:
                    for l2 in [0.0, 1e-4, 1e-3, 1e-2, 1e-1]:
                        add_variant({"n_experts": ne, "variant": var, "temperature": temp, "l2": l2}, f"e{ne}_{var}_t{temp}_l{l2}")

    # calibrator variants for deep tune
    deep_vars = []
    for c in variants:
        for cal in ["none", "scalar", "piecewise_scalar", "isotonic"]:
            deep_vars.append(
                Config(
                    config_id=f"{c.config_id}_cal_{cal}",
                    combiner_family=c.combiner_family,
                    base_pool_id=c.base_pool_id,
                    base_models=c.base_models,
                    calibration_type=cal,
                    hyperparams=c.hyperparams,
                )
            )
    # Keep deep stage broad but bounded to finish in practical time.
    if len(deep_vars) > max_variants_per_seed:
        base_keep = [v for v in deep_vars if v.config_id.startswith(cfg.config_id) and "_d" not in v.config_id][:4]
        rest = [v for v in deep_vars if v not in base_keep]
        take = max(0, max_variants_per_seed - len(base_keep))
        if take > 0 and rest:
            idx = rng.choice(np.arange(len(rest)), size=min(take, len(rest)), replace=False)
            idx = np.sort(idx)
            pick = [rest[i] for i in idx]
        else:
            pick = []
        deep_vars = base_keep + pick
    return deep_vars


def run_ensemble_search(
    ev_df: pd.DataFrame,
    outputs_dir: Path,
    random_state: int = 0,
    screen_target: int = 350,
    full_target: int = 50,
    deep_target: int = 10,
    deep_max_configs: int = 800,
    deep_max_variants_per_seed: int = 180,
    enable_stability_tiebreak: bool = False,
    base_model_limit: int | None = None,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or LOGGER
    rng = np.random.default_rng(random_state)

    base_builders_all = get_model_builders(random_state=random_state)

    # Start from base models that were previously successful if available.
    metrics_path = outputs_dir / "metrics_summary.csv"
    ranked = []
    if metrics_path.exists():
        try:
            ms = pd.read_csv(metrics_path)
            ms = ms[(ms["status"] == "OK") & (~ms["method"].astype(str).str.startswith("ENSEMBLE_"))]
            ms = ms.sort_values("cv_poisson_deviance_mean")
            ranked = ms["method"].astype(str).tolist()
        except Exception as exc:
            log.warning("Could not read prior metrics_summary.csv: %s", exc)

    default_rank = [
        "POISSON_GLM_OFFSET_REG",
        "POISSON_GLM_OFFSET",
        "TWO_STAGE_SHOTS_XG",
        "HURDLE_XG",
        "RIDGE_RAPM_RATE_SOFTPLUS",
        "DEFENSE_ADJ_TWO_STEP",
        "ELASTICNET_RAPM_RATE_SOFTPLUS",
        "TWEEDIE_GLM_RATE",
        "BASELINE_LINE_MEAN_RATE",
    ]
    for m in default_rank:
        if m not in ranked and m in base_builders_all:
            ranked.append(m)

    # keep only available builders
    ranked = [m for m in ranked if m in base_builders_all]
    if base_model_limit is not None:
        ranked = ranked[: int(base_model_limit)]
    if len(ranked) < 3:
        raise RuntimeError("Need at least 3 base models for ensemble search")

    base_builders = {m: base_builders_all[m] for m in ranked}

    # global OOF for pool generation and cheap screening
    log.info("Building global OOF predictions for base model set: %s", ranked)
    global_oof = build_oof_predictions(ev_df, model_builders=base_builders, model_names=ranked, n_splits=5)
    global_oof.to_parquet(outputs_dir / "oof_predictions_ev.parquet", index=False)

    # rank by global OOF deviance
    dev_rank = []
    for m in ranked:
        col = f"mu_pred_total_{m}"
        d = poisson_deviance_safe(global_oof["y_true_total"].to_numpy(dtype=float), global_oof[col].to_numpy(dtype=float))
        dev_rank.append((m, d))
    dev_rank = sorted(dev_rank, key=lambda x: x[1])
    ranked_models = [m for m, _ in dev_rank]

    pools = _build_base_pool_sets(ranked_models, global_oof, random_state=random_state, max_random_per_size=50)
    log.info("Generated %s base pools", len(pools))

    configs_all = _generate_config_candidates(pools, random_state=random_state, max_configs=1500)
    if len(configs_all) < screen_target:
        screen_cfgs = configs_all
    else:
        idx = rng.choice(np.arange(len(configs_all)), size=screen_target, replace=False)
        idx = np.sort(idx)
        screen_cfgs = [configs_all[i] for i in idx]
    log.info("Screening %s configs", len(screen_cfgs))

    cache_root = outputs_dir / "ensemble_oof_cache"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Stage 1/2 caches for all ranked models with inner=3
    fold_cache3, valid_models = _build_cache_for_models(
        ev_df=ev_df,
        base_builders=base_builders,
        model_names=ranked_models,
        outer_splits=5,
        inner_splits=3,
        cache_dir=cache_root / "inner3",
    )
    log.info("Valid models after cache build: %s", valid_models)

    results_rows: list[dict[str, Any]] = []

    # Stage 1 screening
    screened: list[tuple[Config, float, float]] = []
    for cfg in screen_cfgs:
        t0 = time.perf_counter()
        try:
            cfg2 = Config(
                config_id=cfg.config_id,
                combiner_family=cfg.combiner_family,
                base_pool_id=cfg.base_pool_id,
                base_models=[m for m in cfg.base_models if m in valid_models],
                calibration_type=cfg.calibration_type,
                hyperparams=cfg.hyperparams,
            )
            if len(cfg2.base_models) < 2:
                raise RuntimeError("insufficient valid models")
            score, aux = _screen_config(cfg2, fold_cache3)
            runtime = float(time.perf_counter() - t0)
            screened.append((cfg2, score, runtime))
            row = {
                **_config_row(cfg2),
                "stage": "screen",
                "mean_cv_poisson_deviance": float(score),
                "std_cv_poisson_deviance": np.nan,
                "mean_weighted_mse_rate": np.nan,
                "calibration_ratio": np.nan,
                "runtime_seconds": runtime,
                "status": "OK",
                "notes": "",
            }
        except Exception as exc:
            runtime = float(time.perf_counter() - t0)
            row = {
                **_config_row(cfg),
                "stage": "screen",
                "mean_cv_poisson_deviance": np.nan,
                "std_cv_poisson_deviance": np.nan,
                "mean_weighted_mse_rate": np.nan,
                "calibration_ratio": np.nan,
                "runtime_seconds": runtime,
                "status": "FAILED",
                "notes": _config_runtime_note(exc),
            }
        results_rows.append(row)

    screened = sorted(screened, key=lambda x: x[1])
    top_full_cfgs = [x[0] for x in screened[: min(full_target, len(screened))]]
    log.info("Stage1 complete: %s OK configs, taking %s to full 5-fold", len(screened), len(top_full_cfgs))

    # Stage 2 full 5-fold
    full_scored: list[tuple[Config, float]] = []
    for cfg in top_full_cfgs:
        t0 = time.perf_counter()
        try:
            summary, _ = _evaluate_config(cfg, fold_cache3)
            full_scored.append((cfg, summary["mean_cv_poisson_deviance"]))
            row = {
                **_config_row(cfg),
                "stage": "full",
                "mean_cv_poisson_deviance": summary["mean_cv_poisson_deviance"],
                "std_cv_poisson_deviance": summary["std_cv_poisson_deviance"],
                "mean_weighted_mse_rate": summary["mean_weighted_mse_rate"],
                "calibration_ratio": summary["calibration_ratio"],
                "runtime_seconds": float(time.perf_counter() - t0),
                "status": "OK",
                "notes": "",
            }
        except Exception as exc:
            row = {
                **_config_row(cfg),
                "stage": "full",
                "mean_cv_poisson_deviance": np.nan,
                "std_cv_poisson_deviance": np.nan,
                "mean_weighted_mse_rate": np.nan,
                "calibration_ratio": np.nan,
                "runtime_seconds": float(time.perf_counter() - t0),
                "status": "FAILED",
                "notes": _config_runtime_note(exc),
            }
        results_rows.append(row)

    full_scored = sorted(full_scored, key=lambda x: x[1])
    top_deep_seed = [cfg for cfg, _ in full_scored[: min(deep_target, len(full_scored))]]
    log.info("Stage2 complete: taking %s configs to deep tuning", len(top_deep_seed))

    # Stage 3 deep tuning with expanded hyperparams and inner=4 cache on needed models only.
    deep_candidates: list[Config] = []
    for i, cfg in enumerate(top_deep_seed):
        deep_candidates.extend(
            _expand_for_deep_tuning(
                cfg,
                random_state=random_state + i + 1,
                max_variants_per_seed=deep_max_variants_per_seed,
            )
        )

    # dedupe by family/pool/cal/hp
    dedupe = {}
    for c in deep_candidates:
        key = (c.combiner_family, c.base_pool_id, c.calibration_type, json.dumps(c.hyperparams, sort_keys=True), tuple(c.base_models))
        dedupe[key] = c
    deep_candidates = list(dedupe.values())

    if len(deep_candidates) > deep_max_configs:
        keep: list[Config] = []
        seen_seed = set()
        for c in deep_candidates:
            seed = c.config_id.split("_d", 1)[0]
            if seed not in seen_seed:
                keep.append(c)
                seen_seed.add(seed)
        rest = [c for c in deep_candidates if c not in keep]
        need = max(0, deep_max_configs - len(keep))
        if need > 0 and rest:
            idx = rng.choice(np.arange(len(rest)), size=min(need, len(rest)), replace=False)
            idx = np.sort(idx)
            keep.extend([rest[i] for i in idx])
        deep_candidates = keep[:deep_max_configs]
    log.info("Deep candidate count after cap: %s", len(deep_candidates))

    needed_models = sorted({m for c in deep_candidates for m in c.base_models if m in valid_models})
    fold_cache4, valid4 = _build_cache_for_models(
        ev_df=ev_df,
        base_builders=base_builders,
        model_names=needed_models,
        outer_splits=5,
        inner_splits=4,
        cache_dir=cache_root / "inner4_top",
    )

    deep_scored: list[tuple[Config, float]] = []
    for i, cfg in enumerate(deep_candidates, start=1):
        t0 = time.perf_counter()
        cfg2 = Config(
            config_id=cfg.config_id,
            combiner_family=cfg.combiner_family,
            base_pool_id=cfg.base_pool_id,
            base_models=[m for m in cfg.base_models if m in valid4],
            calibration_type=cfg.calibration_type,
            hyperparams=cfg.hyperparams,
        )
        try:
            if len(cfg2.base_models) < 2:
                raise RuntimeError("insufficient valid models for deep")
            summary, _ = _evaluate_config(cfg2, fold_cache4)
            deep_scored.append((cfg2, summary["mean_cv_poisson_deviance"]))
            row = {
                **_config_row(cfg2),
                "stage": "deep",
                "mean_cv_poisson_deviance": summary["mean_cv_poisson_deviance"],
                "std_cv_poisson_deviance": summary["std_cv_poisson_deviance"],
                "mean_weighted_mse_rate": summary["mean_weighted_mse_rate"],
                "calibration_ratio": summary["calibration_ratio"],
                "runtime_seconds": float(time.perf_counter() - t0),
                "status": "OK",
                "notes": "",
            }
        except Exception as exc:
            row = {
                **_config_row(cfg2),
                "stage": "deep",
                "mean_cv_poisson_deviance": np.nan,
                "std_cv_poisson_deviance": np.nan,
                "mean_weighted_mse_rate": np.nan,
                "calibration_ratio": np.nan,
                "runtime_seconds": float(time.perf_counter() - t0),
                "status": "FAILED",
                "notes": _config_runtime_note(exc),
            }
        results_rows.append(row)
        if i % 50 == 0 or i == len(deep_candidates):
            log.info("Deep eval progress: %s/%s", i, len(deep_candidates))

    # Select winner from deep if available else full else screened.
    if deep_scored:
        winner_cfg, winner_dev = sorted(deep_scored, key=lambda x: x[1])[0]
        winner_stage = "deep"
    elif full_scored:
        winner_cfg, winner_dev = sorted(full_scored, key=lambda x: x[1])[0]
        winner_stage = "full"
    elif screened:
        winner_cfg, winner_dev, _ = sorted(screened, key=lambda x: x[1])[0]
        winner_stage = "screen"
    else:
        raise RuntimeError("No successful ensemble config found")

    # Optional tie-break by stability for near ties (<=0.3%).
    best_cfg = winner_cfg
    best_dev = winner_dev
    if enable_stability_tiebreak:
        candidate_rows = [r for r in results_rows if r["status"] == "OK" and r["stage"] in {winner_stage, "deep", "full"}]
        cand = pd.DataFrame(candidate_rows)
        cand = cand.sort_values("mean_cv_poisson_deviance")
        if len(cand) >= 2 and np.isfinite(cand.iloc[0]["mean_cv_poisson_deviance"]):
            d0 = float(cand.iloc[0]["mean_cv_poisson_deviance"])
            d1 = float(cand.iloc[1]["mean_cv_poisson_deviance"])
            if d0 > 0 and ((d1 - d0) / d0) <= 0.003:
                # Optional: expensive; disabled by default for search throughput.
                cands = []
                for _, row in cand.head(2).iterrows():
                    cid = row["config_id"]
                    match = None
                    for c in [winner_cfg] + [cfg for cfg, _ in deep_scored] + [cfg for cfg, _ in full_scored]:
                        if c.config_id == cid:
                            match = c
                            break
                    if match is None:
                        continue
                    try:
                        fm = FittedSearchModel(random_state=random_state, base_builders=base_builders, config=match).fit(ev_df)
                        top10 = compute_disparity_ratios(compute_standardized_strengths(fm, ev_df)).head(10)
                        teams = top10["team"].astype(str).tolist()
                        res = bootstrap_rank_stability(
                            model_factory=lambda m=match: FittedSearchModel(
                                random_state=random_state,
                                base_builders=base_builders,
                                config=m,
                            ),
                            ev_df=ev_df,
                            full_top10_teams=teams,
                            n_boot=5,
                            random_state=random_state,
                            show_progress=False,
                        )
                        cands.append((match, float(row["mean_cv_poisson_deviance"]), res.stability_score))
                    except Exception:
                        continue
                if len(cands) >= 2:
                    cands = sorted(cands, key=lambda x: (x[1], x[2]))
                    if abs(cands[1][1] - cands[0][1]) / max(cands[0][1], EPS) <= 0.003:
                        cands = sorted(cands, key=lambda x: (x[2], x[1]))
                    best_cfg = cands[0][0]
                    best_dev = cands[0][1]

    # Fit best on full EV and compute final top10.
    best_model = FittedSearchModel(random_state=random_state, base_builders=base_builders, config=best_cfg)
    best_model.fit(ev_df)
    strengths = compute_standardized_strengths(best_model, ev_df)
    top10 = compute_disparity_ratios(strengths).head(10).copy()
    top10["method"] = f"ENSEMBLE_SEARCH::{best_cfg.combiner_family}"

    # Save results
    results_df = pd.DataFrame(results_rows)
    results_df = results_df[
        [
            "config_id",
            "combiner_family",
            "base_pool_id",
            "base_models",
            "calibration_type",
            "hyperparams",
            "stage",
            "mean_cv_poisson_deviance",
            "std_cv_poisson_deviance",
            "mean_weighted_mse_rate",
            "calibration_ratio",
            "runtime_seconds",
            "status",
            "notes",
        ]
    ]
    results_df.to_csv(outputs_dir / "ensemble_search_results.csv", index=False)

    best_cfg_payload = {
        "config_id": best_cfg.config_id,
        "combiner_family": best_cfg.combiner_family,
        "base_pool_id": best_cfg.base_pool_id,
        "base_models": best_cfg.base_models,
        "calibration_type": best_cfg.calibration_type,
        "hyperparams": best_cfg.hyperparams,
        "mean_cv_poisson_deviance": float(best_dev),
        "search_counts": {
            "pools": len(pools),
            "screened": len(screen_cfgs),
            "full": len(top_full_cfgs),
            "deep": len(deep_candidates),
        },
    }
    with (outputs_dir / "ensemble_best_config.json").open("w", encoding="utf-8") as f:
        json.dump(best_cfg_payload, f, indent=2)

    top10 = top10[["rank", "team", "line1_strength_xg60", "line2_strength_xg60", "ratio", "method"]]
    top10.to_csv(outputs_dir / "final_top10.csv", index=False)
    top10.to_csv(outputs_dir / "submission_phase1b.csv", index=False)

    with (outputs_dir / "best_method.txt").open("w", encoding="utf-8") as f:
        f.write(f"best_method=ENSEMBLE_SEARCH::{best_cfg.combiner_family}\n")
        f.write(f"config_id={best_cfg.config_id}\n")
        f.write(f"mean_cv_poisson_deviance={best_dev}\n")

    # Append summary block to results readme.
    readme_path = outputs_dir / "README_RESULTS.md"
    extra = []
    extra.append("\n## Ensemble Search (Expanded)")
    extra.append(f"- pools generated: {len(pools)}")
    extra.append(f"- screened configs: {len(screen_cfgs)}")
    extra.append(f"- full 5-fold evaluated: {len(top_full_cfgs)}")
    extra.append(f"- deep tuned configs: {len(deep_candidates)}")
    extra.append(f"- winner: {best_cfg.combiner_family} ({best_cfg.config_id})")
    extra.append(f"- winner deviance: {best_dev}")
    extra.append("")

    top20 = (
        results_df[(results_df["status"] == "OK") & results_df["mean_cv_poisson_deviance"].notna()]
        .sort_values("mean_cv_poisson_deviance")
        .head(20)
    )
    extra.append("Top 20 configs:")
    extra.append("```")
    extra.append(top20.to_string(index=False))
    extra.append("```")
    extra_txt = "\n".join(extra)
    if readme_path.exists():
        readme_path.write_text(readme_path.read_text(encoding="utf-8") + "\n" + extra_txt + "\n", encoding="utf-8")
    else:
        readme_path.write_text("# Results\n\n" + extra_txt + "\n", encoding="utf-8")

    return {
        "best_config": best_cfg_payload,
        "top20": top20,
        "top10": top10,
        "results_df": results_df,
    }
