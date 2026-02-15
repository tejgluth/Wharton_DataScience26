from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge

from whsdsci.ensemble.oof import clip_total_positive
from whsdsci.eval.metrics import poisson_deviance_safe
from whsdsci.models.base import EPS_RATE
from whsdsci.models.ensemble_base import EnsembleBaseModel


EPS = 1e-9


def _power_mean(M: np.ndarray, p: float, trim_frac: float = 0.0) -> np.ndarray:
    X = np.clip(np.asarray(M, dtype=float), EPS, None)
    if trim_frac > 0:
        lo = np.quantile(X, trim_frac, axis=1, keepdims=True)
        hi = np.quantile(X, 1.0 - trim_frac, axis=1, keepdims=True)
        keep = (X >= lo) & (X <= hi)
        # keep at least one per row
        keep = np.where(keep.any(axis=1, keepdims=True), keep, np.ones_like(keep, dtype=bool))
        X = np.where(keep, X, np.nan)

    if p == 0:
        out = np.exp(np.nanmean(np.log(np.clip(X, EPS, None)), axis=1))
    else:
        out = np.power(np.nanmean(np.power(np.clip(X, EPS, None), p), axis=1), 1.0 / p)
    return clip_total_positive(np.nan_to_num(out, nan=EPS, posinf=1e9, neginf=EPS))


class NNLSBlendModel(EnsembleBaseModel):
    name = "ENSEMBLE_NNLS"

    def fit(self, train_df: pd.DataFrame):
        oof = self._build_inner_oof(train_df)
        cols = [f"mu_pred_total_{m}" for m in self.base_model_names]
        P = oof[cols].to_numpy(dtype=float)
        y = np.clip(oof["y_true_total"].to_numpy(dtype=float), 0, None)
        w, _ = nnls(P, y)
        if float(np.sum(w)) <= 0:
            w = np.ones(len(cols), dtype=float)
        self.weights_ = w
        self._fit_base_models(train_df)
        self.artifacts.details = {"weights": {k: float(v) for k, v in zip(self.base_model_names, self.weights_)}}
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        P = self._predict_base_totals(df)
        return clip_total_positive(P @ self.weights_)


class LogSpaceBlendModel(EnsembleBaseModel):
    name = "ENSEMBLE_LOGSPACE"

    def __init__(self, *args, lambda_l2: float = 1e-3, **kwargs):
        super().__init__(*args, **kwargs)
        self.lambda_l2 = lambda_l2

    def fit(self, train_df: pd.DataFrame):
        oof = self._build_inner_oof(train_df)
        cols = [f"mu_pred_total_{m}" for m in self.base_model_names]
        P = np.log(np.clip(oof[cols].to_numpy(dtype=float), EPS, None))
        y = np.clip(oof["y_true_total"].to_numpy(dtype=float), 0, None)

        m = P.shape[1]
        x0 = np.concatenate([np.full(m, 1.0 / m), np.array([0.0])])

        def objective(x: np.ndarray) -> float:
            a = np.clip(x[:-1], 0, 1)
            s = float(np.sum(a))
            if s <= 0:
                a = np.full(m, 1.0 / m)
            else:
                a = a / s
            b = float(x[-1])
            mu = np.exp(b + P @ a)
            mu = clip_total_positive(mu)
            dev = poisson_deviance_safe(y, mu)
            return float(dev + self.lambda_l2 * np.sum(a * a))

        constraints = [{"type": "eq", "fun": lambda x: np.sum(np.clip(x[:-1], 0, 1)) - 1.0}]
        bounds = [(0.0, 1.0)] * m + [(-5.0, 5.0)]
        res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 500})
        x = res.x if res.success else x0
        a = np.clip(x[:-1], 0, 1)
        a = a / max(float(np.sum(a)), EPS)
        b = float(x[-1])

        self.a_ = a
        self.b_ = b
        self._fit_base_models(train_df)
        self.artifacts.details = {
            "lambda_l2": float(self.lambda_l2),
            "intercept": float(self.b_),
            "weights": {k: float(v) for k, v in zip(self.base_model_names, self.a_)},
        }
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        P = np.log(np.clip(self._predict_base_totals(df), EPS, None))
        mu = np.exp(self.b_ + P @ self.a_)
        return clip_total_positive(mu)


class PowerMeanBlendModel(EnsembleBaseModel):
    name = "ENSEMBLE_POWERMEAN"

    def __init__(self, *args, p: float = 1.0, trim_frac: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.trim_frac = trim_frac

    def fit(self, train_df: pd.DataFrame):
        self._fit_base_models(train_df)
        self.artifacts.details = {"p": float(self.p), "trim_frac": float(self.trim_frac)}
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        P = self._predict_base_totals(df)
        return _power_mean(P, p=self.p, trim_frac=self.trim_frac)


class MedianBlendModel(EnsembleBaseModel):
    name = "ENSEMBLE_MEDIAN"

    def __init__(self, *args, trim_frac: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.trim_frac = trim_frac

    def fit(self, train_df: pd.DataFrame):
        self._fit_base_models(train_df)
        self.artifacts.details = {"trim_frac": float(self.trim_frac)}
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        P = np.clip(self._predict_base_totals(df), EPS, None)
        if self.trim_frac > 0:
            lo = np.quantile(P, self.trim_frac, axis=1, keepdims=True)
            hi = np.quantile(P, 1.0 - self.trim_frac, axis=1, keepdims=True)
            keep = (P >= lo) & (P <= hi)
            keep = np.where(keep.any(axis=1, keepdims=True), keep, np.ones_like(keep, dtype=bool))
            P = np.where(keep, P, np.nan)
            mu = np.nanmedian(P, axis=1)
        else:
            mu = np.median(P, axis=1)
        return clip_total_positive(np.nan_to_num(mu, nan=EPS, posinf=1e9, neginf=EPS))


class TreeStackPoissonModel(EnsembleBaseModel):
    name = "ENSEMBLE_TREE_POISSON"

    def __init__(self, *args, max_depth: int = 3, learning_rate: float = 0.05, max_iter: int = 300, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def _meta_features(self, P: np.ndarray, df: pd.DataFrame) -> np.ndarray:
        logP = np.log(np.clip(P, EPS, None))
        lt = np.log(np.maximum(pd.to_numeric(df.get("toi_hr", 1.0), errors="coerce"), EPS)).to_numpy(dtype=float)
        ih = pd.to_numeric(df.get("is_home", 0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return np.column_stack([logP, lt, ih])

    def fit(self, train_df: pd.DataFrame):
        oof = self._build_inner_oof(train_df)
        cols = [f"mu_pred_total_{m}" for m in self.base_model_names]
        P = oof[cols].to_numpy(dtype=float)

        train_with_row = train_df.copy()
        train_with_row["row_id"] = train_df.index.to_numpy()
        aligned = oof[["row_id"]].merge(train_with_row, on="row_id", how="left")

        X = self._meta_features(P, aligned)
        y = np.clip(oof["y_true_total"].to_numpy(dtype=float), 0, None)

        model = HistGradientBoostingRegressor(
            loss="poisson",
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        model.fit(X, y)

        self.meta_ = model
        self._fit_base_models(train_df)
        self.artifacts.details = {
            "backend": "sklearn_histgbr_poisson",
            "max_depth": int(self.max_depth),
            "learning_rate": float(self.learning_rate),
            "max_iter": int(self.max_iter),
        }
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        P = self._predict_base_totals(df)
        X = self._meta_features(P, df)
        mu = self.meta_.predict(X)
        return clip_total_positive(mu)


class ResidualCorrectionModel(EnsembleBaseModel):
    name = "ENSEMBLE_RESIDUAL_CORR"

    def __init__(self, *args, alpha: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    def fit(self, train_df: pd.DataFrame):
        oof = self._build_inner_oof(train_df)
        cols = [f"mu_pred_total_{m}" for m in self.base_model_names]
        P = np.clip(oof[cols].to_numpy(dtype=float), EPS, None)
        y = np.clip(oof["y_true_total"].to_numpy(dtype=float), 0, None)

        baseline_idx = 0
        if "POISSON_GLM_OFFSET_REG" in self.base_model_names:
            baseline_idx = self.base_model_names.index("POISSON_GLM_OFFSET_REG")
        mu0 = P[:, baseline_idx]

        resid_t = np.log((y + EPS) / (mu0 + EPS))
        X = np.log(P)

        model = Ridge(alpha=self.alpha, random_state=self.random_state)
        model.fit(X, resid_t)

        self.baseline_idx_ = baseline_idx
        self.resid_model_ = model
        self._fit_base_models(train_df)
        self.artifacts.details = {
            "baseline_model": self.base_model_names[baseline_idx],
            "alpha": float(self.alpha),
            "coef": [float(c) for c in model.coef_],
            "intercept": float(model.intercept_),
        }
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        P = np.clip(self._predict_base_totals(df), EPS, None)
        mu0 = P[:, self.baseline_idx_]
        X = np.log(P)
        corr = np.exp(self.resid_model_.predict(X))
        mu = mu0 * corr
        return clip_total_positive(mu)


class RegimeBlendModel(EnsembleBaseModel):
    name = "ENSEMBLE_REGIME_BLEND"

    def __init__(self, *args, n_bins: int = 4, method: str = "convex", use_shots_zero: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_bins = n_bins
        self.method = method
        self.use_shots_zero = use_shots_zero

    def _regime_keys(self, df: pd.DataFrame, bin_edges: np.ndarray | None = None):
        lt = np.log(np.maximum(pd.to_numeric(df.get("toi_hr", 1.0), errors="coerce"), EPS)).to_numpy(dtype=float)
        if bin_edges is None:
            q = np.linspace(0, 1, self.n_bins + 1)
            edges = np.quantile(lt, q)
            edges = np.unique(edges)
            if len(edges) < 3:
                edges = np.array([lt.min() - 1e-9, lt.max() + 1e-9])
        else:
            edges = bin_edges
        bid = np.digitize(lt, edges[1:-1], right=False)
        if self.use_shots_zero:
            s = pd.to_numeric(df.get("shots_for", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
            sz = (s <= 0).astype(int)
            key = bid.astype(int) * 10 + sz.astype(int)
        else:
            key = bid.astype(int)
        return key, edges

    def _fit_weights(self, P: np.ndarray, y: np.ndarray) -> np.ndarray:
        m = P.shape[1]
        if self.method == "nnls":
            w, _ = nnls(P, y)
            if float(np.sum(w)) <= 0:
                w = np.ones(m, dtype=float) / m
            return w

        x0 = np.full(m, 1.0 / m, dtype=float)

        def objective(w):
            ww = np.clip(np.asarray(w, dtype=float), 0, 1)
            s = float(np.sum(ww))
            ww = ww / max(s, EPS)
            mu = clip_total_positive(P @ ww)
            return poisson_deviance_safe(y, mu)

        cons = [{"type": "eq", "fun": lambda w: np.sum(np.clip(w, 0, 1)) - 1.0}]
        bnds = [(0.0, 1.0)] * m
        res = minimize(objective, x0, method="SLSQP", bounds=bnds, constraints=cons)
        if not res.success:
            return x0
        w = np.clip(np.asarray(res.x, dtype=float), 0, 1)
        return w / max(float(np.sum(w)), EPS)

    def fit(self, train_df: pd.DataFrame):
        oof = self._build_inner_oof(train_df)
        cols = [f"mu_pred_total_{m}" for m in self.base_model_names]
        P = oof[cols].to_numpy(dtype=float)
        y = np.clip(oof["y_true_total"].to_numpy(dtype=float), 0, None)

        train_with_row = train_df.copy()
        train_with_row["row_id"] = train_df.index.to_numpy()
        aligned = oof[["row_id"]].merge(train_with_row, on="row_id", how="left")
        key, edges = self._regime_keys(aligned, bin_edges=None)

        weights = {}
        global_w = self._fit_weights(P, y)
        for k in np.unique(key):
            m = key == k
            if int(m.sum()) < 30:
                weights[int(k)] = global_w
            else:
                weights[int(k)] = self._fit_weights(P[m], y[m])

        self.regime_weights_ = {int(k): np.asarray(v, dtype=float) for k, v in weights.items()}
        self.bin_edges_ = np.asarray(edges, dtype=float)
        self.global_w_ = global_w
        self._fit_base_models(train_df)
        self.artifacts.details = {
            "n_bins": int(self.n_bins),
            "method": self.method,
            "use_shots_zero": bool(self.use_shots_zero),
            "regimes": {str(k): [float(x) for x in v] for k, v in self.regime_weights_.items()},
        }
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        P = self._predict_base_totals(df)
        key, _ = self._regime_keys(df, bin_edges=self.bin_edges_)
        mu = np.zeros(len(df), dtype=float)
        for i, k in enumerate(key):
            w = self.regime_weights_.get(int(k), self.global_w_)
            mu[i] = float(np.dot(P[i], w))
        return clip_total_positive(mu)
