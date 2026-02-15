from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GroupKFold

from whsdsci.ensemble.oof import clip_total_positive
from whsdsci.eval.metrics import poisson_deviance_safe
from whsdsci.models.ensemble_base import EnsembleBaseModel


class EnsembleStackPoissonModel(EnsembleBaseModel):
    name = "ENSEMBLE_STACK_POISSON"

    def __init__(
        self,
        *args,
        alpha_grid: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.alpha_grid = alpha_grid or [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0]

    def _drop_collinear(self, X: np.ndarray, names: list[str], threshold: float = 0.9999):
        keep_idx: list[int] = []
        for i in range(X.shape[1]):
            xi = X[:, i]
            keep = True
            for j in keep_idx:
                xj = X[:, j]
                if np.std(xi) <= 0 or np.std(xj) <= 0:
                    corr = 1.0
                else:
                    corr = float(np.corrcoef(xi, xj)[0, 1])
                if np.isfinite(corr) and abs(corr) > threshold:
                    keep = False
                    break
            if keep:
                keep_idx.append(i)
        if not keep_idx:
            keep_idx = [0]
        return np.asarray(keep_idx, dtype=int), [names[i] for i in keep_idx]

    def _cv_score_alpha(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray, alpha: float) -> float:
        uniq = np.unique(groups)
        n_splits = min(3, len(uniq))
        if n_splits < 2:
            model = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=1000)
            model.fit(X, y)
            pred = clip_total_positive(model.predict(X))
            return poisson_deviance_safe(y, pred)

        splitter = GroupKFold(n_splits=n_splits)
        rows = np.arange(len(y))
        scores = []
        for tr, va in splitter.split(rows, groups=groups):
            m = PoissonRegressor(alpha=alpha, fit_intercept=True, max_iter=1000)
            m.fit(X[tr], y[tr])
            pred = clip_total_positive(m.predict(X[va]))
            scores.append(poisson_deviance_safe(y[va], pred))
        return float(np.mean(scores))

    def fit(self, train_df: pd.DataFrame):
        oof = self._build_inner_oof(train_df)
        cols = [f"mu_pred_total_{m}" for m in self.base_model_names]
        M = oof[cols].to_numpy(dtype=float)
        X = np.log(clip_total_positive(M))
        y = np.clip(oof["y_true_total"].to_numpy(dtype=float), 0, None)
        groups = oof["game_id"].astype(str).to_numpy()

        keep_idx, keep_names = self._drop_collinear(X, self.base_model_names)
        Xk = X[:, keep_idx]

        best_alpha = self.alpha_grid[0]
        best_score = float("inf")
        for alpha in self.alpha_grid:
            s = self._cv_score_alpha(Xk, y, groups, alpha)
            if s < best_score:
                best_score = s
                best_alpha = alpha

        meta = PoissonRegressor(alpha=best_alpha, fit_intercept=True, max_iter=1000)
        meta.fit(Xk, y)

        self.keep_idx_ = keep_idx
        self.keep_names_ = keep_names
        self.best_alpha_ = best_alpha
        self.meta_ = meta

        self._fit_base_models(train_df)
        self.artifacts.details = {
            "alpha": float(best_alpha),
            "inner_cv_deviance": float(best_score),
            "meta_intercept": float(meta.intercept_),
            "meta_coef": {k: float(v) for k, v in zip(self.keep_names_, meta.coef_)},
        }
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        M = self._predict_base_totals(df)
        X = np.log(clip_total_positive(M))
        Xk = X[:, self.keep_idx_]
        mu = self.meta_.predict(Xk)
        return clip_total_positive(mu)
