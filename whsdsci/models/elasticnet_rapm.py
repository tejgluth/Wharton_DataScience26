from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GroupKFold

from whsdsci.eval.metrics import poisson_deviance_safe
from whsdsci.models.base import BaseModel, SparseDesignMixin, safe_rate_from_linear


class ElasticNetRapmSoftplusModel(BaseModel, SparseDesignMixin):
    name = "ELASTICNET_RAPM_RATE_SOFTPLUS"

    def __init__(
        self,
        random_state: int = 0,
        alpha_grid: list[float] | None = None,
        l1_grid: list[float] | None = None,
    ):
        super().__init__(random_state=random_state)
        self.alpha_grid = alpha_grid or [0.001, 0.01, 0.1, 1.0]
        self.l1_grid = l1_grid or [0.1, 0.5, 0.9]
        self.tune_max_rows = 12000
        self.tune_splits = 2

    def _fit_once(self, df: pd.DataFrame, alpha: float, l1_ratio: float):
        X = self._build_X(df, fit=True)
        y = (np.clip(pd.to_numeric(df["xg_for"], errors="coerce"), 0, None) / np.maximum(df["toi_hr"], 1e-9)).to_numpy(dtype=float)
        w = np.maximum(pd.to_numeric(df["toi_hr"], errors="coerce").to_numpy(dtype=float), 1e-9)

        model = ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=5000,
            random_state=self.random_state,
            fit_intercept=True,
        )
        try:
            model.fit(X, y, sample_weight=w)
        except TypeError:
            model.fit(X, y)

        self.model = model
        self.best_alpha = alpha
        self.best_l1_ratio = l1_ratio
        return self

    def _score_combo(self, df: pd.DataFrame, alpha: float, l1_ratio: float) -> float:
        groups = df["game_id"].astype(str).to_numpy()
        uniq = np.unique(groups)
        if len(uniq) < 2:
            return float("inf")
        n_splits = min(self.tune_splits, len(uniq))
        if n_splits < 2:
            return float("inf")
        splitter = GroupKFold(n_splits=n_splits)
        idx = np.arange(len(df))
        scores = []
        for tr, va in splitter.split(idx, groups=groups):
            dtr = df.iloc[tr]
            dva = df.iloc[va]
            tmp = ElasticNetRapmSoftplusModel(random_state=self.random_state, alpha_grid=[alpha], l1_grid=[l1_ratio])
            tmp._fit_once(dtr, alpha, l1_ratio)
            pred_tot = tmp.predict_total(dva)
            score = poisson_deviance_safe(dva["xg_for"].to_numpy(dtype=float), pred_tot)
            scores.append(score)
        return float(np.mean(scores)) if scores else float("inf")

    def fit(self, df: pd.DataFrame):
        d = self._prepare_df(df)
        if len(d) > self.tune_max_rows:
            d_tune = d.sample(n=self.tune_max_rows, random_state=self.random_state).reset_index(drop=True)
        else:
            d_tune = d
        best_combo = (self.alpha_grid[0], self.l1_grid[0])
        best_score = float("inf")
        for alpha in self.alpha_grid:
            for l1 in self.l1_grid:
                s = self._score_combo(d_tune, alpha, l1)
                if s < best_score:
                    best_score = s
                    best_combo = (alpha, l1)

        self._fit_once(d, *best_combo)
        self.nested_cv_score = best_score
        return self

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        d = self._prepare_df(df)
        X = self._build_X(d, fit=False)
        y_lin = self.model.predict(X)
        return safe_rate_from_linear(y_lin)
