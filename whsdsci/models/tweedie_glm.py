from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import TweedieRegressor
from sklearn.model_selection import GroupKFold

from whsdsci.eval.metrics import poisson_deviance_safe
from whsdsci.models.base import BaseModel, SparseDesignMixin, EPS_RATE


class TweedieGlmRateModel(BaseModel, SparseDesignMixin):
    name = "TWEEDIE_GLM_RATE"

    def __init__(
        self,
        random_state: int = 0,
        power_grid: list[float] | None = None,
        alpha_grid: list[float] | None = None,
    ):
        super().__init__(random_state=random_state)
        self.power_grid = power_grid or [1.1, 1.3, 1.5, 1.7, 1.9]
        self.alpha_grid = alpha_grid or [0.0, 0.1, 1.0, 10.0]
        self.tune_max_rows = 12000
        self.tune_splits = 2

    def _fit_once(self, df: pd.DataFrame, power: float, alpha: float):
        d = self._prepare_df(df)
        X = self._build_X(d, fit=True)
        y = (np.clip(pd.to_numeric(d["xg_for"], errors="coerce"), 0, None) / np.maximum(d["toi_hr"], 1e-9)).to_numpy(dtype=float)
        w = np.maximum(pd.to_numeric(d["toi_hr"], errors="coerce").to_numpy(dtype=float), 1e-9)
        model = TweedieRegressor(power=power, alpha=alpha, link="log", max_iter=1000)
        model.fit(X, y, sample_weight=w)
        self.model = model
        self.best_power = power
        self.best_alpha = alpha
        return self

    def _score_combo(self, df: pd.DataFrame, power: float, alpha: float) -> float:
        groups = df["game_id"].astype(str).to_numpy()
        uniq = np.unique(groups)
        n_splits = min(self.tune_splits, len(uniq))
        if n_splits < 2:
            return float("inf")
        splitter = GroupKFold(n_splits=n_splits)
        idx = np.arange(len(df))
        scores = []
        for tr, va in splitter.split(idx, groups=groups):
            dtr = df.iloc[tr]
            dva = df.iloc[va]
            tmp = TweedieGlmRateModel(random_state=self.random_state, power_grid=[power], alpha_grid=[alpha])
            tmp._fit_once(dtr, power, alpha)
            pred_tot = tmp.predict_total(dva)
            scores.append(poisson_deviance_safe(dva["xg_for"].to_numpy(dtype=float), pred_tot))
        return float(np.mean(scores)) if scores else float("inf")

    def fit(self, df: pd.DataFrame):
        d = self._prepare_df(df)
        if len(d) > self.tune_max_rows:
            d_tune = d.sample(n=self.tune_max_rows, random_state=self.random_state).reset_index(drop=True)
        else:
            d_tune = d
        best = (self.power_grid[0], self.alpha_grid[0])
        best_score = float("inf")
        for p in self.power_grid:
            for a in self.alpha_grid:
                s = self._score_combo(d_tune, p, a)
                if s < best_score:
                    best_score = s
                    best = (p, a)
        self._fit_once(d, *best)
        self.nested_cv_score = best_score
        return self

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        d = self._prepare_df(df)
        X = self._build_X(d, fit=False)
        rate = np.asarray(self.model.predict(X), dtype=float)
        return np.clip(rate, EPS_RATE, None)
