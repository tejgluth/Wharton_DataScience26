from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import GroupKFold

from whsdsci.eval.metrics import poisson_deviance_safe
from whsdsci.models.base import BaseModel, EPS_RATE


class PoissonGlmOffsetRegModel(BaseModel):
    name = "POISSON_GLM_OFFSET_REG"

    def __init__(
        self,
        random_state: int = 0,
        alpha_grid: list[float] | None = None,
        l1_grid: list[float] | None = None,
    ):
        super().__init__(random_state=random_state)
        self.alpha_grid = alpha_grid or [0.0, 0.1, 0.3, 1.0]
        self.l1_grid = l1_grid or [0.0, 0.5, 1.0]
        self.tune_max_rows = 3000
        self.tune_splits = 2

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["off_unit"] = d["off_unit"].astype(str)
        d["def_unit"] = d["def_unit"].astype(str)
        d["is_home"] = pd.to_numeric(d["is_home"], errors="coerce").fillna(0).astype(float)
        d["toi_hr"] = np.maximum(pd.to_numeric(d["toi_hr"], errors="coerce"), 1e-9)
        xg = d["xg_for"] if "xg_for" in d.columns else pd.Series(0.0, index=d.index)
        d["xg_for"] = np.clip(pd.to_numeric(xg, errors="coerce").fillna(0.0), 0, None)
        return d

    def _design(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        d = self._prepare_df(df)
        cat = pd.get_dummies(d[["off_unit", "def_unit"]], prefix=["off", "def"], dtype=float)
        X = pd.concat([d[["is_home"]].astype(float), cat], axis=1)
        X = sm.add_constant(X, has_constant="add")
        if fit:
            self.columns_ = X.columns.tolist()
            return X
        return X.reindex(columns=self.columns_, fill_value=0.0)

    def _fit_once(self, df: pd.DataFrame, alpha: float, l1_wt: float):
        d = self._prepare_df(df)
        X = self._design(d, fit=True)
        y = d["xg_for"].to_numpy(dtype=float)
        exp = d["toi_hr"].to_numpy(dtype=float)
        glm = sm.GLM(y, X, family=sm.families.Poisson(), exposure=exp)

        if alpha == 0.0 and l1_wt == 0.0:
            res = glm.fit(maxiter=200, disp=0)
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = glm.fit_regularized(alpha=alpha, L1_wt=l1_wt, maxiter=30)

        self.res = res
        self.best_alpha = alpha
        self.best_l1_wt = l1_wt
        return self

    def _score_combo(self, df: pd.DataFrame, alpha: float, l1_wt: float) -> float:
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
            tmp = PoissonGlmOffsetRegModel(random_state=self.random_state, alpha_grid=[alpha], l1_grid=[l1_wt])
            try:
                tmp._fit_once(dtr, alpha, l1_wt)
                pred = tmp.predict_total(dva)
                s = poisson_deviance_safe(dva["xg_for"].to_numpy(dtype=float), pred)
                scores.append(s)
            except Exception:
                scores.append(float("inf"))
        return float(np.mean(scores)) if scores else float("inf")

    def fit(self, df: pd.DataFrame):
        d = self._prepare_df(df)
        if len(d) > self.tune_max_rows:
            d_tune = d.sample(n=self.tune_max_rows, random_state=self.random_state).reset_index(drop=True)
        else:
            d_tune = d
        best = (self.alpha_grid[0], self.l1_grid[0])
        best_score = float("inf")
        combos = [(a, l1) for a in self.alpha_grid for l1 in self.l1_grid]
        if len(d) > 1000:
            # Keep representative points from the requested grid for runtime control.
            combos = [(0.0, 0.0), (0.1, 0.0), (0.3, 0.5), (1.0, 1.0)]
        for a, l1 in combos:
            s = self._score_combo(d_tune, a, l1)
            if s < best_score:
                best_score = s
                best = (a, l1)
        self._fit_once(d, *best)
        self.nested_cv_score = best_score
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        d = self._prepare_df(df)
        X = self._design(d, fit=False)
        mu = self.res.predict(X, exposure=d["toi_hr"].to_numpy(dtype=float))
        mu = np.clip(np.asarray(mu, dtype=float), 1e-9, None)
        return mu

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        d = self._prepare_df(df)
        mu = self.predict_total(d)
        rate = mu / np.maximum(d["toi_hr"].to_numpy(dtype=float), 1e-9)
        return np.clip(rate, EPS_RATE, None)
