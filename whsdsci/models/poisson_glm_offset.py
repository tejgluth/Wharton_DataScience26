from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from whsdsci.models.base import BaseModel, EPS_RATE


class PoissonGlmOffsetModel(BaseModel):
    name = "POISSON_GLM_OFFSET"

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

    def fit(self, df: pd.DataFrame):
        d = self._prepare_df(df)
        X = self._design(d, fit=True)
        y = d["xg_for"].to_numpy(dtype=float)
        exposure = d["toi_hr"].to_numpy(dtype=float)
        glm = sm.GLM(y, X, family=sm.families.Poisson(), exposure=exposure)
        self.res = glm.fit(maxiter=200, disp=0)
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
