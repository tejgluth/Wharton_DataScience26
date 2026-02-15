from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

from whsdsci.models.base import BaseModel, SparseDesignMixin, EPS_RATE


class HurdleXgModel(BaseModel, SparseDesignMixin):
    name = "HURDLE_XG"

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        d = super()._prepare_df(df)
        xg = d["xg_for"] if "xg_for" in d.columns else pd.Series(0.0, index=d.index)
        d["xg_for"] = np.clip(pd.to_numeric(xg, errors="coerce").fillna(0.0), 0, None)
        d["log_toi_hr"] = np.log(np.maximum(d["toi_hr"], 1e-9))
        return d

    def _design_gamma(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        cat = pd.get_dummies(df[["off_unit", "def_unit"]], prefix=["off", "def"], dtype=float)
        X = pd.concat([df[["is_home"]].astype(float), cat], axis=1)
        X = sm.add_constant(X, has_constant="add")
        if fit:
            self.gamma_cols_ = X.columns.tolist()
            return X
        return X.reindex(columns=self.gamma_cols_, fill_value=0.0)

    def fit(self, df: pd.DataFrame):
        d = self._prepare_df(df)

        Xlog = self._build_X(d, fit=True)
        y_pos = (d["xg_for"] > 0).astype(int).to_numpy()
        uniq = np.unique(y_pos)
        if len(uniq) < 2:
            self.logit = None
            self.constant_p_pos = float(uniq[0])
        else:
            self.logit = LogisticRegression(max_iter=2000, random_state=self.random_state)
            self.logit.fit(Xlog, y_pos)
            self.constant_p_pos = None

        pos = d[d["xg_for"] > 0].copy()
        if pos.empty:
            self.gamma_res = None
            self.global_pos_mean = 1e-9
            return self

        Xg = self._design_gamma(pos, fit=True)
        yg = np.clip(pos["xg_for"].to_numpy(dtype=float), 1e-9, None)
        off = pos["log_toi_hr"].to_numpy(dtype=float)
        self.gamma_res = sm.GLM(
            yg,
            Xg,
            family=sm.families.Gamma(sm.families.links.Log()),
            offset=off,
        ).fit(maxiter=200, disp=0)
        self.global_pos_mean = float(np.mean(yg))
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        d = self._prepare_df(df)
        if self.logit is None:
            p_pos = np.full(len(d), self.constant_p_pos, dtype=float)
        else:
            Xlog = self._build_X(d, fit=False)
            p_pos = self.logit.predict_proba(Xlog)[:, 1]

        if self.gamma_res is None:
            mu_pos = np.full(len(d), self.global_pos_mean, dtype=float)
        else:
            Xg = self._design_gamma(d, fit=False)
            mu_pos = np.asarray(self.gamma_res.predict(Xg, offset=d["log_toi_hr"].to_numpy(dtype=float)), dtype=float)
            mu_pos = np.clip(mu_pos, 1e-9, None)

        mu = p_pos * mu_pos
        return np.clip(mu, 1e-9, None)

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        d = self._prepare_df(df)
        mu = self.predict_total(d)
        rate = mu / np.maximum(d["toi_hr"].to_numpy(dtype=float), 1e-9)
        return np.clip(rate, EPS_RATE, None)
