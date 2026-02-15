from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from whsdsci.models.base import BaseModel, EPS_RATE


class TwoStageShotsXgModel(BaseModel):
    name = "TWO_STAGE_SHOTS_XG"

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["off_unit"] = d["off_unit"].astype(str)
        d["def_unit"] = d["def_unit"].astype(str)
        d["is_home"] = pd.to_numeric(d["is_home"], errors="coerce").fillna(0).astype(float)
        d["toi_hr"] = np.maximum(pd.to_numeric(d["toi_hr"], errors="coerce"), 1e-9)
        xg = d["xg_for"] if "xg_for" in d.columns else pd.Series(0.0, index=d.index)
        shots = d["shots_for"] if "shots_for" in d.columns else pd.Series(0.0, index=d.index)
        d["xg_for"] = np.clip(pd.to_numeric(xg, errors="coerce").fillna(0.0), 0, None)
        d["shots_for"] = np.clip(pd.to_numeric(shots, errors="coerce").fillna(0.0), 0, None)
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

        y_shots = d["shots_for"].to_numpy(dtype=float)
        exp = d["toi_hr"].to_numpy(dtype=float)
        self.global_shots_rate = float(np.clip(y_shots.sum() / np.maximum(exp.sum(), 1e-9), 1e-9, None))
        try:
            self.shots_res = sm.GLM(y_shots, X, family=sm.families.Poisson(), exposure=exp).fit(maxiter=200, disp=0)
        except Exception:
            self.shots_res = None

        sev_df = d[d["shots_for"] > 0].copy()
        if sev_df.empty:
            self.severity_res = None
            self.global_severity = float(np.clip(d["xg_for"].sum() / np.maximum(d["shots_for"].sum(), 1e-9), 1e-9, None))
            return self

        Xs = self._design(sev_df, fit=False)
        y_sev = np.clip((sev_df["xg_for"] / np.maximum(sev_df["shots_for"], 1e-9)).to_numpy(dtype=float), 1e-9, None)
        w = np.maximum(sev_df["shots_for"].to_numpy(dtype=float), 1.0)
        self.severity_res = sm.GLM(
            y_sev,
            Xs,
            family=sm.families.Gamma(sm.families.links.Log()),
            freq_weights=w,
        ).fit(maxiter=200, disp=0)
        self.global_severity = float(np.average(y_sev, weights=w))
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        d = self._prepare_df(df)
        X = self._design(d, fit=False)

        if self.shots_res is None:
            shots_hat = self.global_shots_rate * d["toi_hr"].to_numpy(dtype=float)
        else:
            shots_hat = self.shots_res.predict(X, exposure=d["toi_hr"].to_numpy(dtype=float))
        shots_hat = np.clip(np.asarray(shots_hat, dtype=float), 1e-9, None)

        if self.severity_res is None:
            sev_hat = np.full(len(d), self.global_severity, dtype=float)
        else:
            sev_hat = np.asarray(self.severity_res.predict(X), dtype=float)
            sev_hat = np.clip(sev_hat, 1e-9, None)

        mu = shots_hat * sev_hat
        return np.clip(mu, 1e-9, None)

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        d = self._prepare_df(df)
        mu = self.predict_total(d)
        rate = mu / np.maximum(d["toi_hr"].to_numpy(dtype=float), 1e-9)
        return np.clip(rate, EPS_RATE, None)
