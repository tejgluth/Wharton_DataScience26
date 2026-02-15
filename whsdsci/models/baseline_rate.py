from __future__ import annotations

import numpy as np
import pandas as pd

from whsdsci.models.base import BaseModel, EPS_RATE


class BaselineLineMeanRateModel(BaseModel):
    name = "BASELINE_LINE_MEAN_RATE"

    def fit(self, df: pd.DataFrame):
        d = df.copy()
        d["toi_hr"] = np.maximum(pd.to_numeric(d["toi_hr"], errors="coerce"), 1e-9)
        d["xg_for"] = np.clip(pd.to_numeric(d["xg_for"], errors="coerce"), 0, None)
        grp = d.groupby("off_unit", as_index=True).agg({"xg_for": "sum", "toi_hr": "sum"})
        self.r_off = (grp["xg_for"] / np.maximum(grp["toi_hr"], 1e-9)).to_dict()
        self.global_rate = float(d["xg_for"].sum() / np.maximum(d["toi_hr"].sum(), 1e-9))
        return self

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        rates = df["off_unit"].astype(str).map(self.r_off).fillna(self.global_rate).to_numpy(dtype=float)
        return np.clip(rates, EPS_RATE, None)
