from __future__ import annotations

import numpy as np
import pandas as pd

from whsdsci.models.base import BaseModel, EPS_RATE


class DefenseAdjTwoStepModel(BaseModel):
    name = "DEFENSE_ADJ_TWO_STEP"

    def fit(self, df: pd.DataFrame):
        d = df.copy()
        d["toi_hr"] = np.maximum(pd.to_numeric(d["toi_hr"], errors="coerce"), 1e-9)
        d["xg_for"] = np.clip(pd.to_numeric(d["xg_for"], errors="coerce"), 0, None)
        d["xg_rate_hr"] = d["xg_for"] / d["toi_hr"]

        def_grp = d.groupby("def_unit", as_index=True).agg({"xg_for": "sum", "toi_hr": "sum"})
        self.r_def = (def_grp["xg_for"] / np.maximum(def_grp["toi_hr"], 1e-9)).to_dict()
        self.global_def = float(d["xg_for"].sum() / np.maximum(d["toi_hr"].sum(), 1e-9))

        d["r_def"] = d["def_unit"].astype(str).map(self.r_def).fillna(self.global_def)
        d["resid"] = d["xg_rate_hr"] - d["r_def"]

        off_stats = d.groupby("off_unit", as_index=True).apply(
            lambda g: float(np.average(g["resid"], weights=np.maximum(g["toi_hr"], 1e-9)))
        )
        self.r_resid = off_stats.to_dict()
        self.global_resid = float(np.average(d["resid"], weights=np.maximum(d["toi_hr"], 1e-9)))
        return self

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        rdef = df["def_unit"].astype(str).map(self.r_def).fillna(self.global_def).to_numpy(dtype=float)
        rres = df["off_unit"].astype(str).map(self.r_resid).fillna(self.global_resid).to_numpy(dtype=float)
        return np.clip(rdef + rres, EPS_RATE, None)
