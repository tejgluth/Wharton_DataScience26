from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.preprocessing import OneHotEncoder


EPS_RATE = 1e-12


class SkipModelError(RuntimeError):
    """Raise when optional model dependencies are unavailable."""


@dataclass
class ModelStatus:
    status: str = "OK"
    message: str = ""


class BaseModel:
    name = "BASE"

    def __init__(self, random_state: int = 0):
        self.random_state = random_state
        self.status = ModelStatus()

    def fit(self, df: pd.DataFrame) -> "BaseModel":
        raise NotImplementedError

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        toi = np.maximum(pd.to_numeric(df["toi_hr"], errors="coerce").to_numpy(dtype=float), 1e-9)
        rate = np.clip(self.predict_rate_hr(df), EPS_RATE, None)
        return rate * toi


class SparseDesignMixin:
    def _new_encoder(self) -> OneHotEncoder:
        try:
            return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
        except TypeError:
            return OneHotEncoder(handle_unknown="ignore", sparse=True)

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["off_unit"] = out["off_unit"].astype(str)
        out["def_unit"] = out["def_unit"].astype(str)
        out["is_home"] = pd.to_numeric(out["is_home"], errors="coerce").fillna(0).astype(float)
        out["toi_hr"] = np.maximum(pd.to_numeric(out["toi_hr"], errors="coerce"), 1e-9)
        return out

    def _build_X(self, df: pd.DataFrame, fit: bool = False):
        d = self._prepare_df(df)
        cat = d[["off_unit", "def_unit"]]
        if fit or getattr(self, "encoder", None) is None:
            self.encoder = self._new_encoder()
            Xcat = self.encoder.fit_transform(cat)
        else:
            Xcat = self.encoder.transform(cat)
        xhome = sparse.csr_matrix(d[["is_home"]].to_numpy(dtype=float))
        X = sparse.hstack([Xcat, xhome], format="csr")
        return X


def softplus(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def safe_rate_from_linear(y_lin: np.ndarray) -> np.ndarray:
    return np.clip(softplus(y_lin), EPS_RATE, None)
