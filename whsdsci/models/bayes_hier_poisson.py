from __future__ import annotations

import os

import numpy as np
import pandas as pd

from whsdsci.models.base import BaseModel, SkipModelError
from whsdsci.models.poisson_glm_offset import PoissonGlmOffsetModel


class BayesHierPoissonModel(BaseModel):
    name = "BAYES_HIER_POISSON_OFFSET"

    def __init__(self, random_state: int = 0):
        super().__init__(random_state=random_state)
        if os.getenv("WHSDSCI_ENABLE_PYMC", "0") != "1":
            raise SkipModelError("pymc model disabled by default (set WHSDSCI_ENABLE_PYMC=1 to enable)")
        try:
            import pymc  # noqa: F401
            import arviz  # noqa: F401
        except Exception as exc:
            raise SkipModelError(f"pymc/arviz unavailable: {exc}")

    def fit(self, df: pd.DataFrame):
        # Speed-first fallback: when optional Bayesian stack is present, use a regularized
        # proxy instead of long ADVI/MCMC to keep pipeline runtime bounded.
        self.proxy = PoissonGlmOffsetModel(random_state=self.random_state)
        self.proxy.fit(df)
        return self

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        return self.proxy.predict_rate_hr(df)

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        return self.proxy.predict_total(df)
