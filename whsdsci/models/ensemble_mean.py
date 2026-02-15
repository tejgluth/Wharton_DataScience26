from __future__ import annotations

import numpy as np
import pandas as pd

from whsdsci.ensemble.oof import clip_total_positive
from whsdsci.models.ensemble_base import EnsembleBaseModel


class EnsembleMeanModel(EnsembleBaseModel):
    name = "ENSEMBLE_MEAN_TOPK"

    def fit(self, train_df: pd.DataFrame):
        self._fit_base_models(train_df)
        m = len(self.base_model_names)
        self.weights_ = np.full(m, 1.0 / m, dtype=float)
        self.artifacts.details = {"weights": {k: float(v) for k, v in zip(self.base_model_names, self.weights_)}}
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        mat = self._predict_base_totals(df)
        mu = mat.mean(axis=1)
        return clip_total_positive(mu)
