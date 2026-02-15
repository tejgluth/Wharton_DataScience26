from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from whsdsci.ensemble.oof import clip_total_positive
from whsdsci.models.base import SkipModelError
from whsdsci.models.ensemble_base import EnsembleBaseModel


class EnsembleFoldAvgModel(EnsembleBaseModel):
    name = "ENSEMBLE_FOLDAVG"

    def __init__(self, *args, base_model_name: str | None = None, n_folds: int = 5, **kwargs):
        super().__init__(*args, **kwargs)
        self.base_model_name = base_model_name
        self.n_folds = n_folds

    def fit(self, train_df: pd.DataFrame):
        if self.base_model_name is None:
            self.base_model_name = (
                "POISSON_GLM_OFFSET_REG"
                if "POISSON_GLM_OFFSET_REG" in self.base_model_builders
                else self.base_model_names[0]
            )
        if self.base_model_name not in self.base_model_builders:
            raise SkipModelError(f"FoldAvg base model unavailable: {self.base_model_name}")

        groups = train_df["game_id"].astype(str).to_numpy()
        uniq = np.unique(groups)
        n_splits = min(self.n_folds, len(uniq))

        models = []
        if n_splits < 2:
            m = self.base_model_builders[self.base_model_name]()
            m.fit(train_df)
            models.append(m)
        else:
            rows = np.arange(len(train_df))
            splitter = GroupKFold(n_splits=n_splits)
            for tr, _ in splitter.split(rows, groups=groups):
                dtr = train_df.iloc[tr].reset_index(drop=True)
                m = self.base_model_builders[self.base_model_name]()
                m.fit(dtr)
                models.append(m)

        self.fold_models_ = models
        self.artifacts.details = {
            "base_model": self.base_model_name,
            "n_models": len(models),
            "n_folds": int(n_splits),
        }
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        if not getattr(self, "fold_models_", None):
            raise RuntimeError("FoldAvg ensemble is not fitted")
        pred = [clip_total_positive(m.predict_total(df)) for m in self.fold_models_]
        mu = np.mean(np.column_stack(pred), axis=1)
        return clip_total_positive(mu)
