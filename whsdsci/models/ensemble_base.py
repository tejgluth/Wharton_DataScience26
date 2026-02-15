from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from whsdsci.ensemble.oof import build_oof_predictions, clip_total_positive, dataset_key
from whsdsci.models.base import BaseModel, EPS_RATE, SkipModelError
from whsdsci.models.defense_two_step import DefenseAdjTwoStepModel
from whsdsci.models.hurdle_xg import HurdleXgModel
from whsdsci.models.poisson_glm_offset import PoissonGlmOffsetModel
from whsdsci.models.poisson_glm_offset_reg import PoissonGlmOffsetRegModel
from whsdsci.models.ridge_rapm import RidgeRapmSoftplusModel
from whsdsci.models.tweedie_glm import TweedieGlmRateModel
from whsdsci.models.two_stage_shots_xg import TwoStageShotsXgModel


@dataclass
class EnsembleArtifacts:
    base_model_names: list[str]
    details: dict


_OOF_CACHE: dict[tuple[str, tuple[str, ...], int], pd.DataFrame] = {}


def get_default_ensemble_base_builders(random_state: int = 0) -> dict[str, Callable[[], BaseModel]]:
    return {
        "POISSON_GLM_OFFSET_REG": lambda: PoissonGlmOffsetRegModel(random_state=random_state),
        "POISSON_GLM_OFFSET": lambda: PoissonGlmOffsetModel(random_state=random_state),
        "TWEEDIE_GLM_RATE": lambda: TweedieGlmRateModel(random_state=random_state),
        "TWO_STAGE_SHOTS_XG": lambda: TwoStageShotsXgModel(random_state=random_state),
        "HURDLE_XG": lambda: HurdleXgModel(random_state=random_state),
        "RIDGE_RAPM_RATE_SOFTPLUS": lambda: RidgeRapmSoftplusModel(random_state=random_state),
        "DEFENSE_ADJ_TWO_STEP": lambda: DefenseAdjTwoStepModel(random_state=random_state),
    }


class EnsembleBaseModel(BaseModel):
    name = "ENSEMBLE_BASE"

    def __init__(
        self,
        base_model_builders: dict[str, Callable[[], BaseModel]],
        base_model_names: list[str] | None = None,
        inner_splits: int = 3,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.base_model_builders = dict(base_model_builders)
        self.base_model_names = (
            [m for m in base_model_names if m in self.base_model_builders]
            if base_model_names
            else list(self.base_model_builders.keys())
        )
        self.inner_splits = inner_splits

        if not self.base_model_names:
            raise SkipModelError("No usable base models provided for ensemble")

        self.base_models_: dict[str, BaseModel] = {}
        self.artifacts = EnsembleArtifacts(base_model_names=self.base_model_names.copy(), details={})

    def _fit_base_models(self, train_df: pd.DataFrame) -> None:
        fitted = {}
        for method in self.base_model_names:
            model = self.base_model_builders[method]()
            model.fit(train_df)
            fitted[method] = model
        self.base_models_ = fitted

    def _predict_base_totals(self, df: pd.DataFrame, model_names: list[str] | None = None) -> np.ndarray:
        names = model_names or self.base_model_names
        if not self.base_models_:
            raise RuntimeError("Base models are not fitted")
        mats = []
        for method in names:
            mu = self.base_models_[method].predict_total(df)
            mats.append(clip_total_positive(mu))
        return np.column_stack(mats)

    def _build_inner_oof(self, train_df: pd.DataFrame, model_names: list[str] | None = None) -> pd.DataFrame:
        names = model_names or self.base_model_names
        key = (dataset_key(train_df), tuple(names), int(self.inner_splits))
        if key in _OOF_CACHE:
            return _OOF_CACHE[key].copy()

        builders = {m: self.base_model_builders[m] for m in names}
        oof = build_oof_predictions(train_df, model_builders=builders, model_names=names, n_splits=self.inner_splits)
        _OOF_CACHE[key] = oof.copy()
        return oof

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        mu = self.predict_total(df)
        toi = np.maximum(pd.to_numeric(df["toi_hr"], errors="coerce").to_numpy(dtype=float), 1e-9)
        return np.clip(mu / toi, EPS_RATE, None)
