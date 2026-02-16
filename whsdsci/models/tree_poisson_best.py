from __future__ import annotations

from pathlib import Path
from typing import Callable

import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from whsdsci.best_config import BestConfig, resolve_best_config
from whsdsci.calibration import fit_calibrator
from whsdsci.models.base import BaseModel
from whsdsci.models.defense_two_step import DefenseAdjTwoStepModel
from whsdsci.models.hurdle_xg import HurdleXgModel
from whsdsci.models.poisson_glm_offset import PoissonGlmOffsetModel
from whsdsci.models.poisson_glm_offset_reg import PoissonGlmOffsetRegModel
from whsdsci.models.ridge_rapm import RidgeRapmSoftplusModel
from whsdsci.models.tweedie_glm import TweedieGlmRateModel
from whsdsci.models.two_stage_shots_xg import TwoStageShotsXgModel


EPS = 1e-9
LOGGER = logging.getLogger(__name__)


def get_best_base_model_builders(random_state: int = 0) -> dict[str, Callable[[], BaseModel]]:
    return {
        "POISSON_GLM_OFFSET": lambda: PoissonGlmOffsetModel(random_state=random_state),
        "TWEEDIE_GLM_RATE": lambda: TweedieGlmRateModel(random_state=random_state),
        "TWO_STAGE_SHOTS_XG": lambda: TwoStageShotsXgModel(random_state=random_state),
        "POISSON_GLM_OFFSET_REG": lambda: PoissonGlmOffsetRegModel(random_state=random_state),
        "HURDLE_XG": lambda: HurdleXgModel(random_state=random_state),
        "RIDGE_RAPM_RATE_SOFTPLUS": lambda: RidgeRapmSoftplusModel(random_state=random_state),
        "DEFENSE_ADJ_TWO_STEP": lambda: DefenseAdjTwoStepModel(random_state=random_state),
    }


def _clip_pos(x: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(x, dtype=float), EPS, None)


def _context_from_df(
    df: pd.DataFrame,
    default_shots_for: float,
    xg_per_shot: float,
    mu_proxy: np.ndarray,
) -> dict[str, np.ndarray]:
    toi = np.maximum(pd.to_numeric(df.get("toi_hr", 1.0), errors="coerce").to_numpy(dtype=float), EPS)
    log_toi = np.log(toi)
    is_home = pd.to_numeric(df.get("is_home", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if "shots_for" in df.columns:
        shots = pd.to_numeric(df["shots_for"], errors="coerce").fillna(default_shots_for).to_numpy(dtype=float)
    else:
        shots = np.asarray(mu_proxy, dtype=float) / max(float(xg_per_shot), 1e-6)
    shots = np.clip(shots, 0.0, None)
    return {
        "log_toi_hr": log_toi,
        "is_home": is_home,
        "shots_zero": (shots <= 0).astype(float),
    }


def _build_meta_features(P: np.ndarray, ctx: dict[str, np.ndarray]) -> np.ndarray:
    return np.column_stack([np.log(_clip_pos(P)), ctx["log_toi_hr"], ctx["is_home"], ctx["shots_zero"]])


def _ensure_variation(mu_raw: np.ndarray, P: np.ndarray) -> np.ndarray:
    mu = _clip_pos(mu_raw)
    if mu.size <= 1 or float(np.std(mu)) > 1e-12:
        return mu
    anchor = np.mean(_clip_pos(P), axis=1)
    if float(np.std(anchor)) <= 1e-12:
        return mu
    scale = float(np.mean(mu) / max(float(np.mean(anchor)), EPS))
    return _clip_pos(anchor * scale)


class _GlobalRateFallbackModel(BaseModel):
    name = "GLOBAL_RATE_FALLBACK"

    def fit(self, df: pd.DataFrame) -> "_GlobalRateFallbackModel":
        d = df.copy()
        y = np.clip(pd.to_numeric(d.get("xg_for", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float), 0, None)
        toi = np.maximum(pd.to_numeric(d.get("toi_hr", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float), EPS)
        self.rate_ = float(np.sum(y) / max(float(np.sum(toi)), EPS))
        return self

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        return np.full(len(df), max(float(getattr(self, "rate_", 0.1)), 1e-12), dtype=float)

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        toi = np.maximum(pd.to_numeric(df.get("toi_hr", 1.0), errors="coerce").fillna(1.0).to_numpy(dtype=float), EPS)
        return np.clip(self.predict_rate_hr(df) * toi, EPS, None)


class TreePoissonBestModel(BaseModel):
    name = "TREE_POISSON_BEST"

    def __init__(
        self,
        config: BestConfig | None = None,
        outputs_dir: Path | None = None,
        random_state: int = 0,
    ):
        super().__init__(random_state=random_state)
        self.outputs_dir = Path(outputs_dir) if outputs_dir is not None else Path("outputs")
        self.config = config or resolve_best_config(self.outputs_dir)
        if self.config.combiner_family != "tree_poisson":
            raise ValueError(
                f"Best config combiner_family={self.config.combiner_family!r} is unsupported by TreePoissonBestModel"
            )
        self.base_builders = get_best_base_model_builders(random_state=random_state)
        missing = [m for m in self.config.base_models if m not in self.base_builders]
        if missing:
            raise ValueError(f"Best config references unsupported base models: {missing}")

    def _fit_meta(self, X: np.ndarray, y: np.ndarray):
        hp = self.config.hyperparams
        backend = str(hp.get("backend", "auto"))
        max_depth = int(hp.get("max_depth", 3))
        learning_rate = float(hp.get("learning_rate", 0.08))
        n_estimators = int(hp.get("n_estimators", 200))

        self.meta_backend_ = "sklearn_histgbr"
        self.meta_model_ = None
        if backend in {"auto", "xgboost"}:
            try:
                from xgboost import XGBRegressor

                self.meta_model_ = XGBRegressor(
                    objective="count:poisson",
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    subsample=float(hp.get("subsample", 0.9)),
                    colsample_bytree=float(hp.get("colsample_bytree", 0.9)),
                    reg_lambda=float(hp.get("reg_lambda", 1.0)),
                    reg_alpha=float(hp.get("reg_alpha", 0.0)),
                    max_delta_step=float(hp.get("max_delta_step", 0.0)),
                    random_state=self.random_state,
                    tree_method=str(hp.get("tree_method", "hist")),
                    n_jobs=1,
                )
                self.meta_model_.fit(X, y)
                self.meta_backend_ = "xgboost"
                return
            except Exception:
                self.meta_model_ = None

        self.meta_model_ = HistGradientBoostingRegressor(
            loss="poisson",
            max_depth=max_depth,
            learning_rate=learning_rate,
            max_iter=n_estimators,
            random_state=self.random_state,
        )
        self.meta_model_.fit(X, y)

    def fit(self, df: pd.DataFrame) -> "TreePoissonBestModel":
        d = df.copy().reset_index(drop=True)
        y = _clip_pos(pd.to_numeric(d["xg_for"], errors="coerce").fillna(0.0).to_numpy(dtype=float))

        shots_vec = pd.to_numeric(d.get("shots_for", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        self.default_shots_for_ = float(np.nanmedian(np.clip(shots_vec, 0.0, None)))
        xg_vec = pd.to_numeric(d.get("xg_for", 0.0), errors="coerce").fillna(0.0).to_numpy(dtype=float)
        shots_sum = float(np.clip(shots_vec, 0.0, None).sum())
        xg_sum = float(np.clip(xg_vec, 0.0, None).sum())
        self.xg_per_shot_ = float(xg_sum / shots_sum) if shots_sum > EPS else 0.08

        self.base_models_ = {}
        P_parts = []
        for name in self.config.base_models:
            model = self.base_builders[name]()
            try:
                model.fit(d)
            except Exception as exc:
                LOGGER.warning("Base model %s failed during fit; using global-rate fallback. error=%s", name, exc)
                model = _GlobalRateFallbackModel(random_state=self.random_state).fit(d)
            self.base_models_[name] = model
            P_parts.append(_clip_pos(model.predict_total(d)))
        P = np.column_stack(P_parts)

        ctx = _context_from_df(d, self.default_shots_for_, self.xg_per_shot_, mu_proxy=np.mean(P, axis=1))
        X = _build_meta_features(P, ctx)
        self._fit_meta(X, y)

        mu_train_raw = _ensure_variation(_clip_pos(self.meta_model_.predict(X)), P)
        self.calibrator_ = fit_calibrator(
            y_true=y,
            mu_raw=mu_train_raw,
            calibration_type=self.config.calibration_type,
            log_toi_hr=ctx["log_toi_hr"],
            n_bins=int(self.config.hyperparams.get("cal_bins", 4)),
        )
        return self

    def _predict_base(self, df: pd.DataFrame) -> np.ndarray:
        mats = []
        for name in self.config.base_models:
            mats.append(_clip_pos(self.base_models_[name].predict_total(df)))
        return np.column_stack(mats)

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        d = df.copy()
        P = self._predict_base(d)
        ctx = _context_from_df(d, self.default_shots_for_, self.xg_per_shot_, mu_proxy=np.mean(P, axis=1))
        X = _build_meta_features(P, ctx)
        mu_raw = _ensure_variation(_clip_pos(self.meta_model_.predict(X)), P)
        mu = self.calibrator_.predict(mu_raw, log_toi_hr=ctx["log_toi_hr"])
        return _clip_pos(mu)

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        mu = self.predict_total(df)
        toi = np.maximum(pd.to_numeric(df.get("toi_hr", 1.0), errors="coerce").to_numpy(dtype=float), EPS)
        return np.clip(mu / toi, 1e-12, None)
