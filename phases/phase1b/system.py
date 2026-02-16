from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from whsdsci.eval.metrics import poisson_deviance_safe
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


@dataclass
class BestConfig:
    config_id: str
    combiner_family: str
    base_pool_id: str
    base_models: list[str]
    calibration_type: str
    hyperparams: dict[str, Any]
    metadata: dict[str, Any]


def _coerce_payload(payload: dict[str, Any]) -> BestConfig:
    return BestConfig(
        config_id=str(payload["config_id"]),
        combiner_family=str(payload.get("combiner_family", "tree_poisson")),
        base_pool_id=str(payload.get("base_pool_id", "unknown_pool")),
        base_models=[str(x) for x in payload.get("base_models", [])],
        calibration_type=str(payload.get("calibration_type", "none")),
        hyperparams=dict(payload.get("hyperparams", {})),
        metadata={
            k: v
            for k, v in payload.items()
            if k not in {"config_id", "combiner_family", "base_pool_id", "base_models", "calibration_type", "hyperparams"}
        },
    )


def resolve_best_config(outputs_dir: Path, config_name: str | None = None) -> BestConfig:
    outputs_dir = Path(outputs_dir)
    if config_name and config_name.endswith(".json"):
        return _coerce_payload(json.loads(Path(config_name).read_text(encoding="utf-8")))

    candidate_files = [
        outputs_dir / "confirmed_best_config.json",
        outputs_dir / "ensemble_best_config.json",
    ]
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in candidate_files if p.exists()]

    if config_name:
        for payload in payloads:
            if str(payload.get("config_id")) == str(config_name):
                return _coerce_payload(payload)
        if re.match(r"^cfg_[A-Za-z0-9_\\.-]+$", str(config_name)):
            raise FileNotFoundError(f"Could not resolve best config id: {config_name}")

    if payloads:
        return _coerce_payload(payloads[0])

    best_txt = outputs_dir / "best_method.txt"
    if best_txt.exists():
        m = re.search(r"config_id=(cfg_[A-Za-z0-9_\\.-]+)", best_txt.read_text(encoding="utf-8"))
        if m:
            return resolve_best_config(outputs_dir=outputs_dir, config_name=m.group(1))
    raise FileNotFoundError("No best config artifact found in outputs/")


@dataclass
class CalibrationModel:
    kind: str
    params: dict

    def predict(self, mu_raw: np.ndarray, log_toi_hr: np.ndarray | None = None) -> np.ndarray:
        mu = np.clip(np.asarray(mu_raw, dtype=float), EPS, None)
        if self.kind == "none":
            return mu
        if self.kind == "scalar":
            s = float(self.params.get("scale", 1.0))
            return np.clip(s * mu, EPS, None)
        if self.kind == "piecewise_scalar":
            bins = np.asarray(self.params.get("bin_edges", []), dtype=float)
            scales = np.asarray(self.params.get("scales", []), dtype=float)
            if bins.size < 2 or scales.size == 0 or log_toi_hr is None:
                s = float(self.params.get("global_scale", 1.0))
                return np.clip(s * mu, EPS, None)
            lt = np.asarray(log_toi_hr, dtype=float)
            idx = np.digitize(lt, bins[1:-1], right=False)
            idx = np.clip(idx, 0, len(scales) - 1)
            return np.clip(mu * scales[idx], EPS, None)
        if self.kind == "isotonic":
            iso = self.params.get("iso")
            if iso is None:
                return mu
            return np.clip(np.asarray(iso.predict(mu), dtype=float), EPS, None)
        return mu


def _fit_scalar(y_true: np.ndarray, mu_raw: np.ndarray) -> float:
    y = np.clip(np.asarray(y_true, dtype=float), 0, None)
    mu = np.clip(np.asarray(mu_raw, dtype=float), EPS, None)
    s0 = float(np.sum(y) / max(float(np.sum(mu)), EPS))
    s0 = max(s0, EPS)

    def obj(s: float) -> float:
        return poisson_deviance_safe(y, np.clip(s * mu, EPS, None))

    lo = max(EPS, s0 * 0.1)
    hi = max(s0 * 10.0, lo * 1.1)
    res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    if res.success and np.isfinite(res.x):
        return float(max(res.x, EPS))
    return s0


def fit_calibrator(
    y_true: np.ndarray,
    mu_raw: np.ndarray,
    calibration_type: str,
    log_toi_hr: np.ndarray | None = None,
    n_bins: int = 4,
) -> CalibrationModel:
    y = np.clip(np.asarray(y_true, dtype=float), 0, None)
    mu = np.clip(np.asarray(mu_raw, dtype=float), EPS, None)
    ctype = str(calibration_type or "none")
    if ctype == "none":
        return CalibrationModel(kind="none", params={})
    if ctype == "scalar":
        return CalibrationModel(kind="scalar", params={"scale": _fit_scalar(y, mu)})
    if ctype == "piecewise_scalar":
        if log_toi_hr is None:
            return CalibrationModel(kind="scalar", params={"scale": _fit_scalar(y, mu)})
        lt = np.asarray(log_toi_hr, dtype=float)
        if len(lt) < n_bins * 10:
            return CalibrationModel(kind="scalar", params={"scale": _fit_scalar(y, mu)})
        edges = np.quantile(lt, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 3:
            return CalibrationModel(kind="scalar", params={"scale": _fit_scalar(y, mu)})
        idx = np.digitize(lt, edges[1:-1], right=False)
        nb = len(edges) - 1
        g = _fit_scalar(y, mu)
        scales = np.full(nb, g, dtype=float)
        for b in range(nb):
            m = idx == b
            if int(m.sum()) >= 20:
                scales[b] = _fit_scalar(y[m], mu[m])
        return CalibrationModel(
            kind="piecewise_scalar",
            params={"bin_edges": edges.tolist(), "scales": scales.tolist(), "global_scale": float(g)},
        )
    if ctype == "isotonic":
        iso = IsotonicRegression(y_min=EPS, increasing=True, out_of_bounds="clip")
        iso.fit(mu, y)
        return CalibrationModel(kind="isotonic", params={"iso": iso})
    return CalibrationModel(kind="none", params={})


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
            raise ValueError(f"Unsupported combiner_family for this frozen system: {self.config.combiner_family!r}")
        self.base_builders = get_best_base_model_builders(random_state=random_state)
        missing = [m for m in self.config.base_models if m not in self.base_builders]
        if missing:
            raise ValueError(f"Best config references unsupported base models: {missing}")

    def _fit_meta(self, X: np.ndarray, y: np.ndarray) -> None:
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
        preds = []
        for name in self.config.base_models:
            model = self.base_builders[name]()
            try:
                model.fit(d)
            except Exception as exc:
                LOGGER.warning("Base model %s failed; using fallback. error=%s", name, exc)
                model = _GlobalRateFallbackModel(random_state=self.random_state).fit(d)
            self.base_models_[name] = model
            preds.append(_clip_pos(model.predict_total(d)))
        P = np.column_stack(preds)

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
        return _clip_pos(self.calibrator_.predict(mu_raw, log_toi_hr=ctx["log_toi_hr"]))

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        mu = self.predict_total(df)
        toi = np.maximum(pd.to_numeric(df.get("toi_hr", 1.0), errors="coerce").to_numpy(dtype=float), EPS)
        return np.clip(mu / toi, 1e-12, None)

