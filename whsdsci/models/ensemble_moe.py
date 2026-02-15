from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from whsdsci.ensemble.oof import clip_total_positive
from whsdsci.eval.metrics import poisson_deviance_safe
from whsdsci.models.base import SkipModelError
from whsdsci.models.ensemble_base import EnsembleBaseModel


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


class EnsembleMoEModel(EnsembleBaseModel):
    name = "ENSEMBLE_MOE_2EXPERT"

    def __init__(
        self,
        *args,
        l2_grid: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.l2_grid = l2_grid or [0.0, 1e-3, 1e-2]

    def _gating_features(self, df: pd.DataFrame) -> np.ndarray:
        toi_raw = df["toi_hr"] if "toi_hr" in df.columns else pd.Series(1.0, index=df.index)
        toi = np.maximum(pd.to_numeric(toi_raw, errors="coerce"), 1e-9)
        log_toi = np.log(toi)
        home_raw = df["is_home"] if "is_home" in df.columns else pd.Series(0.0, index=df.index)
        shots_raw = df["shots_for"] if "shots_for" in df.columns else pd.Series(0.0, index=df.index)
        is_home = pd.to_numeric(home_raw, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        shots = pd.to_numeric(shots_raw, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        shots_zero = (shots <= 0).astype(float)
        F = np.column_stack([np.ones(len(df)), log_toi.to_numpy(dtype=float), is_home, shots_zero])
        return F

    def _fit_theta(self, F: np.ndarray, mu1: np.ndarray, mu2: np.ndarray, y: np.ndarray, l2: float) -> tuple[np.ndarray, float]:
        p = F.shape[1]
        x0 = np.zeros(p, dtype=float)

        def objective(theta: np.ndarray) -> float:
            w = _sigmoid(F @ theta)
            mu = clip_total_positive(w * mu1 + (1.0 - w) * mu2)
            dev = poisson_deviance_safe(y, mu)
            return float(dev + l2 * np.sum(theta * theta))

        res = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 500})
        theta = res.x if res.success else x0
        score = objective(theta)
        return theta, float(score)

    def fit(self, train_df: pd.DataFrame):
        if len(self.base_model_names) < 2:
            raise SkipModelError("MoE requires at least two base models")
        self.expert_names_ = self.base_model_names[:2]

        oof = self._build_inner_oof(train_df, model_names=self.expert_names_)
        mu1 = oof[f"mu_pred_total_{self.expert_names_[0]}"].to_numpy(dtype=float)
        mu2 = oof[f"mu_pred_total_{self.expert_names_[1]}"].to_numpy(dtype=float)
        y = np.clip(oof["y_true_total"].to_numpy(dtype=float), 0, None)

        train_with_row = train_df.copy()
        train_with_row["row_id"] = train_df.index.to_numpy()
        aligned = oof[["row_id"]].merge(train_with_row, on="row_id", how="left")
        F = self._gating_features(aligned)

        best_l2 = self.l2_grid[0]
        best_theta, best_score = self._fit_theta(F, mu1, mu2, y, best_l2)
        for l2 in self.l2_grid[1:]:
            theta, score = self._fit_theta(F, mu1, mu2, y, l2)
            if score < best_score:
                best_theta, best_score, best_l2 = theta, score, l2

        self.theta_ = best_theta
        self.best_l2_ = best_l2

        self._fit_base_models(train_df)
        self.artifacts.details = {
            "experts": self.expert_names_,
            "l2": float(best_l2),
            "theta": [float(x) for x in self.theta_],
            "inner_oof_objective": float(best_score),
        }
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        M = self._predict_base_totals(df, model_names=self.expert_names_)
        mu1 = M[:, 0]
        mu2 = M[:, 1]
        F = self._gating_features(df)
        w = _sigmoid(F @ self.theta_)
        mu = w * mu1 + (1.0 - w) * mu2
        return clip_total_positive(mu)
