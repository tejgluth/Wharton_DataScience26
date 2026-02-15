from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from whsdsci.ensemble.oof import clip_total_positive
from whsdsci.eval.metrics import poisson_deviance_safe
from whsdsci.models.ensemble_base import EnsembleBaseModel


class EnsembleConvexModel(EnsembleBaseModel):
    name = "ENSEMBLE_CONVEX_TOPK"

    def __init__(
        self,
        *args,
        lambda_grid: list[float] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.lambda_grid = lambda_grid or [0.0, 1e-4, 1e-3, 1e-2]

    def _solve_weights(self, M: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
        m = M.shape[1]
        x0 = np.full(m, 1.0 / m, dtype=float)

        def objective(w: np.ndarray) -> float:
            mu = clip_total_positive(M @ w)
            dev = poisson_deviance_safe(y, mu)
            return float(dev + lam * np.sum(w * w))

        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        bounds = [(0.0, 1.0)] * m

        res = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        if not res.success:
            return x0
        w = np.asarray(res.x, dtype=float)
        w = np.clip(w, 0.0, 1.0)
        s = float(np.sum(w))
        if s <= 0:
            return x0
        return w / s

    def fit(self, train_df: pd.DataFrame):
        oof = self._build_inner_oof(train_df)
        cols = [f"mu_pred_total_{m}" for m in self.base_model_names]
        M = oof[cols].to_numpy(dtype=float)
        y = np.clip(oof["y_true_total"].to_numpy(dtype=float), 0, None)

        best_lam = self.lambda_grid[0]
        best_w = np.full(M.shape[1], 1.0 / M.shape[1], dtype=float)
        best_score = float("inf")

        for lam in self.lambda_grid:
            w = self._solve_weights(M, y, lam)
            mu = clip_total_positive(M @ w)
            score = poisson_deviance_safe(y, mu)
            if score < best_score:
                best_score = score
                best_w = w
                best_lam = lam

        self.weights_ = best_w
        self.best_lambda_ = best_lam
        self.inner_oof_deviance_ = best_score

        self._fit_base_models(train_df)
        self.artifacts.details = {
            "lambda": float(best_lam),
            "inner_oof_deviance": float(best_score),
            "weights": {k: float(v) for k, v in zip(self.base_model_names, self.weights_)},
        }
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        M = self._predict_base_totals(df)
        mu = M @ self.weights_
        return clip_total_positive(mu)
