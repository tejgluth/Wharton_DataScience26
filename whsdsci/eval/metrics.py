from __future__ import annotations

import logging
import numpy as np
from sklearn.metrics import mean_poisson_deviance


LOGGER = logging.getLogger(__name__)
EPS_PRED = 1e-9
EPS_RATE = 1e-12


def clip_domain(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    neg_true = int((yt < 0).sum())
    nonpos_pred = int((yp <= 0).sum())
    if neg_true > 0:
        LOGGER.warning("Clipping %s negative y_true values to zero", neg_true)
    yt = np.clip(yt, 0, None)
    yp = np.clip(yp, EPS_PRED, None)
    return yt, yp, {"neg_true_clipped": neg_true, "nonpos_pred_clipped": nonpos_pred}


def poisson_deviance_safe(y_true: np.ndarray, y_pred_total: np.ndarray) -> float:
    yt, yp, _ = clip_domain(y_true, y_pred_total)
    return float(mean_poisson_deviance(yt, yp))


def weighted_mse_rate(y_rate: np.ndarray, pred_rate: np.ndarray, weights: np.ndarray) -> float:
    y = np.asarray(y_rate, dtype=float)
    p = np.asarray(pred_rate, dtype=float)
    w = np.asarray(weights, dtype=float)
    w = np.clip(w, 1e-12, None)
    return float(np.average((y - p) ** 2, weights=w))


def mae_total(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(yt - yp)))


def calibration_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    den = float(np.sum(yt))
    if den <= 0:
        return float("nan")
    return float(np.sum(yp) / den)
