from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression

from whsdsci.eval.metrics import poisson_deviance_safe


EPS = 1e-9


@dataclass
class CalibrationModel:
    kind: str
    params: dict

    def predict(self, mu_raw: np.ndarray, log_toi_hr: np.ndarray | None = None) -> np.ndarray:
        mu = np.clip(np.asarray(mu_raw, dtype=float), EPS, None)

        if self.kind == "none":
            return np.clip(mu, EPS, None)

        if self.kind == "scalar":
            s = float(self.params.get("scale", 1.0))
            return np.clip(s * mu, EPS, None)

        if self.kind == "piecewise_scalar":
            bins = np.asarray(self.params.get("bin_edges", []), dtype=float)
            scales = np.asarray(self.params.get("scales", []), dtype=float)
            if log_toi_hr is None or bins.size < 2 or scales.size == 0:
                s = float(self.params.get("global_scale", 1.0))
                return np.clip(s * mu, EPS, None)
            lt = np.asarray(log_toi_hr, dtype=float)
            idx = np.digitize(lt, bins[1:-1], right=False)
            idx = np.clip(idx, 0, len(scales) - 1)
            out = mu * scales[idx]
            return np.clip(out, EPS, None)

        if self.kind == "isotonic":
            iso = self.params.get("iso")
            if iso is None:
                return np.clip(mu, EPS, None)
            out = iso.predict(mu)
            return np.clip(np.asarray(out, dtype=float), EPS, None)

        return np.clip(mu, EPS, None)


def _fit_scalar(y: np.ndarray, mu_raw: np.ndarray) -> float:
    yv = np.clip(np.asarray(y, dtype=float), 0, None)
    mu = np.clip(np.asarray(mu_raw, dtype=float), EPS, None)

    s0 = float(np.sum(yv) / np.maximum(np.sum(mu), EPS))
    s0 = max(s0, EPS)

    def objective(s: float) -> float:
        return poisson_deviance_safe(yv, np.clip(s * mu, EPS, None))

    lo = max(EPS, s0 * 0.1)
    hi = max(s0 * 10.0, lo * 1.1)
    res = minimize_scalar(objective, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
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
    yv = np.clip(np.asarray(y_true, dtype=float), 0, None)
    mu = np.clip(np.asarray(mu_raw, dtype=float), EPS, None)

    ctype = calibration_type or "none"

    if ctype == "none":
        return CalibrationModel(kind="none", params={})

    if ctype == "scalar":
        s = _fit_scalar(yv, mu)
        return CalibrationModel(kind="scalar", params={"scale": float(s)})

    if ctype == "piecewise_scalar":
        if log_toi_hr is None:
            s = _fit_scalar(yv, mu)
            return CalibrationModel(kind="scalar", params={"scale": float(s)})

        lt = np.asarray(log_toi_hr, dtype=float)
        if len(lt) < n_bins * 10:
            s = _fit_scalar(yv, mu)
            return CalibrationModel(kind="scalar", params={"scale": float(s)})

        q = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(lt, q)
        edges = np.unique(edges)
        if len(edges) < 3:
            s = _fit_scalar(yv, mu)
            return CalibrationModel(kind="scalar", params={"scale": float(s)})

        idx = np.digitize(lt, edges[1:-1], right=False)
        nb = len(edges) - 1
        scales = np.ones(nb, dtype=float)
        global_s = _fit_scalar(yv, mu)
        for b in range(nb):
            m = idx == b
            if int(m.sum()) < 20:
                scales[b] = global_s
            else:
                scales[b] = _fit_scalar(yv[m], mu[m])

        return CalibrationModel(
            kind="piecewise_scalar",
            params={"bin_edges": edges.tolist(), "scales": scales.tolist(), "global_scale": float(global_s)},
        )

    if ctype == "isotonic":
        iso = IsotonicRegression(y_min=EPS, increasing=True, out_of_bounds="clip")
        iso.fit(mu, yv)
        return CalibrationModel(kind="isotonic", params={"iso": iso})

    return CalibrationModel(kind="none", params={})
