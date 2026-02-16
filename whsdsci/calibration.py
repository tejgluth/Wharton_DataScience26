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
            out = iso.predict(mu)
            return np.clip(np.asarray(out, dtype=float), EPS, None)
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

