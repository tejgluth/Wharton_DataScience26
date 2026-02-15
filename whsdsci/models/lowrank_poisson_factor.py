from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from whsdsci.eval.metrics import poisson_deviance_safe
from whsdsci.models.base import BaseModel, EPS_RATE, SkipModelError


class LowRankPoissonFactorModel(BaseModel):
    name = "LOWRANK_POISSON_FACTOR"

    def __init__(
        self,
        random_state: int = 0,
        k_grid: list[int] | None = None,
        reg_grid: list[float] | None = None,
        epochs: int = 60,
        lr: float = 0.05,
    ):
        super().__init__(random_state=random_state)
        if os.getenv("WHSDSCI_ENABLE_TORCH", "0") != "1":
            raise SkipModelError("torch model disabled by default (set WHSDSCI_ENABLE_TORCH=1 to enable)")
        self.k_grid = k_grid or [2, 4, 8]
        self.reg_grid = reg_grid or [0.01, 0.1, 1.0]
        self.epochs = epochs
        self.lr = lr

        try:
            import torch  # noqa: F401
        except Exception as exc:
            raise SkipModelError(f"torch unavailable: {exc}")

    def _prepare_df(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["off_unit"] = d["off_unit"].astype(str)
        d["def_unit"] = d["def_unit"].astype(str)
        d["is_home"] = pd.to_numeric(d["is_home"], errors="coerce").fillna(0).astype(float)
        d["toi_hr"] = np.maximum(pd.to_numeric(d["toi_hr"], errors="coerce"), 1e-9)
        d["xg_for"] = np.clip(pd.to_numeric(d["xg_for"], errors="coerce"), 0, None)
        return d

    def _encode(self, df: pd.DataFrame, fit: bool = False):
        if fit:
            off_levels = ["__UNK__"] + sorted(df["off_unit"].unique().tolist())
            def_levels = ["__UNK__"] + sorted(df["def_unit"].unique().tolist())
            self.off_to_ix = {k: i for i, k in enumerate(off_levels)}
            self.def_to_ix = {k: i for i, k in enumerate(def_levels)}

        off_ix = df["off_unit"].map(self.off_to_ix).fillna(0).astype(int).to_numpy()
        def_ix = df["def_unit"].map(self.def_to_ix).fillna(0).astype(int).to_numpy()
        is_home = df["is_home"].to_numpy(dtype=float)
        log_toi = np.log(np.maximum(df["toi_hr"].to_numpy(dtype=float), 1e-9))
        y = df["xg_for"].to_numpy(dtype=float)
        return off_ix, def_ix, is_home, log_toi, y

    def _train_torch(self, df: pd.DataFrame, k: int, reg: float):
        import torch

        d = self._prepare_df(df)
        off_ix, def_ix, is_home, log_toi, y = self._encode(d, fit=True)

        device = torch.device("cpu")
        g = torch.Generator(device="cpu")
        g.manual_seed(self.random_state)

        off_ix_t = torch.tensor(off_ix, dtype=torch.long, device=device)
        def_ix_t = torch.tensor(def_ix, dtype=torch.long, device=device)
        is_home_t = torch.tensor(is_home, dtype=torch.float32, device=device)
        log_toi_t = torch.tensor(log_toi, dtype=torch.float32, device=device)
        y_t = torch.tensor(y, dtype=torch.float32, device=device)

        n_off = len(self.off_to_ix)
        n_def = len(self.def_to_ix)

        self.b0 = torch.nn.Parameter(torch.zeros(1, device=device))
        self.bh = torch.nn.Parameter(torch.zeros(1, device=device))
        self.off_b = torch.nn.Parameter(torch.zeros(n_off, device=device))
        self.def_b = torch.nn.Parameter(torch.zeros(n_def, device=device))
        self.off_e = torch.nn.Parameter(torch.normal(0.0, 0.01, size=(n_off, k), generator=g, device=device))
        self.def_e = torch.nn.Parameter(torch.normal(0.0, 0.01, size=(n_def, k), generator=g, device=device))

        params = [self.b0, self.bh, self.off_b, self.def_b, self.off_e, self.def_e]
        opt = torch.optim.Adam(params, lr=self.lr)

        for _ in range(self.epochs):
            opt.zero_grad()
            lin = (
                log_toi_t
                + self.b0
                + self.bh * is_home_t
                + self.off_b[off_ix_t]
                + self.def_b[def_ix_t]
                + (self.off_e[off_ix_t] * self.def_e[def_ix_t]).sum(dim=1)
            )
            mu = torch.exp(lin)
            nll = (mu - y_t * torch.log(mu + 1e-9)).mean()
            l2 = (
                (self.off_b ** 2).mean()
                + (self.def_b ** 2).mean()
                + (self.off_e ** 2).mean()
                + (self.def_e ** 2).mean()
                + (self.bh ** 2).mean()
            )
            loss = nll + reg * l2
            loss.backward()
            opt.step()

        self.k = k
        self.reg = reg

    def _predict_total_df(self, df: pd.DataFrame) -> np.ndarray:
        import torch

        d = self._prepare_df(df)
        off_ix, def_ix, is_home, log_toi, _ = self._encode(d, fit=False)
        off_ix_t = torch.tensor(off_ix, dtype=torch.long)
        def_ix_t = torch.tensor(def_ix, dtype=torch.long)
        is_home_t = torch.tensor(is_home, dtype=torch.float32)
        log_toi_t = torch.tensor(log_toi, dtype=torch.float32)
        with torch.no_grad():
            lin = (
                log_toi_t
                + self.b0.detach().cpu()
                + self.bh.detach().cpu() * is_home_t
                + self.off_b.detach().cpu()[off_ix_t]
                + self.def_b.detach().cpu()[def_ix_t]
                + (self.off_e.detach().cpu()[off_ix_t] * self.def_e.detach().cpu()[def_ix_t]).sum(dim=1)
            )
            mu = torch.exp(lin).numpy()
        return np.clip(mu, 1e-9, None)

    def _score_combo(self, df: pd.DataFrame, k: int, reg: float) -> float:
        groups = df["game_id"].astype(str).to_numpy()
        uniq = np.unique(groups)
        n_splits = min(3, len(uniq))
        if n_splits < 2:
            return float("inf")
        splitter = GroupKFold(n_splits=n_splits)
        idx = np.arange(len(df))
        scores = []
        for tr, va in splitter.split(idx, groups=groups):
            dtr = df.iloc[tr]
            dva = df.iloc[va]
            tmp = LowRankPoissonFactorModel(
                random_state=self.random_state,
                k_grid=[k],
                reg_grid=[reg],
                epochs=max(20, self.epochs // 2),
                lr=self.lr,
            )
            tmp._train_torch(dtr, k, reg)
            pred = tmp._predict_total_df(dva)
            s = poisson_deviance_safe(dva["xg_for"].to_numpy(dtype=float), pred)
            scores.append(s)
        return float(np.mean(scores)) if scores else float("inf")

    def fit(self, df: pd.DataFrame):
        d = self._prepare_df(df)
        best = (self.k_grid[0], self.reg_grid[0])
        best_score = float("inf")
        for k in self.k_grid:
            for reg in self.reg_grid:
                try:
                    s = self._score_combo(d, k, reg)
                except Exception:
                    s = float("inf")
                if s < best_score:
                    best_score = s
                    best = (k, reg)
        self._train_torch(d, *best)
        self.nested_cv_score = best_score
        return self

    def predict_total(self, df: pd.DataFrame) -> np.ndarray:
        return self._predict_total_df(df)

    def predict_rate_hr(self, df: pd.DataFrame) -> np.ndarray:
        d = self._prepare_df(df)
        mu = self._predict_total_df(d)
        rate = mu / np.maximum(d["toi_hr"].to_numpy(dtype=float), 1e-9)
        return np.clip(rate, EPS_RATE, None)
