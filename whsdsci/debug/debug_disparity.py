from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


@dataclass
class Phase1bValidityResult:
    valid: bool
    reason: str
    n_teams: int
    strength_std_xg60: float
    log_ratio_std: float
    ratio_unique_1e6: int
    ratio_one_share_1e6: float


def check_phase1b_validity_tables(
    strength_df: pd.DataFrame,
    ratio_df: pd.DataFrame,
) -> Phase1bValidityResult:
    if strength_df.empty:
        return Phase1bValidityResult(
            valid=False,
            reason="empty_strength",
            n_teams=0,
            strength_std_xg60=0.0,
            log_ratio_std=0.0,
            ratio_unique_1e6=0,
            ratio_one_share_1e6=1.0,
        )
    if ratio_df.empty:
        return Phase1bValidityResult(
            valid=False,
            reason="empty_ratio",
            n_teams=0,
            strength_std_xg60=float(np.std(pd.to_numeric(strength_df["strength_xg60"], errors="coerce").fillna(0.0))),
            log_ratio_std=0.0,
            ratio_unique_1e6=0,
            ratio_one_share_1e6=1.0,
        )

    strengths = pd.to_numeric(strength_df["strength_xg60"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    ratios = pd.to_numeric(ratio_df["ratio"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    ratios = np.clip(ratios, 1e-12, None)

    strength_std = float(np.std(strengths))
    log_ratio_std = float(np.std(np.log(ratios)))
    rounded = np.round(ratios, 6)
    unique_cnt = int(np.unique(rounded).size)
    one_share = float(np.mean(rounded == 1.0))
    n_teams = int(ratio_df["team"].astype(str).nunique())

    # Hard invalidity constraints from the protocol.
    if strength_std < 1e-6:
        return Phase1bValidityResult(
            valid=False,
            reason="degenerate_strength_std",
            n_teams=n_teams,
            strength_std_xg60=strength_std,
            log_ratio_std=log_ratio_std,
            ratio_unique_1e6=unique_cnt,
            ratio_one_share_1e6=one_share,
        )
    if log_ratio_std < 1e-6:
        return Phase1bValidityResult(
            valid=False,
            reason="flat_log_ratio_std",
            n_teams=n_teams,
            strength_std_xg60=strength_std,
            log_ratio_std=log_ratio_std,
            ratio_unique_1e6=unique_cnt,
            ratio_one_share_1e6=one_share,
        )
    if unique_cnt < 5:
        return Phase1bValidityResult(
            valid=False,
            reason="ratio_unique_lt_5",
            n_teams=n_teams,
            strength_std_xg60=strength_std,
            log_ratio_std=log_ratio_std,
            ratio_unique_1e6=unique_cnt,
            ratio_one_share_1e6=one_share,
        )
    if one_share > 0.9:
        return Phase1bValidityResult(
            valid=False,
            reason="ratio_one_share_gt_90pct",
            n_teams=n_teams,
            strength_std_xg60=strength_std,
            log_ratio_std=log_ratio_std,
            ratio_unique_1e6=unique_cnt,
            ratio_one_share_1e6=one_share,
        )

    return Phase1bValidityResult(
        valid=True,
        reason="ok",
        n_teams=n_teams,
        strength_std_xg60=strength_std,
        log_ratio_std=log_ratio_std,
        ratio_unique_1e6=unique_cnt,
        ratio_one_share_1e6=one_share,
    )


def debug_best_disparity(
    model,
    ev_df: pd.DataFrame,
    out_path: Path,
    assert_valid: bool = True,
) -> Phase1bValidityResult:
    strengths = compute_standardized_strengths(model=model, train_ev_df=ev_df)
    ratios = compute_disparity_ratios(strengths)
    validity = check_phase1b_validity_tables(strength_df=strengths, ratio_df=ratios)

    payload = asdict(validity)
    payload["n_off_units"] = int(strengths["off_unit"].astype(str).nunique()) if not strengths.empty else 0
    payload["n_def_units"] = int(ev_df["def_unit"].astype(str).nunique()) if "def_unit" in ev_df.columns else 0
    payload["top10"] = ratios.head(10).to_dict(orient="records") if not ratios.empty else []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if assert_valid and not validity.valid:
        raise AssertionError(f"Phase1b validity failed: {validity.reason}")
    return validity

