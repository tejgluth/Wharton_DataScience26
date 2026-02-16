from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from whsdsci.build_long import build_canonical_long
from whsdsci.debug.debug_disparity import check_phase1b_validity_tables, debug_best_disparity
from whsdsci.ensemble.search import (
    Config,
    FittedSearchModel,
    _build_cache_for_models,
    _evaluate_config,
)
from whsdsci.eval.bootstrap import bootstrap_rank_stability
from whsdsci.io import discover_paths
from whsdsci.models import get_model_builders
from whsdsci.pdf_read import write_pdf_notes
from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths
from whsdsci.tuning.maximize_tree_poisson import maximize_tree_poisson


RANDOM_STATE = 0
LOGGER = logging.getLogger("whsdsci.best_system_maximizer")


def setup_logging(outputs_dir: Path) -> logging.Logger:
    logger = LOGGER
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(outputs_dir / "best_system_maximizer.log", mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def run_pytest(repo_root: Path, logger: logging.Logger) -> None:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    logger.info("Running pytest before best-system maximizer: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    logger.info("pytest stdout:\n%s", proc.stdout)
    if proc.stderr:
        logger.info("pytest stderr:\n%s", proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError("pytest failed; stopping")


def write_web_refs_best_system(out_path: Path) -> None:
    notes = []
    notes.append("# Web References for Best-System Maximizer")
    notes.append("")
    notes.append("Checked: 2026-02-15")
    notes.append("")
    notes.append("1) XGBoost Poisson objective (`count:poisson`) and stabilization knobs")
    notes.append("- https://xgboost.readthedocs.io/en/stable/parameter.html")
    notes.append("- Notes used: Poisson objective, `max_delta_step`, regularization parameters, and histogram tree method.")
    notes.append("")
    notes.append("2) Poisson deviance domain constraints")
    notes.append("- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html")
    notes.append("- Notes used: target must be non-negative and predictions strictly positive.")
    notes.append("")
    notes.append("3) Grouped CV leakage control")
    notes.append("- https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html")
    notes.append("- https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data")
    notes.append("- Notes used: groups ensure train/test partitioning respects game-level boundaries and avoids leakage.")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(notes) + "\n", encoding="utf-8")


def _parse_config_ids_from_text(txt: str) -> list[str]:
    return sorted(set(re.findall(r"(cfg_[A-Za-z0-9_\\.\\-]+)", txt)))


def _row_to_config(row: pd.Series) -> Config:
    return Config(
        config_id=str(row["config_id"]),
        combiner_family=str(row["combiner_family"]),
        base_pool_id=str(row.get("base_pool_id", "unknown")),
        base_models=list(json.loads(row["base_models"])),
        calibration_type=str(row.get("calibration_type", "none")),
        hyperparams=dict(json.loads(row.get("hyperparams", "{}"))),
    )


def _load_candidate_configs(outputs_dir: Path, logger: logging.Logger) -> list[Config]:
    res_path = outputs_dir / "ensemble_search_results.csv"
    cfgs: dict[str, Config] = {}
    if res_path.exists():
        r = pd.read_csv(res_path)
        r = r[
            (r["status"] == "OK")
            & (r["stage"].isin(["full", "deep"]))
            & (r["mean_cv_poisson_deviance"].notna())
        ].copy()
        r = r.sort_values("mean_cv_poisson_deviance")
        top = r.head(50)
        for _, row in top.iterrows():
            c = _row_to_config(row)
            cfgs[c.config_id] = c

        explicit_ids = set()
        if (outputs_dir / "ensemble_best_config.json").exists():
            payload = json.loads((outputs_dir / "ensemble_best_config.json").read_text(encoding="utf-8"))
            cid = payload.get("config_id")
            if cid:
                explicit_ids.add(str(cid))
        if (outputs_dir / "best_method.txt").exists():
            explicit_ids.update(_parse_config_ids_from_text((outputs_dir / "best_method.txt").read_text(encoding="utf-8")))
        explicit_ids.update({"cfg_00039", "cfg_00699"})
        for cid in explicit_ids:
            m = r[r["config_id"].astype(str) == cid]
            if not m.empty:
                c = _row_to_config(m.iloc[0])
                cfgs[c.config_id] = c
        logger.info("Loaded %s candidate configs from ensemble search results", len(cfgs))

    if not cfgs:
        logger.warning("No ensemble_search_results.csv candidates; using tree-poisson fallback candidates")
        fallback = [
            Config(
                config_id="fallback_cfg_00699_style",
                combiner_family="tree_poisson",
                base_pool_id="fallback_top3",
                base_models=["POISSON_GLM_OFFSET", "POISSON_GLM_OFFSET_REG", "TWO_STAGE_SHOTS_XG"],
                calibration_type="scalar",
                hyperparams={"backend": "auto", "max_depth": 3, "learning_rate": 0.08, "n_estimators": 200},
            ),
            Config(
                config_id="fallback_cfg_00039_style",
                combiner_family="tree_poisson",
                base_pool_id="fallback_top3",
                base_models=["POISSON_GLM_OFFSET", "POISSON_GLM_OFFSET_REG", "TWO_STAGE_SHOTS_XG"],
                calibration_type="none",
                hyperparams={"backend": "auto", "max_depth": 3, "learning_rate": 0.05, "n_estimators": 400},
            ),
        ]
        for c in fallback:
            cfgs[c.config_id] = c
    return list(cfgs.values())


def _evaluate_candidates_strict(
    ev_df: pd.DataFrame,
    outputs_dir: Path,
    candidates: list[Config],
    random_state: int,
    logger: logging.Logger,
    validity_top_n: int = 20,
) -> pd.DataFrame:
    builders = get_model_builders(random_state=random_state)
    all_models = sorted({m for c in candidates for m in c.base_models if m in builders})
    cache_dir = outputs_dir / "ensemble_oof_cache" / f"confirm_seed_{random_state}"
    fold_cache, valid_models = _build_cache_for_models(
        ev_df=ev_df,
        base_builders=builders,
        model_names=all_models,
        outer_splits=5,
        inner_splits=3,
        cache_dir=cache_dir,
    )
    valid_models = set(valid_models)

    rows = []
    for i, c in enumerate(candidates, start=1):
        t0 = time.perf_counter()
        c2 = Config(
            config_id=c.config_id,
            combiner_family=c.combiner_family,
            base_pool_id=c.base_pool_id,
            base_models=[m for m in c.base_models if m in valid_models],
            calibration_type=c.calibration_type,
            hyperparams=c.hyperparams,
        )
        row = {
            "config_id": c2.config_id,
            "combiner_family": c2.combiner_family,
            "base_pool_id": c2.base_pool_id,
            "base_models": json.dumps(c2.base_models),
            "calibration_type": c2.calibration_type,
            "hyperparams": json.dumps(c2.hyperparams, sort_keys=True),
            "seed_eval": random_state,
        }
        if len(c2.base_models) < 2:
            row.update(
                {
                    "status": "INVALID",
                    "invalid_reason": "insufficient_valid_base_models",
                    "mean_cv_poisson_deviance": np.nan,
                    "std_cv_poisson_deviance": np.nan,
                    "mean_weighted_mse_rate": np.nan,
                    "calibration_ratio": np.nan,
                    "phase1b_valid": False,
                    "strength_std_xg60": np.nan,
                    "log_ratio_std": np.nan,
                    "ratio_unique_1e6": np.nan,
                    "ratio_one_share_1e6": np.nan,
                    "runtime_seconds": float(time.perf_counter() - t0),
                }
            )
            rows.append(row)
            continue

        try:
            summary, _ = _evaluate_config(c2, fold_cache)
            row.update(
                {
                    "status": "OK",
                    "invalid_reason": "",
                    "mean_cv_poisson_deviance": summary["mean_cv_poisson_deviance"],
                    "std_cv_poisson_deviance": summary["std_cv_poisson_deviance"],
                    "mean_weighted_mse_rate": summary["mean_weighted_mse_rate"],
                    "calibration_ratio": summary["calibration_ratio"],
                    "phase1b_valid": np.nan,
                    "strength_std_xg60": np.nan,
                    "log_ratio_std": np.nan,
                    "ratio_unique_1e6": np.nan,
                    "ratio_one_share_1e6": np.nan,
                    "runtime_seconds": float(time.perf_counter() - t0),
                }
            )
        except Exception as exc:
            row.update(
                {
                    "status": "FAILED",
                    "invalid_reason": str(exc)[:300],
                    "mean_cv_poisson_deviance": np.nan,
                    "std_cv_poisson_deviance": np.nan,
                    "mean_weighted_mse_rate": np.nan,
                    "calibration_ratio": np.nan,
                    "phase1b_valid": False,
                    "strength_std_xg60": np.nan,
                    "log_ratio_std": np.nan,
                    "ratio_unique_1e6": np.nan,
                    "ratio_one_share_1e6": np.nan,
                    "runtime_seconds": float(time.perf_counter() - t0),
                }
            )
        rows.append(row)
        if i % 10 == 0 or i == len(candidates):
            logger.info("Strict CV progress: %s/%s candidates", i, len(candidates))

    out = pd.DataFrame(rows)
    if "invalid_reason" not in out.columns:
        out["invalid_reason"] = ""
    out["invalid_reason"] = out["invalid_reason"].fillna("").astype(str)
    # Keep nullable boolean dtype to avoid dtype assignment issues across pandas versions.
    if "phase1b_valid" not in out.columns:
        out["phase1b_valid"] = pd.Series([pd.NA] * len(out), dtype="boolean")
    else:
        out["phase1b_valid"] = pd.Series(out["phase1b_valid"], dtype="boolean")
    ok = out[out["status"] == "OK"].copy().sort_values("mean_cv_poisson_deviance")
    to_validate_ids = ok.head(max(1, int(validity_top_n)))["config_id"].astype(str).tolist()
    logger.info("Running full-data Phase1b validity on top %s configs", len(to_validate_ids))

    cfg_map = {c.config_id: c for c in candidates}
    for i, cid in enumerate(to_validate_ids, start=1):
        c = cfg_map[cid]
        try:
            model = FittedSearchModel(random_state=random_state, base_builders=builders, config=c).fit(ev_df)
            strengths = compute_standardized_strengths(model=model, train_ev_df=ev_df)
            ratios = compute_disparity_ratios(strengths)
            validity = check_phase1b_validity_tables(strength_df=strengths, ratio_df=ratios)
            out.loc[out["config_id"] == cid, "phase1b_valid"] = bool(validity.valid)
            out.loc[out["config_id"] == cid, "invalid_reason"] = "" if validity.valid else validity.reason
            out.loc[out["config_id"] == cid, "strength_std_xg60"] = validity.strength_std_xg60
            out.loc[out["config_id"] == cid, "log_ratio_std"] = validity.log_ratio_std
            out.loc[out["config_id"] == cid, "ratio_unique_1e6"] = validity.ratio_unique_1e6
            out.loc[out["config_id"] == cid, "ratio_one_share_1e6"] = validity.ratio_one_share_1e6
            if not validity.valid:
                out.loc[out["config_id"] == cid, "status"] = "INVALID"
        except Exception as exc:
            out.loc[out["config_id"] == cid, "status"] = "FAILED"
            out.loc[out["config_id"] == cid, "invalid_reason"] = str(exc)[:300]
            out.loc[out["config_id"] == cid, "phase1b_valid"] = False
        if i % 5 == 0 or i == len(to_validate_ids):
            logger.info("Validity progress: %s/%s", i, len(to_validate_ids))

    out["phase1b_valid"] = out["phase1b_valid"].fillna(False).astype(bool)
    out.loc[(out["status"] == "OK") & (~out["config_id"].astype(str).isin(to_validate_ids)), "status"] = "INVALID"
    out.loc[(out["invalid_reason"] == "") & (out["status"] == "INVALID"), "invalid_reason"] = "validity_not_checked"
    return out


def _choose_best_primary(df: pd.DataFrame) -> pd.Series:
    x = df[(df["status"] == "OK") & (df["phase1b_valid"] == True)].copy()  # noqa: E712
    if x.empty:
        raise RuntimeError("No Phase1b-valid candidates passed strict evaluation")
    x = x.sort_values(["mean_cv_poisson_deviance", "std_cv_poisson_deviance"], ascending=[True, True]).reset_index(drop=True)
    if len(x) == 1:
        return x.iloc[0]

    a = x.iloc[0]
    b = x.iloc[1]
    da = float(a["mean_cv_poisson_deviance"])
    db = float(b["mean_cv_poisson_deviance"])

    # Tie-break rule 1: if deviance differs by > 0.2%, pick lower deviance.
    if da > 0 and ((db - da) / da) > 0.002:
        return a

    # Tie-break rule 2: lower fold std deviance.
    sa = float(a["std_cv_poisson_deviance"])
    sb = float(b["std_cv_poisson_deviance"])
    if sb < sa:
        return b
    return a


def _run_stability_tiebreak_if_needed(
    ev_df: pd.DataFrame,
    outputs_dir: Path,
    candidates_df: pd.DataFrame,
    random_state: int,
) -> pd.DataFrame:
    x = candidates_df[(candidates_df["status"] == "OK") & (candidates_df["phase1b_valid"] == True)].copy()  # noqa: E712
    x = x.sort_values(["mean_cv_poisson_deviance", "std_cv_poisson_deviance"]).head(2).reset_index(drop=True)
    if len(x) < 2:
        return x

    a = x.iloc[0]
    b = x.iloc[1]
    da = float(a["mean_cv_poisson_deviance"])
    db = float(b["mean_cv_poisson_deviance"])
    sa = float(a["std_cv_poisson_deviance"])
    sb = float(b["std_cv_poisson_deviance"])
    if not (da > 0 and ((db - da) / da) <= 0.002 and np.isclose(sa, sb, rtol=0.0, atol=1e-12)):
        return x

    builders = get_model_builders(random_state=random_state)
    rows = []
    for _, r in x.iterrows():
        cfg = Config(
            config_id=str(r["config_id"]),
            combiner_family=str(r["combiner_family"]),
            base_pool_id=str(r["base_pool_id"]),
            base_models=list(json.loads(r["base_models"])),
            calibration_type=str(r["calibration_type"]),
            hyperparams=dict(json.loads(r["hyperparams"])),
        )
        model = FittedSearchModel(random_state=random_state, base_builders=builders, config=cfg).fit(ev_df)
        top10 = compute_disparity_ratios(compute_standardized_strengths(model, ev_df)).head(10)
        teams = top10["team"].astype(str).tolist()
        boot = bootstrap_rank_stability(
            model_factory=lambda c=cfg: FittedSearchModel(
                random_state=random_state,
                base_builders=builders,
                config=c,
            ),
            ev_df=ev_df,
            full_top10_teams=teams,
            n_boot=200,
            random_state=random_state,
            show_progress=False,
        )
        rows.append(
            {
                "config_id": cfg.config_id,
                "stability_score": boot.stability_score,
            }
        )
    stab = pd.DataFrame(rows)
    out = x.merge(stab, on="config_id", how="left")
    out = out.sort_values(["mean_cv_poisson_deviance", "std_cv_poisson_deviance", "stability_score"], ascending=[True, True, True])
    return out


def confirm_best_system(
    ev_df: pd.DataFrame,
    outputs_dir: Path,
    random_state: int = 0,
    recheck_seeds: list[int] | None = None,
    recheck_top_k: int = 3,
    validity_top_n: int = 50,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    log = logger or LOGGER
    candidates = _load_candidate_configs(outputs_dir=outputs_dir, logger=log)
    strict = _evaluate_candidates_strict(
        ev_df=ev_df,
        outputs_dir=outputs_dir,
        candidates=candidates,
        random_state=random_state,
        logger=log,
        validity_top_n=validity_top_n,
    )
    strict_path = outputs_dir / "confirmed_best_candidates_seed0.csv"
    strict.to_csv(strict_path, index=False)

    best_primary = _choose_best_primary(strict)
    tie_df = _run_stability_tiebreak_if_needed(ev_df=ev_df, outputs_dir=outputs_dir, candidates_df=strict, random_state=random_state)
    best_after_tie = tie_df.iloc[0] if not tie_df.empty else best_primary

    # Repeatability check: top 3 valid candidates across seeds 0,1,2.
    valid = strict[(strict["status"] == "OK") & (strict["phase1b_valid"] == True)].copy()  # noqa: E712
    top3_ids = valid.sort_values("mean_cv_poisson_deviance").head(3)["config_id"].astype(str).tolist()
    seeds = recheck_seeds or [0, 1, 2]
    re_rows = []
    for seed in seeds:
        sub = [c for c in candidates if c.config_id in top3_ids[: max(1, int(recheck_top_k))]]
        ev_seed = _evaluate_candidates_strict(
            ev_df=ev_df,
            outputs_dir=outputs_dir,
            candidates=sub,
            random_state=seed,
            logger=log,
            validity_top_n=max(1, min(validity_top_n, len(sub))),
        )
        re_rows.append(ev_seed)
    recheck = pd.concat(re_rows, ignore_index=True) if re_rows else pd.DataFrame()
    if not recheck.empty:
        recheck.to_csv(outputs_dir / "confirmed_best_recheck.csv", index=False)
        agg = (
            recheck[(recheck["status"] == "OK") & (recheck["phase1b_valid"] == True)]  # noqa: E712
            .groupby("config_id", as_index=False)["mean_cv_poisson_deviance"]
            .mean()
            .rename(columns={"mean_cv_poisson_deviance": "seed_mean_cv_poisson_deviance"})
            .sort_values("seed_mean_cv_poisson_deviance")
        )
        if not agg.empty:
            winner_id = str(agg.iloc[0]["config_id"])
        else:
            winner_id = str(best_after_tie["config_id"])
    else:
        recheck = pd.DataFrame()
        winner_id = str(best_after_tie["config_id"])

    chosen = strict[strict["config_id"].astype(str) == winner_id].iloc[0]
    cfg = next(c for c in candidates if c.config_id == winner_id)

    best_payload = {
        "config_id": cfg.config_id,
        "combiner_family": cfg.combiner_family,
        "base_pool_id": cfg.base_pool_id,
        "base_models": cfg.base_models,
        "calibration_type": cfg.calibration_type,
        "hyperparams": cfg.hyperparams,
        "mean_cv_poisson_deviance": float(chosen["mean_cv_poisson_deviance"]),
        "std_cv_poisson_deviance": float(chosen["std_cv_poisson_deviance"]),
        "phase1b_validity": {
            "valid": bool(chosen["phase1b_valid"]),
            "reason": str(chosen["invalid_reason"]) if not bool(chosen["phase1b_valid"]) else "ok",
            "strength_std_xg60": float(chosen["strength_std_xg60"]),
            "log_ratio_std": float(chosen["log_ratio_std"]),
            "ratio_unique_1e6": int(chosen["ratio_unique_1e6"]),
            "ratio_one_share_1e6": float(chosen["ratio_one_share_1e6"]),
        },
        "repeatability_seeds": seeds,
    }

    (outputs_dir / "confirmed_best_config.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    (outputs_dir / "ensemble_best_config.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
    (outputs_dir / "best_method.txt").write_text(
        f"best_method=CONFIRMED::{cfg.combiner_family}\n"
        f"config_id={cfg.config_id}\n"
        f"mean_cv_poisson_deviance={best_payload['mean_cv_poisson_deviance']}\n",
        encoding="utf-8",
    )

    rep = []
    rep.append("# Confirmed Best Config Report")
    rep.append("")
    rep.append("Protocol:")
    rep.append("- Primary metric: mean 5-fold GroupKFold EV Poisson deviance on xG totals.")
    rep.append("- Hard validity filter: non-flat disparity + non-degenerate standardized strengths.")
    rep.append("- Tie rules: 0.2% deviance threshold, then std deviance, then bootstrap stability if needed.")
    rep.append("- Recheck: seeds {0,1,2} over top strict candidates.")
    rep.append("")
    rep.append(f"Chosen config: `{cfg.config_id}` (`{cfg.combiner_family}`)")
    rep.append(f"- mean_cv_poisson_deviance: {best_payload['mean_cv_poisson_deviance']}")
    rep.append(f"- std_cv_poisson_deviance: {best_payload['std_cv_poisson_deviance']}")
    rep.append(f"- phase1b_valid: {best_payload['phase1b_validity']['valid']}")
    rep.append("")
    rep.append("## Strict Seed-0 Ranking (Top 20 Valid)")
    rep.append("```")
    rep.append(
        strict[(strict["status"] == "OK") & (strict["phase1b_valid"] == True)]  # noqa: E712
        .sort_values("mean_cv_poisson_deviance")
        .head(20)[
            [
                "config_id",
                "combiner_family",
                "mean_cv_poisson_deviance",
                "std_cv_poisson_deviance",
                "strength_std_xg60",
                "log_ratio_std",
                "ratio_unique_1e6",
            ]
        ]
        .to_string(index=False)
    )
    rep.append("```")
    if not recheck.empty:
        rep.append("")
        rep.append("## Recheck Across Seeds")
        rep.append("```")
        rep.append(
            recheck[(recheck["status"] == "OK") & (recheck["phase1b_valid"] == True)]  # noqa: E712
            .sort_values(["config_id", "seed_eval"])[
                ["config_id", "seed_eval", "mean_cv_poisson_deviance", "std_cv_poisson_deviance"]
            ]
            .to_string(index=False)
        )
        rep.append("```")
    (outputs_dir / "confirmed_best_report.md").write_text("\n".join(rep) + "\n", encoding="utf-8")

    return {"best_payload": best_payload, "best_config": cfg, "strict_df": strict, "recheck_df": recheck}


def _select_tree_baseline_for_maximize(
    outputs_dir: Path,
    confirmed_payload: dict[str, Any],
    logger: logging.Logger,
) -> dict[str, Any]:
    if str(confirmed_payload.get("combiner_family", "")) == "tree_poisson":
        return {
            "base_models": list(confirmed_payload.get("base_models", [])),
            "hyperparams": dict(confirmed_payload.get("hyperparams", {})),
        }
    # fallback to best tree_poisson in search results
    res_path = outputs_dir / "ensemble_search_results.csv"
    if res_path.exists():
        r = pd.read_csv(res_path)
        r = r[
            (r["status"] == "OK")
            & (r["combiner_family"].astype(str) == "tree_poisson")
            & (r["stage"].isin(["full", "deep"]))
            & (r["mean_cv_poisson_deviance"].notna())
        ].sort_values("mean_cv_poisson_deviance")
        if not r.empty:
            rr = r.iloc[0]
            logger.info("Using tree-poisson baseline for maximize from %s", rr["config_id"])
            return {
                "base_models": list(json.loads(rr["base_models"])),
                "hyperparams": dict(json.loads(rr["hyperparams"])),
            }
    return {
        "base_models": ["POISSON_GLM_OFFSET", "POISSON_GLM_OFFSET_REG", "TWO_STAGE_SHOTS_XG", "TWEEDIE_GLM_RATE", "HURDLE_XG"],
        "hyperparams": {"backend": "auto", "max_depth": 3, "learning_rate": 0.08, "n_estimators": 200},
    }


def main() -> None:
    repo_root = Path.cwd()
    outputs_dir = repo_root / "outputs"
    logger = setup_logging(outputs_dir=outputs_dir)
    logger.info("Starting run_best_system_maximizer")

    np.random.seed(RANDOM_STATE)
    os.environ["PYTHONHASHSEED"] = "0"

    write_web_refs_best_system(outputs_dir / "web_refs_best_system.txt")
    paths = discover_paths(repo_root=repo_root, outputs_dir=outputs_dir)
    write_pdf_notes(paths=paths, out_path=outputs_dir / "pdf_notes.txt")

    raw = pd.read_csv(paths["whl_2025"])
    _, ev_df, _ = build_canonical_long(raw_df=raw, outputs_dir=outputs_dir)
    if ev_df.empty:
        raise RuntimeError("EV dataframe is empty")

    run_pytest(repo_root=repo_root, logger=logger)

    confirm = confirm_best_system(
        ev_df=ev_df,
        outputs_dir=outputs_dir,
        random_state=RANDOM_STATE,
        recheck_seeds=[0, 1, 2],
        recheck_top_k=1,
        validity_top_n=50,
        logger=logger,
    )
    best_cfg: Config = confirm["best_config"]
    best_payload: dict[str, Any] = confirm["best_payload"]
    logger.info("CONFIRM_BEST winner: %s", best_payload)

    builders = get_model_builders(random_state=RANDOM_STATE)
    best_model = FittedSearchModel(random_state=RANDOM_STATE, base_builders=builders, config=best_cfg).fit(ev_df)
    debug_best_disparity(
        model=best_model,
        ev_df=ev_df,
        out_path=outputs_dir / "debug_best_flatness.json",
        assert_valid=True,
    )

    baseline = _select_tree_baseline_for_maximize(outputs_dir=outputs_dir, confirmed_payload=best_payload, logger=logger)
    max_time_hours = float(os.getenv("WHSDSCI_MAX_TIME_HOURS", "6"))
    result = maximize_tree_poisson(
        ev_df=ev_df,
        outputs_dir=outputs_dir,
        base_builders=builders,
        baseline_config=baseline,
        random_state=RANDOM_STATE,
        stage1_trials=500,
        stage2_trials=1500,
        max_time_hours=max_time_hours,
        logger=logger,
    )
    tuned_best = result["best_payload"]
    confirmed_dev = float(best_payload["mean_cv_poisson_deviance"])
    tuned_dev = float(tuned_best["mean_cv_poisson_deviance"])

    if confirmed_dev <= tuned_dev:
        strengths = compute_standardized_strengths(model=best_model, train_ev_df=ev_df)
        top10 = compute_disparity_ratios(strengths).head(10).copy()
        top10["method"] = f"CONFIRMED::{best_cfg.combiner_family}::{best_cfg.config_id}"
        top10 = top10[["rank", "team", "line1_strength_xg60", "line2_strength_xg60", "ratio", "method"]]
        top10.to_csv(outputs_dir / "final_top10.csv", index=False)
        top10.to_csv(outputs_dir / "submission_phase1b.csv", index=False)
        (outputs_dir / "best_method.txt").write_text(
            f"best_method=CONFIRMED::{best_cfg.combiner_family}\n"
            f"config_id={best_cfg.config_id}\n"
            f"mean_cv_poisson_deviance={confirmed_dev}\n",
            encoding="utf-8",
        )
        (outputs_dir / "ensemble_best_config.json").write_text(json.dumps(best_payload, indent=2), encoding="utf-8")
        (outputs_dir / "best_overall_selection.md").write_text(
            "# Best Overall Selection\n\n"
            "Confirmed winner remained better than maximize winner.\n\n"
            f"- confirmed: {best_cfg.config_id} deviance={confirmed_dev}\n"
            f"- maximize: {tuned_best['trial_id']} deviance={tuned_dev}\n",
            encoding="utf-8",
        )
        logger.info("Retained confirmed winner as best overall (%.9f <= %.9f)", confirmed_dev, tuned_dev)
    else:
        logger.info("Maximize winner improved over confirmed baseline (%.9f < %.9f)", tuned_dev, confirmed_dev)

    report = []
    report.append("# Best System Maximizer Final Report")
    report.append("")
    report.append("## Confirmed Baseline")
    report.append(f"- config_id: {best_payload['config_id']}")
    report.append(f"- family: {best_payload['combiner_family']}")
    report.append(f"- mean_cv_poisson_deviance: {best_payload['mean_cv_poisson_deviance']}")
    report.append("")
    report.append("## Tuned Winner")
    report.append(f"- trial_id: {tuned_best['trial_id']}")
    report.append(f"- family: {tuned_best['family']}")
    report.append(f"- mean_cv_poisson_deviance: {tuned_best['mean_cv_poisson_deviance']}")
    report.append("")
    delta = float(best_payload["mean_cv_poisson_deviance"]) - float(tuned_best["mean_cv_poisson_deviance"])
    report.append(f"## Improvement Delta")
    report.append(f"- deviance_delta = confirmed_baseline - tuned_best = {delta}")
    report.append(f"- best_overall = {'confirmed' if confirmed_dev <= tuned_dev else 'maximize'}")
    report.append("")
    report.append("## Final Top10")
    report.append("```")
    report.append(pd.read_csv(outputs_dir / "final_top10.csv").to_string(index=False))
    report.append("```")
    (outputs_dir / "maximize_tree_poisson_best_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("Confirmed baseline best:")
    print(json.dumps(best_payload, indent=2))
    print()
    print("Tuned best:")
    print(json.dumps(tuned_best, indent=2))


if __name__ == "__main__":
    main()
