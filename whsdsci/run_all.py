from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from whsdsci.build_long import build_canonical_long
from whsdsci.ensemble.oof import build_oof_predictions, select_diverse_models
from whsdsci.eval.bootstrap import bootstrap_rank_stability
from whsdsci.eval.cv import make_group_kfold_splits, make_time_split
from whsdsci.eval.metrics import calibration_ratio, mae_total, poisson_deviance_safe, weighted_mse_rate
from whsdsci.io import discover_paths
from whsdsci.models import get_ensemble_model_builders, get_model_builders
from whsdsci.models.base import EPS_RATE, SkipModelError
from whsdsci.models.elasticnet_rapm import ElasticNetRapmSoftplusModel
from whsdsci.models.poisson_glm_offset_reg import PoissonGlmOffsetRegModel
from whsdsci.models.ridge_rapm import RidgeRapmSoftplusModel
from whsdsci.models.tweedie_glm import TweedieGlmRateModel
from whsdsci.pdf_read import write_pdf_notes
from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


RANDOM_STATE = 0


def setup_logging(outputs_dir: Path) -> logging.Logger:
    logger = logging.getLogger("whsdsci")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    outputs_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(outputs_dir / "run.log", mode="w", encoding="utf-8")
    sh = logging.StreamHandler(sys.stdout)

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger


def run_pytest(repo_root: Path, logger: logging.Logger) -> None:
    cmd = [sys.executable, "-m", "pytest", "-q"]
    logger.info("Running pytest before benchmarking: %s", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=repo_root, capture_output=True, text=True)
    logger.info("pytest stdout:\n%s", proc.stdout)
    if proc.stderr:
        logger.info("pytest stderr:\n%s", proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError("pytest failed; stopping run")


def _evaluate_one_method(
    method_name: str,
    factory,
    df: pd.DataFrame,
    logger: logging.Logger,
    n_splits: int = 5,
    include_time_split: bool = True,
) -> tuple[list[dict], str, str]:
    rows: list[dict] = []

    splits = make_group_kfold_splits(df, n_splits=n_splits)
    ts = make_time_split(df) if include_time_split else None
    if include_time_split and ts is not None:
        splits.append(ts)

    for split in splits:
        tr = df.iloc[split.train_idx].reset_index(drop=True)
        te = df.iloc[split.test_idx].reset_index(drop=True)

        try:
            model = factory()
            model.fit(tr)
            pred_total = np.clip(model.predict_total(te), 1e-9, None)
            pred_rate = np.clip(model.predict_rate_hr(te), EPS_RATE, None)

            y_true = np.clip(te["xg_for"].to_numpy(dtype=float), 0, None)
            toi = np.maximum(te["toi_hr"].to_numpy(dtype=float), 1e-9)
            y_rate = y_true / toi

            row = {
                "dataset": "EV",
                "method": method_name,
                "split_type": split.split_type,
                "fold_id": split.fold_id,
                "n_train": len(tr),
                "n_test": len(te),
                "poisson_deviance": poisson_deviance_safe(y_true, pred_total),
                "weighted_mse_rate": weighted_mse_rate(y_rate, pred_rate, toi),
                "mae_total": mae_total(y_true, pred_total),
                "calibration_ratio": calibration_ratio(y_true, pred_total),
                "status": "OK",
                "message": "",
            }
            rows.append(row)

        except SkipModelError as exc:
            msg = str(exc)
            logger.warning("Method %s skipped: %s", method_name, msg)
            rows.append(
                {
                    "dataset": "EV",
                    "method": method_name,
                    "split_type": split.split_type,
                    "fold_id": split.fold_id,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "poisson_deviance": np.nan,
                    "weighted_mse_rate": np.nan,
                    "mae_total": np.nan,
                    "calibration_ratio": np.nan,
                    "status": "SKIPPED",
                    "message": msg,
                }
            )
            return rows, "SKIPPED", msg

        except Exception as exc:
            msg = f"{exc}\n{traceback.format_exc()}"
            logger.exception("Method %s failed on %s/%s", method_name, split.split_type, split.fold_id)
            rows.append(
                {
                    "dataset": "EV",
                    "method": method_name,
                    "split_type": split.split_type,
                    "fold_id": split.fold_id,
                    "n_train": len(tr),
                    "n_test": len(te),
                    "poisson_deviance": np.nan,
                    "weighted_mse_rate": np.nan,
                    "mae_total": np.nan,
                    "calibration_ratio": np.nan,
                    "status": "FAILED",
                    "message": msg,
                }
            )
            return rows, "FAILED", msg

    return rows, "OK", ""


def evaluate_methods(
    df: pd.DataFrame,
    builders: dict,
    logger: logging.Logger,
    dataset_label: str = "EV",
    n_splits: int = 5,
    include_time_split: bool = True,
) -> tuple[pd.DataFrame, dict[str, tuple[str, str]]]:
    all_rows: list[dict] = []
    status_map: dict[str, tuple[str, str]] = {}

    for method_name, factory in builders.items():
        logger.info("Evaluating method %s on %s", method_name, dataset_label)
        if dataset_label != "EV":
            # Reuse same evaluation but change dataset tag.
            rows, status, msg = _evaluate_one_method(
                method_name, factory, df, logger, n_splits=n_splits, include_time_split=include_time_split
            )
            for r in rows:
                r["dataset"] = dataset_label
        else:
            rows, status, msg = _evaluate_one_method(
                method_name, factory, df, logger, n_splits=n_splits, include_time_split=include_time_split
            )

        all_rows.extend(rows)
        status_map[method_name] = (status, msg)

    return pd.DataFrame(all_rows), status_map


def summarize_metrics(metrics_df: pd.DataFrame, status_map: dict[str, tuple[str, str]]) -> pd.DataFrame:
    rows: list[dict] = []
    methods = sorted(status_map.keys())
    for method in methods:
        status, msg = status_map[method]
        sub = metrics_df[(metrics_df["method"] == method) & (metrics_df["split_type"] == "groupkfold")]
        time_sub = metrics_df[(metrics_df["method"] == method) & (metrics_df["split_type"] == "time_split")]

        row = {
            "method": method,
            "status": status,
            "message": msg,
            "cv_folds": int(sub["poisson_deviance"].notna().sum()),
            "cv_poisson_deviance_mean": float(sub["poisson_deviance"].mean()) if not sub.empty else np.nan,
            "cv_poisson_deviance_std": float(sub["poisson_deviance"].std(ddof=0)) if not sub.empty else np.nan,
            "cv_weighted_mse_rate_mean": float(sub["weighted_mse_rate"].mean()) if not sub.empty else np.nan,
            "cv_mae_total_mean": float(sub["mae_total"].mean()) if not sub.empty else np.nan,
            "cv_calibration_ratio_mean": float(sub["calibration_ratio"].mean()) if not sub.empty else np.nan,
            "time_poisson_deviance": float(time_sub["poisson_deviance"].mean()) if not time_sub.empty else np.nan,
            "time_weighted_mse_rate": float(time_sub["weighted_mse_rate"].mean()) if not time_sub.empty else np.nan,
            "time_mae_total": float(time_sub["mae_total"].mean()) if not time_sub.empty else np.nan,
            "time_calibration_ratio": float(time_sub["calibration_ratio"].mean()) if not time_sub.empty else np.nan,
        }
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("cv_poisson_deviance_mean", na_position="last")
    return out


def fit_full_and_rank(
    ev_df: pd.DataFrame,
    builders: dict,
    logger: logging.Logger,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, tuple[str, str]], dict[str, object]]:
    top10_map: dict[str, pd.DataFrame] = {}
    status_map: dict[str, tuple[str, str]] = {}
    fitted_models: dict[str, object] = {}

    for method_name, factory in builders.items():
        try:
            model = factory()
            model.fit(ev_df)
            strengths = compute_standardized_strengths(model=model, train_ev_df=ev_df)
            ratios = compute_disparity_ratios(strengths)
            ratios["method"] = method_name
            top10_map[method_name] = ratios.head(10).copy()
            status_map[method_name] = ("OK", "")
            fitted_models[method_name] = model
        except SkipModelError as exc:
            status_map[method_name] = ("SKIPPED", str(exc))
            logger.warning("Full fit skipped %s: %s", method_name, exc)
        except Exception as exc:
            status_map[method_name] = ("FAILED", str(exc))
            logger.exception("Full fit failed %s", method_name)

    all_top10 = []
    for method_name, df in top10_map.items():
        all_top10.append(df)
    all_top10_df = pd.concat(all_top10, ignore_index=True) if all_top10 else pd.DataFrame()

    return top10_map, all_top10_df, status_map, fitted_models


def prepare_ensemble_base_pool(
    primary_df: pd.DataFrame,
    base_builders: dict,
    outputs_dir: Path,
    logger: logging.Logger,
    max_models: int = 6,
) -> list[str]:
    default_order = [
        "POISSON_GLM_OFFSET_REG",
        "POISSON_GLM_OFFSET",
        "TWEEDIE_GLM_RATE",
        "TWO_STAGE_SHOTS_XG",
        "HURDLE_XG",
        "RIDGE_RAPM_RATE_SOFTPLUS",
        "DEFENSE_ADJ_TWO_STEP",
    ]
    available = [m for m in default_order if m in base_builders]
    prior_metrics_path = outputs_dir / "metrics_summary.csv"

    ranked: list[str] = []
    if prior_metrics_path.exists():
        try:
            prev = pd.read_csv(prior_metrics_path)
            prev = prev[(prev["status"] == "OK") & prev["method"].isin(available)]
            prev = prev.sort_values("cv_poisson_deviance_mean")
            ranked = prev["method"].astype(str).tolist()
        except Exception as exc:
            logger.warning("Failed to read prior metrics summary for ensemble pool: %s", exc)

    for m in available:
        if m not in ranked:
            ranked.append(m)
    ranked = ranked[: max(8, max_models)]

    required = [m for m in ["POISSON_GLM_OFFSET_REG"] if m in ranked]
    if not ranked:
        return []

    logger.info("Building EV OOF predictions for ensemble candidate pool: %s", ranked)
    try:
        oof_all = build_oof_predictions(primary_df, model_builders=base_builders, model_names=ranked, n_splits=5)
        working = ranked
    except Exception as exc:
        logger.warning("Bulk OOF generation failed, falling back per-model: %s", exc)
        working = []
        base = pd.DataFrame(
            {
                "row_id": primary_df.index.to_numpy(),
                "game_id": primary_df["game_id"].astype(str).to_numpy(),
                "toi_hr": np.maximum(pd.to_numeric(primary_df["toi_hr"], errors="coerce").to_numpy(dtype=float), 1e-9),
                "y_true_total": np.clip(pd.to_numeric(primary_df["xg_for"], errors="coerce").to_numpy(dtype=float), 0, None),
            }
        )
        for m in ranked:
            try:
                o = build_oof_predictions(primary_df, model_builders=base_builders, model_names=[m], n_splits=5)
                base[f"mu_pred_total_{m}"] = o[f"mu_pred_total_{m}"]
                working.append(m)
            except Exception as exc_m:
                logger.warning("Skipping ensemble candidate %s due OOF failure: %s", m, exc_m)
        oof_all = base

    selected = select_diverse_models(
        oof_df=oof_all,
        ranked_model_names=working,
        max_models=max_models,
        corr_threshold=0.995,
        required_models=required,
    )
    for m in required:
        if m not in selected and m in working:
            selected.insert(0, m)
    selected = selected[:max_models]

    if len(selected) < 2:
        # Fallback safety.
        selected = working[: min(max_models, len(working))]

    keep_cols = ["row_id", "game_id", "toi_hr", "y_true_total"] + [f"mu_pred_total_{m}" for m in selected]
    oof_selected = oof_all[keep_cols].copy()
    oof_selected.to_parquet(outputs_dir / "oof_predictions_ev.parquet", index=False)

    logger.info("Selected ensemble base pool: %s", selected)
    return selected


def _plot_method(
    method_name: str,
    model,
    ev_df: pd.DataFrame,
    top10_df: pd.DataFrame,
    plots_dir: Path,
) -> None:
    pred_total = np.clip(model.predict_total(ev_df), 1e-9, None)
    actual = np.clip(ev_df["xg_for"].to_numpy(dtype=float), 0, None)
    resid = pred_total - actual

    n = len(actual)
    rng = np.random.default_rng(RANDOM_STATE)
    take = min(2000, n)
    idx = rng.choice(np.arange(n), size=take, replace=False) if n > take else np.arange(n)

    plt.figure(figsize=(7, 5))
    plt.scatter(actual[idx], pred_total[idx], s=10, alpha=0.4)
    lo = float(min(actual[idx].min(), pred_total[idx].min()))
    hi = float(max(actual[idx].max(), pred_total[idx].max()))
    plt.plot([lo, hi], [lo, hi], color="red", linewidth=1)
    plt.xlabel("Actual xG total")
    plt.ylabel("Predicted xG total")
    plt.title(f"Predicted vs Actual: {method_name}")
    plt.tight_layout()
    plt.savefig(plots_dir / f"predicted_vs_actual_xg_total_{method_name}.png", dpi=120)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.hist(resid, bins=40, color="#2d6a4f", alpha=0.85)
    plt.xlabel("Residual (pred - actual)")
    plt.ylabel("Count")
    plt.title(f"Residual Histogram: {method_name}")
    plt.tight_layout()
    plt.savefig(plots_dir / f"residual_hist_{method_name}.png", dpi=120)
    plt.close()

    plt.figure(figsize=(8, 5))
    bar_df = top10_df.sort_values("ratio", ascending=True)
    plt.barh(bar_df["team"], bar_df["ratio"], color="#1d3557")
    plt.xlabel("Disparity ratio")
    plt.ylabel("Team")
    plt.title(f"Top10 Ratio: {method_name}")
    plt.tight_layout()
    plt.savefig(plots_dir / f"top10_ratio_bar_{method_name}.png", dpi=120)
    plt.close()


def select_best_method(
    metrics_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
) -> tuple[str, str]:
    ok = metrics_summary[metrics_summary["status"] == "OK"].copy()
    ok = ok.dropna(subset=["cv_poisson_deviance_mean"]).sort_values("cv_poisson_deviance_mean")
    if ok.empty:
        raise RuntimeError("No successful method available for selection")

    top1 = ok.iloc[0]
    rationale = "Selected by lowest mean CV Poisson deviance (EV-only)."

    if len(ok) >= 2:
        top2 = ok.iloc[1]
        if top1["cv_poisson_deviance_mean"] > 0:
            rel_gap = (top2["cv_poisson_deviance_mean"] - top1["cv_poisson_deviance_mean"]) / top1[
                "cv_poisson_deviance_mean"
            ]
            if rel_gap <= 0.01:
                ss = stability_summary.set_index("method") if not stability_summary.empty else pd.DataFrame()
                s1 = ss.loc[top1["method"], "stability_score"] if (not ss.empty and top1["method"] in ss.index) else np.nan
                s2 = ss.loc[top2["method"], "stability_score"] if (not ss.empty and top2["method"] in ss.index) else np.nan
                if np.isfinite(s2) and (not np.isfinite(s1) or s2 < s1):
                    top1 = top2
                    rationale = (
                        "Top two methods were within 1% deviance; tie broken by lower bootstrap stability score."
                    )
                else:
                    rationale = (
                        "Top two methods were within 1% deviance; retained method with equal/better bootstrap stability score."
                    )

    return str(top1["method"]), rationale


def write_results_readme(
    out_path: Path,
    profile: dict,
    metrics_summary: pd.DataFrame,
    stability_summary: pd.DataFrame,
    best_method: str,
    best_rationale: str,
    final_top10: pd.DataFrame,
    ensemble_base_models: list[str],
) -> None:
    lines = []
    lines.append("# Phase 1b Results")
    lines.append("")
    lines.append("## Dataset Build Summary")
    lines.append(f"- rows_raw: {profile.get('rows_raw')}")
    lines.append(f"- rows_long: {profile.get('rows_long')}")
    lines.append(f"- rows_ev: {profile.get('rows_ev')}")
    lines.append(f"- rows_non_ev: {profile.get('rows_non_ev')}")
    lines.append(f"- unique_off_unit: {profile.get('unique_off_unit')}")
    lines.append(f"- unique_def_unit: {profile.get('unique_def_unit')}")
    lines.append("")

    lines.append("## Methods + Status")
    for _, row in metrics_summary.iterrows():
        lines.append(
            f"- {row['method']}: {row['status']} (cv_poisson_deviance_mean={row['cv_poisson_deviance_mean']})"
        )
    lines.append("")

    lines.append("## CV Metrics (sorted by mean CV Poisson deviance)")
    lines.append("```")
    lines.append(metrics_summary.to_string(index=False))
    lines.append("```")
    lines.append("")

    if not stability_summary.empty:
        lines.append("## Bootstrap Stability")
        lines.append("```")
        lines.append(stability_summary.to_string(index=False))
        lines.append("```")
    lines.append("")

    lines.append("## Ensemble Notes")
    lines.append("- Stacking and super learner variants were trained with grouped inner OOF predictions on training folds only.")
    lines.append("- Convex blend used nonnegative simplex weights minimizing OOF Poisson deviance (with small L2 grid).")
    lines.append("- Stacking used PoissonRegressor on log base predictions.")
    lines.append(f"- Ensemble base pool: {ensemble_base_models}")
    lines.append("")

    lines.append("## References")
    lines.append("- Wolpert (1992), Stacked Generalization: https://www.researchgate.net/publication/222467943_Stacked_Generalization")
    lines.append("- van der Laan et al. (2007), Super Learner: https://doi.org/10.2202/1544-6115.1309")
    lines.append("- Polley (2010), Super Learner technical report: https://biostats.bepress.com/ucbbiostat/paper222/")
    lines.append("- sklearn StackingRegressor API: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html")
    lines.append("- sklearn mean_poisson_deviance API: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_poisson_deviance.html")
    lines.append("- sklearn PoissonRegressor API: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PoissonRegressor.html")
    lines.append("")

    lines.append("## Best Method")
    lines.append(f"- method: {best_method}")
    lines.append(f"- rationale: {best_rationale}")
    lines.append("")

    lines.append("## Final Top 10")
    lines.append("```")
    lines.append(final_top10.to_string(index=False))
    lines.append("```")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def make_bootstrap_factory(method_name: str, fitted_model, default_factory):
    if fitted_model is None:
        return default_factory
    if method_name == "RIDGE_RAPM_RATE_SOFTPLUS":
        alpha = float(getattr(fitted_model, "best_alpha", 1.0))
        return lambda: RidgeRapmSoftplusModel(random_state=RANDOM_STATE, alpha_grid=[alpha])
    if method_name == "ELASTICNET_RAPM_RATE_SOFTPLUS":
        alpha = float(getattr(fitted_model, "best_alpha", 0.1))
        l1 = float(getattr(fitted_model, "best_l1_ratio", 0.5))
        return lambda: ElasticNetRapmSoftplusModel(
            random_state=RANDOM_STATE,
            alpha_grid=[alpha],
            l1_grid=[l1],
        )
    if method_name == "TWEEDIE_GLM_RATE":
        p = float(getattr(fitted_model, "best_power", 1.5))
        a = float(getattr(fitted_model, "best_alpha", 0.1))
        return lambda: TweedieGlmRateModel(random_state=RANDOM_STATE, power_grid=[p], alpha_grid=[a])
    if method_name == "POISSON_GLM_OFFSET_REG":
        a = float(getattr(fitted_model, "best_alpha", 0.1))
        l1 = float(getattr(fitted_model, "best_l1_wt", 0.5))
        return lambda: PoissonGlmOffsetRegModel(random_state=RANDOM_STATE, alpha_grid=[a], l1_grid=[l1])
    return default_factory


def write_ensemble_weights(out_path: Path, fitted_models: dict[str, object], base_pool: list[str]) -> None:
    payload: dict[str, object] = {
        "ensemble_base_pool": base_pool,
        "ensembles": {},
    }
    for method_name, model in fitted_models.items():
        if not method_name.startswith("ENSEMBLE_"):
            continue
        details = {}
        artifacts = getattr(model, "artifacts", None)
        if artifacts is not None:
            details = getattr(artifacts, "details", {}) or {}
            payload["ensembles"][method_name] = {
                "base_model_names": list(getattr(artifacts, "base_model_names", [])),
                "details": details,
            }
        else:
            payload["ensembles"][method_name] = {"details": {}}

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    repo_root = Path.cwd()
    outputs_dir = repo_root / "outputs"
    plots_dir = outputs_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(outputs_dir)
    logger.info("Starting Phase 1b pipeline")

    np.random.seed(RANDOM_STATE)
    os.environ["PYTHONHASHSEED"] = "0"

    paths = discover_paths(repo_root=repo_root, outputs_dir=outputs_dir)
    write_pdf_notes(paths=paths, out_path=outputs_dir / "pdf_notes.txt")

    whl_path = Path(paths["whl_2025"])
    logger.info("Loading whl data: %s", whl_path)
    raw = pd.read_csv(whl_path)

    long_df, ev_df, profile = build_canonical_long(raw_df=raw, outputs_dir=outputs_dir)
    logger.info("Built canonical long tables: long=%s ev=%s", len(long_df), len(ev_df))

    if ev_df.empty:
        logger.warning("EV subset is empty; falling back to ALL states for model selection")

    run_pytest(repo_root=repo_root, logger=logger)

    base_builders = get_model_builders(random_state=RANDOM_STATE)

    # Primary benchmark on EV-only (or fallback to all states if EV empty).
    primary_df = ev_df if not ev_df.empty else long_df
    ensemble_base_models = prepare_ensemble_base_pool(
        primary_df=primary_df,
        base_builders={k: v for k, v in base_builders.items() if not k.startswith("LOWRANK") and not k.startswith("BAYES")},
        outputs_dir=outputs_dir,
        logger=logger,
        max_models=6,
    )
    ensemble_builders = get_ensemble_model_builders(
        random_state=RANDOM_STATE,
        base_model_builders={k: base_builders[k] for k in ensemble_base_models if k in base_builders},
        base_model_names=ensemble_base_models,
    )
    builders = {**base_builders, **ensemble_builders}
    metrics_cv, status_ev = evaluate_methods(
        df=primary_df,
        builders=builders,
        logger=logger,
        dataset_label="EV",
        n_splits=5,
        include_time_split=True,
    )
    metrics_cv.to_csv(outputs_dir / "metrics_cv.csv", index=False)

    metrics_summary = summarize_metrics(metrics_cv, status_ev)
    metrics_summary.to_csv(outputs_dir / "metrics_summary.csv", index=False)

    # Sensitivity benchmark on all states.
    metrics_all, status_all = evaluate_methods(
        df=long_df,
        builders=base_builders,
        logger=logger,
        dataset_label="ALL_STATES",
        n_splits=3,
        include_time_split=False,
    )
    metrics_summary_all = summarize_metrics(metrics_all, status_all)
    metrics_summary_all.to_csv(outputs_dir / "metrics_summary_allstates.csv", index=False)

    # Full-data fit + top10 for each method.
    fit_df = primary_df
    top10_map, all_top10_df, fit_status, fitted_models = fit_full_and_rank(fit_df, builders, logger)
    all_top10_df.to_csv(outputs_dir / "all_methods_top10.csv", index=False)

    # Bootstrap stability: top-3 by deviance get 300 reps, others (successful) get 100.
    ok_methods = (
        metrics_summary[(metrics_summary["status"] == "OK") & metrics_summary["cv_poisson_deviance_mean"].notna()]
        .sort_values("cv_poisson_deviance_mean")
        ["method"]
        .tolist()
    )
    ok_methods_fast = [m for m in ok_methods if not m.startswith("ENSEMBLE_")]
    bootstrap_methods = ok_methods_fast[:3] if len(ok_methods_fast) >= 1 else ok_methods[:3]
    top_for_300 = set(bootstrap_methods[:1])

    stability_rows: list[dict] = []
    for method_name in bootstrap_methods:
        if method_name not in top10_map:
            continue
        full_top10_teams = top10_map[method_name]["team"].astype(str).tolist()
        heavy = method_name in {"LOWRANK_POISSON_FACTOR", "BAYES_HIER_POISSON_OFFSET"}
        n_boot = 20 if method_name in top_for_300 else 10
        if heavy:
            n_boot = min(n_boot, 50)
        logger.info("Bootstrapping stability for %s with B=%s", method_name, n_boot)

        try:
            bootstrap_factory = make_bootstrap_factory(method_name, fitted_models.get(method_name), builders[method_name])
            result = bootstrap_rank_stability(
                model_factory=bootstrap_factory,
                ev_df=fit_df,
                full_top10_teams=full_top10_teams,
                n_boot=n_boot,
                random_state=RANDOM_STATE,
                show_progress=False,
            )
            team_stats = result.team_stats
            mask = team_stats["team"].isin(full_top10_teams) if not team_stats.empty else []
            mean_rank_std = float(team_stats.loc[mask, "rank_std"].mean()) if not team_stats.empty else np.nan
            mean_top10_rate = float(team_stats.loc[mask, "top10_rate"].mean()) if not team_stats.empty else np.nan
            stability_rows.append(
                {
                    "method": method_name,
                    "status": "OK",
                    "n_boot": result.n_boot,
                    "stability_score": result.stability_score,
                    "mean_rank_std_full_top10": mean_rank_std,
                    "mean_top10_rate_full_top10": mean_top10_rate,
                    "note": "",
                }
            )
        except Exception as exc:
            logger.exception("Bootstrap failed for %s", method_name)
            stability_rows.append(
                {
                    "method": method_name,
                    "status": "FAILED",
                    "n_boot": n_boot,
                    "stability_score": np.nan,
                    "mean_rank_std_full_top10": np.nan,
                    "mean_top10_rate_full_top10": np.nan,
                    "note": str(exc),
                }
            )

    skipped_bootstrap = [m for m in ok_methods if m not in bootstrap_methods]
    for method_name in skipped_bootstrap:
        stability_rows.append(
            {
                "method": method_name,
                "status": "SKIPPED",
                "n_boot": 0,
                "stability_score": np.nan,
                "mean_rank_std_full_top10": np.nan,
                "mean_top10_rate_full_top10": np.nan,
                "note": "Bootstrap skipped to control runtime; fast top methods were bootstrapped.",
            }
        )

    stability_summary = pd.DataFrame(stability_rows)
    stability_summary.to_csv(outputs_dir / "stability_summary.csv", index=False)

    best_method, best_rationale = select_best_method(metrics_summary, stability_summary)

    if best_method not in top10_map:
        raise RuntimeError(f"Best method {best_method} has no full-fit top10 output")

    final_top10 = top10_map[best_method].copy()
    final_top10 = final_top10[["rank", "team", "line1_strength_xg60", "line2_strength_xg60", "ratio", "method"]]
    final_top10.to_csv(outputs_dir / "final_top10.csv", index=False)
    shutil.copyfile(outputs_dir / "final_top10.csv", outputs_dir / "submission_phase1b.csv")

    with (outputs_dir / "best_method.txt").open("w", encoding="utf-8") as f:
        f.write(f"best_method={best_method}\n")
        f.write(f"rationale={best_rationale}\n")

    write_ensemble_weights(
        out_path=outputs_dir / "ensemble_weights.json",
        fitted_models=fitted_models,
        base_pool=ensemble_base_models,
    )

    # Plots for each method that fit on full data.
    for method_name, model in fitted_models.items():
        top10_df = top10_map.get(method_name)
        if top10_df is None or top10_df.empty:
            continue
        try:
            _plot_method(method_name, model, fit_df, top10_df, plots_dir)
        except Exception:
            logger.exception("Plotting failed for %s", method_name)

    write_results_readme(
        out_path=outputs_dir / "README_RESULTS.md",
        profile=profile,
        metrics_summary=metrics_summary,
        stability_summary=stability_summary,
        best_method=best_method,
        best_rationale=best_rationale,
        final_top10=final_top10,
        ensemble_base_models=ensemble_base_models,
    )

    logger.info("Methods sorted by mean CV Poisson deviance:\n%s", metrics_summary.to_string(index=False))
    logger.info("Best method: %s | %s", best_method, best_rationale)
    logger.info("Best method Top 10:\n%s", final_top10.to_string(index=False))

    print("Methods sorted by mean CV Poisson deviance")
    cols = ["method", "cv_poisson_deviance_mean", "cv_poisson_deviance_std", "status"]
    print(metrics_summary[cols].to_string(index=False))
    print()
    print(f"Best method: {best_method}")
    print(best_rationale)
    print()
    print("Best method Top 10 teams")
    print(final_top10.to_string(index=False))


if __name__ == "__main__":
    main()
