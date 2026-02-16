from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.olqd import evaluate_baseline_vs_olqd, fit_predict_baseline_plus_olqd
from phases.common import (
    get_git_commit,
    load_ev_dataset,
    resolve_phase1b_config,
    setup_logger,
    write_simple_yaml,
)
from whsdsci.eval.cv import make_group_kfold_splits
from whsdsci.eval.metrics import calibration_ratio, mae_total, poisson_deviance_safe, weighted_mse_rate
from whsdsci.models.tree_poisson_best import TreePoissonBestModel
from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


def _cv_baseline_predictions(
    ev_df: pd.DataFrame,
    cfg,
    seed: int,
    n_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = ev_df.reset_index(drop=False).rename(columns={"index": "row_id"}).copy()
    splits = make_group_kfold_splits(work, n_splits=n_splits)

    rows = []
    oof_parts = []
    for split in splits:
        tr = work.iloc[split.train_idx].reset_index(drop=True)
        te = work.iloc[split.test_idx].reset_index(drop=True)
        model = TreePoissonBestModel(config=cfg, outputs_dir=Path("outputs"), random_state=seed).fit(tr)
        pred = np.clip(model.predict_total(te), 1e-9, None)
        y = np.clip(pd.to_numeric(te["xg_for"], errors="coerce").to_numpy(dtype=float), 0, None)
        toi = np.maximum(pd.to_numeric(te["toi_hr"], errors="coerce").to_numpy(dtype=float), 1e-9)
        rows.append(
            {
                "method": "baseline",
                "fold_id": split.fold_id,
                "poisson_deviance": float(poisson_deviance_safe(y, pred)),
                "weighted_mse_rate": float(weighted_mse_rate(y / toi, pred / toi, toi)),
                "mae_total": float(mae_total(y, pred)),
                "calibration_ratio": float(calibration_ratio(y, pred)),
                "n_test": int(len(te)),
            }
        )
        oof_parts.append(
            pd.DataFrame(
                {
                    "row_id": te["row_id"].to_numpy(),
                    "game_id": te["game_id"].astype(str).to_numpy(),
                    "y_true_total": y,
                    "toi_hr": toi,
                    "pred_total": pred,
                    "feature_mode": "baseline",
                }
            )
        )
    fold_df = pd.DataFrame(rows)
    oof = pd.concat(oof_parts, ignore_index=True) if oof_parts else pd.DataFrame()
    return fold_df, oof


def _metric_summary(fold_df: pd.DataFrame) -> dict[str, Any]:
    return {
        "n_folds": int(fold_df["fold_id"].nunique()),
        "mean_poisson_deviance": float(fold_df["poisson_deviance"].mean()),
        "std_poisson_deviance": float(fold_df["poisson_deviance"].std(ddof=0)),
        "mean_weighted_mse_rate": float(fold_df["weighted_mse_rate"].mean()),
        "mean_mae_total": float(fold_df["mae_total"].mean()),
        "mean_calibration_ratio": float(fold_df["calibration_ratio"].mean()),
    }


def _team_strength_table(paths: dict[str, Any], ev_df: pd.DataFrame) -> pd.DataFrame:
    league_path = paths.get("league_table")
    if league_path and Path(league_path).exists():
        league = pd.read_csv(league_path)
        if {"team", "points", "rank"}.issubset(set(league.columns)):
            return league[["team", "points", "rank"]].copy()
    # fallback proxy if optional file missing
    g = (
        ev_df.groupby("offense_team", as_index=False)["xg_diff"]
        .sum()
        .rename(columns={"offense_team": "team", "xg_diff": "points"})
        .sort_values("points", ascending=False)
        .reset_index(drop=True)
    )
    g["rank"] = np.arange(1, len(g) + 1)
    return g


def _make_phase1c_plot(df: pd.DataFrame, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["ratio"], df["points"], s=42, alpha=0.85)
    if len(df) >= 2:
        x = pd.to_numeric(df["ratio"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["points"], errors="coerce").to_numpy(dtype=float)
        a, b = np.polyfit(x, y, deg=1)
        xx = np.linspace(np.min(x), np.max(x), 100)
        ax.plot(xx, a * xx + b, linewidth=2)
    top_label = df.sort_values("ratio", ascending=False).head(8)
    for _, r in top_label.iterrows():
        ax.annotate(str(r["team"]), (r["ratio"], r["points"]), fontsize=8, alpha=0.85)
    ax.set_title("Phase 1c: Offensive Line Disparity vs Team Strength")
    ax.set_xlabel("Offensive Line Quality Disparity Ratio (Line1 / Line2)")
    ax.set_ylabel("Team Strength (Points)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def run_phase1c(
    config_name: str | None,
    seed: int,
    out_dir: Path,
    small: bool,
    features: str,
) -> dict[str, Any]:
    repo_root = Path.cwd()
    logger = setup_logger("phase1c", out_dir / "run.log")
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    cfg, payload = resolve_phase1b_config(config_name=config_name, outputs_dir=repo_root / "outputs")
    logger.info("Using config: %s", payload)
    ev_df, paths = load_ev_dataset(repo_root=repo_root, outputs_dir=out_dir)

    if small:
        game_ids = sorted(ev_df["game_id"].astype(str).unique())[:80]
        ev_df = ev_df[ev_df["game_id"].astype(str).isin(game_ids)].reset_index(drop=True)
        n_splits = 3
    else:
        n_splits = 5
    n_splits = min(n_splits, max(2, ev_df["game_id"].astype(str).nunique()))
    logger.info("EV rows=%s unique_games=%s n_splits=%s small=%s", len(ev_df), ev_df["game_id"].nunique(), n_splits, small)

    if features == "baseline":
        fold_metrics, oof = _cv_baseline_predictions(ev_df=ev_df, cfg=cfg, seed=seed, n_splits=n_splits)
        metric_summary = _metric_summary(fold_metrics)
    else:
        ab = evaluate_baseline_vs_olqd(ev_df=ev_df, cfg=cfg, seed=seed, n_splits=n_splits)
        fold_metrics = ab.fold_metrics.copy()
        metric_summary = {
            **_metric_summary(fold_metrics[fold_metrics["method"] == "baseline+olqd"]),
            "delta_poisson_deviance_mean": float(ab.delta_summary.iloc[0]["mean_delta"]),
            "delta_poisson_deviance_ci95_half_width": float(ab.delta_summary.iloc[0]["ci95_half_width"]),
        }
        oof = ab.oof_predictions.rename(columns={"pred_baseline_olqd_total": "pred_total"}).copy()
        oof["feature_mode"] = "baseline+olqd"

    # Full model fit for Phase 1b disparity + Phase 1c visualization relationship
    full_model = TreePoissonBestModel(config=cfg, outputs_dir=Path("outputs"), random_state=seed).fit(ev_df)
    strengths = compute_standardized_strengths(model=full_model, train_ev_df=ev_df)
    ratios = compute_disparity_ratios(strengths)
    team_strength = _team_strength_table(paths=paths, ev_df=ev_df)
    viz_df = ratios.merge(team_strength, on="team", how="inner").sort_values("ratio", ascending=False).reset_index(drop=True)
    _make_phase1c_plot(viz_df, out_dir / "phase1c_line_disparity_vs_team_strength.png")

    # Full predictions from selected model mode.
    full_pred = np.clip(full_model.predict_total(ev_df), 1e-9, None)
    if features == "baseline+olqd":
        _, _, full_pred_olqd, _ = fit_predict_baseline_plus_olqd(
            train_df=ev_df,
            infer_df=ev_df,
            cfg=cfg,
            seed=seed,
        )
        full_pred = np.clip(full_pred_olqd, 1e-9, None)
    full_pred_df = pd.DataFrame(
        {
            "row_id": ev_df.index.to_numpy(),
            "game_id": ev_df["game_id"].astype(str).to_numpy(),
            "y_true_total": np.clip(pd.to_numeric(ev_df["xg_for"], errors="coerce").to_numpy(dtype=float), 0, None),
            "toi_hr": np.maximum(pd.to_numeric(ev_df["toi_hr"], errors="coerce").to_numpy(dtype=float), 1e-9),
            "pred_total": full_pred,
            "feature_mode": features,
        }
    )

    # Persist artifacts.
    out_dir.mkdir(parents=True, exist_ok=True)
    oof.to_parquet(out_dir / "predictions_oof.parquet", index=False)
    oof.to_csv(out_dir / "predictions_oof.csv", index=False)
    full_pred_df.to_parquet(out_dir / "predictions_full.parquet", index=False)
    full_pred_df.to_csv(out_dir / "predictions_full.csv", index=False)
    viz_df.to_csv(out_dir / "phase1c_viz_table.csv", index=False)

    config_export = {
        "config_id": cfg.config_id,
        "combiner_family": cfg.combiner_family,
        "base_pool_id": cfg.base_pool_id,
        "base_models": cfg.base_models,
        "calibration_type": cfg.calibration_type,
        "hyperparams": cfg.hyperparams,
        "selected_feature_mode": features,
        "seed": seed,
    }
    (out_dir / "best_config.json").write_text(json.dumps(config_export, indent=2), encoding="utf-8")
    write_simple_yaml(out_dir / "best_config.yaml", config_export)

    metadata = {
        "seed": seed,
        "small_mode": small,
        "n_rows_ev": int(len(ev_df)),
        "n_unique_games": int(ev_df["game_id"].astype(str).nunique()),
        "n_splits": n_splits,
        "git_commit": get_git_commit(repo_root),
    }
    (out_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    metrics_payload = {
        "feature_mode": features,
        "config_id": cfg.config_id,
        "config_family": cfg.combiner_family,
        **metric_summary,
    }
    (out_dir / "metrics_summary.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    md = []
    md.append("# Phase 1c Metrics Summary")
    md.append("")
    md.append(f"- config_id: `{cfg.config_id}`")
    md.append(f"- feature_mode: `{features}`")
    md.append(f"- mean_poisson_deviance: `{metrics_payload['mean_poisson_deviance']:.9f}`")
    md.append(f"- std_poisson_deviance: `{metrics_payload['std_poisson_deviance']:.9f}`")
    if "delta_poisson_deviance_mean" in metrics_payload:
        md.append(f"- delta_poisson_deviance_mean (baseline+olqd - baseline): `{metrics_payload['delta_poisson_deviance_mean']:.9f}`")
        md.append(f"- delta_poisson_deviance_ci95_half_width: `{metrics_payload['delta_poisson_deviance_ci95_half_width']:.9f}`")
    md.append("")
    md.append("## Phase 1c Deliverables")
    md.append("- `phase1c_line_disparity_vs_team_strength.png`")
    md.append("- `phase1c_viz_table.csv`")
    md.append("- `predictions_oof.parquet`")
    md.append("- `predictions_full.parquet`")
    (out_dir / "metrics_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    logger.info("Phase 1c artifacts written to %s", out_dir)
    return {"metrics": metrics_payload, "config": config_export}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Phase 1c pipeline")
    ap.add_argument("--config", type=str, default=None, help="Config id (e.g., cfg_00699) or config JSON path")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="artifacts/phase1c/")
    ap.add_argument("--small", action="store_true", help="Quick sanity mode (<2 mins target)")
    ap.add_argument(
        "--features",
        choices=["baseline", "baseline+olqd"],
        default="baseline",
        help="Feature set / prediction mode",
    )
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    result = run_phase1c(
        config_name=args.config,
        seed=args.seed,
        out_dir=Path(args.out),
        small=args.small,
        features=args.features,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
