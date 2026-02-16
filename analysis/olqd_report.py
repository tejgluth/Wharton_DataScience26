from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.olqd import compute_team_olqd_table, evaluate_baseline_vs_olqd
from phases.common import load_ev_dataset, resolve_phase1b_config, setup_logger
from whsdsci.ensemble.search import FittedSearchModel
from whsdsci.models import get_model_builders


def _team_strength_table(paths: dict, ev_df: pd.DataFrame) -> pd.DataFrame:
    league_path = paths.get("league_table")
    if league_path and Path(league_path).exists():
        league = pd.read_csv(league_path)
        if {"team", "points", "rank"}.issubset(set(league.columns)):
            return league[["team", "points", "rank"]].copy()
    fallback = (
        ev_df.groupby("offense_team", as_index=False)["xg_diff"]
        .sum()
        .rename(columns={"offense_team": "team", "xg_diff": "points"})
        .sort_values("points", ascending=False)
        .reset_index(drop=True)
    )
    fallback["rank"] = np.arange(1, len(fallback) + 1)
    return fallback


def _plot_ratio_vs_points(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(df["olqd_ratio"], df["points"], s=44, alpha=0.85)
    if len(df) >= 2:
        x = pd.to_numeric(df["olqd_ratio"], errors="coerce").to_numpy(dtype=float)
        y = pd.to_numeric(df["points"], errors="coerce").to_numpy(dtype=float)
        a, b = np.polyfit(x, y, deg=1)
        xx = np.linspace(np.min(x), np.max(x), 120)
        ax.plot(xx, a * xx + b, linewidth=2)
    for _, r in df.sort_values("olqd_ratio", ascending=False).head(8).iterrows():
        ax.annotate(str(r["team"]), (r["olqd_ratio"], r["points"]), fontsize=8)
    ax.set_title("Offensive Line Quality Disparity vs Team Strength")
    ax.set_xlabel("OLQD Ratio (Line1 / Line2)")
    ax.set_ylabel("Team Strength (Points)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_fold_deltas(fold_metrics: pd.DataFrame, out_path: Path) -> None:
    piv = fold_metrics.pivot(index="fold_id", columns="method", values="poisson_deviance").reset_index()
    piv = piv.sort_values("fold_id")
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(len(piv))
    if "baseline" in piv.columns:
        ax.plot(x, piv["baseline"], marker="o", label="baseline")
    if "baseline+olqd" in piv.columns:
        ax.plot(x, piv["baseline+olqd"], marker="o", label="baseline+olqd")
    if {"baseline", "baseline+olqd"}.issubset(set(piv.columns)):
        delta = piv["baseline+olqd"] - piv["baseline"]
        for i, d in enumerate(delta):
            ax.text(i, max(piv["baseline+olqd"].iloc[i], piv["baseline"].iloc[i]) + 5e-5, f"{d:+.5f}", fontsize=7)
    ax.set_xticks(x, piv["fold_id"].tolist(), rotation=45)
    ax.set_ylabel("Poisson Deviance (lower is better)")
    ax.set_title("Ablation by Fold: Baseline vs Baseline+OLQD")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_olqd_report(
    config_name: str | None,
    out_dir: Path,
    seed: int,
    small: bool,
) -> dict:
    repo_root = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    logger = setup_logger("olqd_report", out_dir / "phase1d_olqd.log")
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)

    cfg, payload = resolve_phase1b_config(config_name=config_name, outputs_dir=repo_root / "outputs")
    ev_df, paths = load_ev_dataset(repo_root=repo_root, outputs_dir=repo_root / "outputs")
    if small:
        game_ids = sorted(ev_df["game_id"].astype(str).unique())[:80]
        ev_df = ev_df[ev_df["game_id"].astype(str).isin(game_ids)].reset_index(drop=True)
        n_splits = 3
    else:
        n_splits = 5
    n_splits = min(n_splits, max(2, ev_df["game_id"].astype(str).nunique()))

    logger.info("Running OLQD ablation config=%s rows=%s splits=%s", cfg.config_id, len(ev_df), n_splits)
    ab = evaluate_baseline_vs_olqd(ev_df=ev_df, cfg=cfg, seed=seed, n_splits=n_splits)
    ab.fold_metrics.to_csv(out_dir / "olqd_ablation_fold_metrics.csv", index=False)
    ab.summary.to_csv(out_dir / "olqd_ablation_summary.csv", index=False)
    ab.delta_summary.to_csv(out_dir / "olqd_ablation_delta.csv", index=False)
    ab.oof_predictions.to_parquet(out_dir / "olqd_ablation_oof.parquet", index=False)

    # Full-data OLQD table from chosen baseline config.
    full_model = FittedSearchModel(
        random_state=seed,
        base_builders=get_model_builders(random_state=seed),
        config=cfg,
    ).fit(ev_df)
    team_olqd = compute_team_olqd_table(model=full_model, train_ev_df=ev_df)
    team_strength = _team_strength_table(paths=paths, ev_df=ev_df)
    viz_df = team_olqd.merge(team_strength, on="team", how="inner").sort_values("olqd_ratio", ascending=False).reset_index(drop=True)
    viz_df.to_csv(out_dir / "olqd_team_table.csv", index=False)

    _plot_ratio_vs_points(viz_df, fig_dir / "olqd_ratio_vs_team_strength.png")
    _plot_fold_deltas(ab.fold_metrics, fig_dir / "olqd_ablation_by_fold.png")

    delta_mean = float(ab.delta_summary.iloc[0]["mean_delta"])
    ci = float(ab.delta_summary.iloc[0]["ci95_half_width"])

    md = []
    md.append("# Phase 1d (Offensive Line Quality Disparity)")
    md.append("")
    md.append("## Definition")
    md.append(
        "Offensive Line Quality Disparity (OLQD) is the team-level ratio of standardized first-line offensive strength to second-line offensive strength."
    )
    md.append("")
    md.append("## How OLQD Was Computed")
    md.append("- Fit the selected Phase 1b model on EV data.")
    md.append("- Compute schedule-standardized line strengths (`xG/60`) via marginalization over defense usage.")
    md.append("- Compute `OLQD = line1_strength_xg60 / line2_strength_xg60` per team.")
    md.append("")
    md.append("## Why It Matters")
    md.append(
        "The competition question asks whether teams with more even offensive lines perform better. OLQD provides a compact measure of top-line concentration versus balanced depth."
    )
    md.append("")
    md.append("## Baseline vs Baseline+OLQD Ablation (Grouped CV)")
    md.append("```")
    md.append(ab.summary.to_string(index=False))
    md.append("```")
    md.append("")
    md.append("Per-fold deviance delta (`baseline+olqd - baseline`):")
    md.append("```")
    md.append(ab.delta_summary.to_string(index=False))
    md.append("```")
    md.append("")
    md.append("Interpretation:")
    md.append(
        f"- Mean delta = {delta_mean:.9f} with CI half-width ≈ {ci:.9f}. Negative favors OLQD augmentation; positive favors baseline."
    )
    md.append(
        "- In this run, OLQD-derived context was tested as an additive correction layer to isolate whether disparity signal improves out-of-sample xG prediction."
    )
    md.append("")
    md.append("## Figures")
    md.append("- `figures/olqd_ratio_vs_team_strength.png`")
    md.append("- `figures/olqd_ablation_by_fold.png`")
    md.append("")
    md.append("## Model Behavior and Error Modes")
    md.append(
        "OLQD features mainly target systematic errors where baseline predictions under/over-estimate teams with unusually concentrated top-line offense. "
        "If fold deltas are near zero, disparity information may already be captured by matchup-adjusted base predictors."
    )
    (out_dir / "phase1d_offensive_line_quality_disparity.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    result = {
        "config_id": cfg.config_id,
        "mean_delta_poisson_deviance": delta_mean,
        "ci95_half_width": ci,
        "summary": ab.summary.to_dict(orient="records"),
        "source_config": payload,
    }
    (out_dir / "phase1d_olqd_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("OLQD report outputs written to %s", out_dir)
    return result


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate Phase 1d OLQD section artifacts")
    ap.add_argument("--config", type=str, default=None, help="Config id (e.g., cfg_00699) or JSON path")
    ap.add_argument("--out", type=str, default="reports/")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--small", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    result = run_olqd_report(
        config_name=args.config,
        out_dir=Path(args.out),
        seed=args.seed,
        small=args.small,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

