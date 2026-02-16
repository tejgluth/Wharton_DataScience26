from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from phases.common import get_git_commit, load_ev_dataset, resolve_phase1b_config, setup_logger, write_simple_yaml
from whsdsci.models.tree_poisson_best import TreePoissonBestModel
from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


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
    small: bool = False,
) -> dict:
    repo_root = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("phase1c", out_dir / "run.log")
    np.random.seed(seed)

    cfg, payload = resolve_phase1b_config(config_name=config_name, outputs_dir=repo_root / "outputs")
    logger.info("Using config: %s", payload)
    ev_df, paths = load_ev_dataset(repo_root=repo_root, outputs_dir=out_dir)
    if small:
        keep_games = sorted(ev_df["game_id"].astype(str).unique())[:80]
        ev_df = ev_df[ev_df["game_id"].astype(str).isin(keep_games)].reset_index(drop=True)

    model = TreePoissonBestModel(config=cfg, outputs_dir=repo_root / "outputs", random_state=seed).fit(ev_df)
    strengths = compute_standardized_strengths(model=model, train_ev_df=ev_df)
    ratios = compute_disparity_ratios(strengths)
    team_strength = _team_strength_table(paths=paths, ev_df=ev_df)
    viz_df = ratios.merge(team_strength, on="team", how="inner").sort_values("ratio", ascending=False).reset_index(drop=True)

    _make_phase1c_plot(viz_df, out_dir / "phase1c_line_disparity_vs_team_strength.png")
    viz_df.to_csv(out_dir / "phase1c_viz_table.csv", index=False)
    viz_df.to_csv(Path("outputs") / "phase1c_output.csv", index=False)

    config_export = {
        "config_id": cfg.config_id,
        "combiner_family": cfg.combiner_family,
        "base_pool_id": cfg.base_pool_id,
        "base_models": cfg.base_models,
        "calibration_type": cfg.calibration_type,
        "hyperparams": cfg.hyperparams,
        "seed": seed,
        "small_mode": bool(small),
        "git_commit": get_git_commit(repo_root),
    }
    (out_dir / "best_config.json").write_text(json.dumps(config_export, indent=2), encoding="utf-8")
    write_simple_yaml(out_dir / "best_config.yaml", config_export)

    md = []
    md.append("# Phase 1c Output")
    md.append("")
    md.append(f"- config_id: `{cfg.config_id}`")
    md.append(f"- rows_ev_used: `{len(ev_df)}`")
    md.append("- artifact: `phase1c_line_disparity_vs_team_strength.png`")
    md.append("- table: `phase1c_viz_table.csv`")
    (out_dir / "phase1c_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    logger.info("Phase 1c artifacts written to %s", out_dir)
    return {"config_id": cfg.config_id, "rows_ev": int(len(ev_df)), "out_dir": str(out_dir)}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run Phase 1c visualization")
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=str, default="outputs/phase1c/")
    ap.add_argument("--small", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    result = run_phase1c(
        config_name=args.config,
        seed=args.seed,
        out_dir=Path(args.out),
        small=args.small,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

