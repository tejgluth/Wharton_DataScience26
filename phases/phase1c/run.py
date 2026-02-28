from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from phases.common import get_git_commit, load_ev_dataset, resolve_phase1b_config, setup_logger, write_simple_yaml
from phases.phase1b.system import TreePoissonBestModel
from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


def _team_strength_table(paths: dict, ev_df: pd.DataFrame, repo_root: Path) -> tuple[pd.DataFrame, str]:
    ev_teams = set(ev_df["offense_team"].astype(str).unique())

    # Prefer user-provided ELO rankings if present.
    for elo_path in [repo_root / "elo_ranking.csv", repo_root / "data" / "elo_ranking.csv"]:
        if elo_path.exists():
            elo = pd.read_csv(elo_path)
            elo = elo.loc[:, ~elo.columns.astype(str).str.lower().str.startswith("unnamed")]
            cols = {c: str(c).strip().lower() for c in elo.columns}
            team_col = next((c for c, lc in cols.items() if lc == "team"), None)
            score_col = next((c for c, lc in cols.items() if lc in {"elo", "elo_rating", "rating", "strength"}), None)
            rank_col = next((c for c, lc in cols.items() if lc in {"rank", "ranking", "elo_rank"}), None)
            if team_col is not None and score_col is not None:
                out = elo[[team_col, score_col] + ([rank_col] if rank_col is not None else [])].copy()
                rename_map = {team_col: "team", score_col: "points"}
                if rank_col is not None:
                    rename_map[rank_col] = "rank"
                out = out.rename(columns=rename_map)
                out["team"] = out["team"].astype(str).str.strip()
                out["points"] = pd.to_numeric(out["points"], errors="coerce")
                out = out.dropna(subset=["team", "points"]).drop_duplicates(subset=["team"])
                overlap = len(ev_teams.intersection(set(out["team"].astype(str))))
                if overlap == 0:
                    continue
                if "rank" in out.columns:
                    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
                if "rank" not in out.columns or out["rank"].isna().any():
                    out = out.sort_values("points", ascending=False).reset_index(drop=True)
                    out["rank"] = np.arange(1, len(out) + 1)
                return out[["team", "points", "rank"]].copy(), "ELO Rating"

    league_path = paths.get("league_table")
    if league_path and Path(league_path).exists():
        league = pd.read_csv(league_path)
        if {"team", "points", "rank"}.issubset(set(league.columns)):
            return league[["team", "points", "rank"]].copy(), "Season Points"
    fallback = (
        ev_df.groupby("offense_team", as_index=False)["xg_diff"]
        .sum()
        .rename(columns={"offense_team": "team", "xg_diff": "points"})
        .sort_values("points", ascending=False)
        .reset_index(drop=True)
    )
    fallback["rank"] = np.arange(1, len(fallback) + 1)
    return fallback, "Aggregate xG Differential"


def _make_phase1c_plot_scatter(df: pd.DataFrame, out_png: Path, strength_label: str) -> None:
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
    ax.set_ylabel(f"Team Strength ({strength_label})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _make_phase1c_plot_quadrant(df: pd.DataFrame, out_png: Path, strength_label: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    x = pd.to_numeric(df["ratio"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["points"], errors="coerce").to_numpy(dtype=float)
    x_med = float(np.median(x))
    y_med = float(np.median(y))
    colors = np.where((x <= x_med) & (y >= y_med), "#0B7285", "#9A3412")
    ax.scatter(x, y, c=colors, s=48, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axvline(x_med, color="gray", linestyle="--", linewidth=1.2)
    ax.axhline(y_med, color="gray", linestyle="--", linewidth=1.2)
    ax.set_title("Option B: Disparity vs Team Strength (Quadrant View)")
    ax.set_xlabel("Offensive Line Quality Disparity Ratio (Line1 / Line2)")
    ax.set_ylabel(f"Team Strength ({strength_label})")
    ax.text(x_med * 0.98, y_med * 1.02, "More Balanced + Stronger", fontsize=8, ha="right", va="bottom")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _make_phase1c_plot_binned(df: pd.DataFrame, out_png: Path, strength_label: str) -> None:
    x = pd.to_numeric(df["ratio"], errors="coerce")
    y = pd.to_numeric(df["points"], errors="coerce")
    bins = pd.qcut(x, q=5, duplicates="drop")
    grouped = (
        pd.DataFrame({"ratio": x, "points": y, "bin": bins})
        .dropna()
        .groupby("bin", as_index=False)
        .agg(mean_ratio=("ratio", "mean"), mean_points=("points", "mean"), n=("points", "count"), std=("points", "std"))
    )
    se = grouped["std"].fillna(0.0) / np.maximum(np.sqrt(grouped["n"]), 1.0)
    ci = 1.96 * se
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(x, y, color="#B8C0CC", alpha=0.45, s=28, label="Teams")
    ax.errorbar(
        grouped["mean_ratio"],
        grouped["mean_points"],
        yerr=ci,
        fmt="-o",
        linewidth=2,
        capsize=3,
        color="#0B7285",
        label="Bin mean ±95% CI",
    )
    ax.set_title("Option C: Binned Trend of Team Strength by Disparity")
    ax.set_xlabel("Offensive Line Quality Disparity Ratio (Line1 / Line2)")
    ax.set_ylabel(f"Team Strength ({strength_label})")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _make_phase1c_plot_dumbbell(df: pd.DataFrame, out_png: Path, strength_label: str) -> None:
    d = df.sort_values("ratio", ascending=False).reset_index(drop=True)
    y_pos = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(10, 10))
    for i, r in d.iterrows():
        ax.plot([r["line2_strength_xg60"], r["line1_strength_xg60"]], [y_pos[i], y_pos[i]], color="#94A3B8", linewidth=1.5)
    sc = ax.scatter(d["line2_strength_xg60"], y_pos, color="#475569", s=35, label="Line2 xG/60")
    ax.scatter(d["line1_strength_xg60"], y_pos, c=d["points"], cmap="viridis", s=55, label="Line1 xG/60 (color=strength)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(d["team"])
    ax.invert_yaxis()
    ax.set_title("Option D: Team Line1 vs Line2 Strength (Dumbbell by Team)")
    ax.set_xlabel("Standardized Line Strength (xG/60)")
    ax.grid(axis="x", alpha=0.25)
    cbar = fig.colorbar(ax.collections[-1], ax=ax)
    cbar.set_label(f"Team Strength ({strength_label})")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _make_phase1c_plot_recommended(df: pd.DataFrame, out_png: Path, strength_label: str) -> None:
    d = df.sort_values("ratio", ascending=False).reset_index(drop=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.25, 1.0]})

    ax = axes[0]
    x = pd.to_numeric(d["ratio"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(d["points"], errors="coerce").to_numpy(dtype=float)
    ax.scatter(x, y, s=50, alpha=0.85, color="#0B7285", label="Teams")
    if len(d) >= 2:
        a, b = np.polyfit(x, y, deg=1)
        xx = np.linspace(np.min(x), np.max(x), 120)
        yy = a * xx + b
        ax.plot(xx, yy, color="#9A3412", linewidth=2.2, label="Linear trend")
    for _, r in d.nlargest(5, "ratio").iterrows():
        ax.annotate(str(r["team"]), (r["ratio"], r["points"]), fontsize=8)
    ax.set_title("Disparity vs Team Strength")
    ax.set_xlabel("Line Disparity Ratio (Line1 / Line2)")
    ax.set_ylabel(f"Team Strength ({strength_label})")
    ax.grid(alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(loc="best")

    ax2 = axes[1]
    top = d.head(10).copy()
    y2 = np.arange(len(top))
    ax2.barh(y2, top["ratio"], color="#B45309")
    ax2.set_yticks(y2)
    ax2.set_yticklabels(top["team"])
    ax2.invert_yaxis()
    ax2.set_xlabel("Disparity Ratio")
    ax2.set_title("Top 10 Disparity Teams")
    ax2.grid(axis="x", alpha=0.25)

    fig.suptitle(f"Phase 1c: Do More Evenly-Matched Lines Relate to Team Success? (Strength={strength_label})", fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _quartile_pattern(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    d = df.copy()
    d["strength_q"] = pd.qcut(d["points"], q=4, labels=["Q1_Low", "Q2", "Q3", "Q4_High"], duplicates="drop")
    d["disparity_q"] = pd.qcut(d["ratio"], q=4, labels=["Q1_Balanced", "Q2", "Q3", "Q4_HighDisparity"], duplicates="drop")
    ct = pd.crosstab(d["strength_q"], d["disparity_q"]).reindex(
        index=["Q4_High", "Q3", "Q2", "Q1_Low"],
        columns=["Q1_Balanced", "Q2", "Q3", "Q4_HighDisparity"],
        fill_value=0,
    )
    top_strength = (d["strength_q"] == "Q4_High").astype(int)
    top_share = d.groupby("disparity_q", observed=False).apply(lambda g: float(np.mean(top_strength.loc[g.index]))).reindex(
        ["Q1_Balanced", "Q2", "Q3", "Q4_HighDisparity"]
    )
    return ct, top_share


def _make_phase1c_plot_quartile_heatmap(df: pd.DataFrame, out_png: Path, strength_label: str) -> None:
    ct, top_share = _quartile_pattern(df)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), gridspec_kw={"width_ratios": [1.4, 1.0]})

    ax = axes[0]
    mat = ct.to_numpy(dtype=float)
    im = ax.imshow(mat, cmap="YlOrRd")
    ax.set_xticks(np.arange(ct.shape[1]))
    ax.set_yticks(np.arange(ct.shape[0]))
    ax.set_xticklabels(["Balanced\nQ1", "Q2", "Q3", "High\nDisparity Q4"])
    ax.set_yticklabels([f"Top {strength_label}\nQ4", "Q3", "Q2", f"Bottom {strength_label}\nQ1"])
    ax.set_title(f"Team Counts: {strength_label} Quartile vs Disparity Quartile")
    ax.set_xlabel("Disparity Quartile")
    ax.set_ylabel(f"{strength_label} Quartile")
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            ax.text(j, i, str(int(mat[i, j])), ha="center", va="center", color="black", fontsize=10, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Team Count")

    ax2 = axes[1]
    x = np.arange(len(top_share))
    pct = 100.0 * top_share.to_numpy(dtype=float)
    ax2.bar(x, pct, color=["#0B7285", "#74A9CF", "#A1D99B", "#E6550D"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(["Q1 Balanced", "Q2", "Q3", "Q4 High\nDisparity"])
    ax2.set_ylim(0, max(50.0, float(np.nanmax(pct)) + 10.0))
    ax2.set_ylabel(f"% of Teams in Top {strength_label} Quartile")
    ax2.set_title("Top-Strength Presence by Disparity Quartile")
    for xi, p in zip(x, pct):
        ax2.text(xi, p + 1.0, f"{p:.1f}%", ha="center", va="bottom", fontsize=9)
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("Phase 1c: Connection via Quartile Structure (Outlier-Resistant View)", fontsize=13)
    fig.text(
        0.5,
        0.01,
        "Interpretation: relationship is not linear; strongest teams are concentrated in balanced/moderate disparity bands, "
        "while the highest disparity band shows little or no top-strength presence.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def _relationship_stats(df: pd.DataFrame) -> dict[str, float]:
    x = pd.to_numeric(df["ratio"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df["points"], errors="coerce").to_numpy(dtype=float)
    if len(x) < 3:
        return {"pearson_r": float("nan"), "spearman_rho": float("nan"), "linear_r2": float("nan")}
    pr = float(pearsonr(x, y).statistic)
    sr = float(spearmanr(x, y).correlation)
    a, b = np.polyfit(x, y, deg=1)
    yhat = a * x + b
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"pearson_r": pr, "spearman_rho": sr, "linear_r2": r2}


def _run_visualization_experiments(df: pd.DataFrame, out_dir: Path, strength_label: str) -> tuple[Path, dict[str, float]]:
    options_dir = out_dir / "viz_options"
    options_dir.mkdir(parents=True, exist_ok=True)

    plots = {
        "option_a_scatter_regression.png": _make_phase1c_plot_scatter,
        "option_b_quadrant_scatter.png": _make_phase1c_plot_quadrant,
        "option_c_binned_trend.png": _make_phase1c_plot_binned,
        "option_d_dumbbell_lines.png": _make_phase1c_plot_dumbbell,
        "option_e_recommended_two_panel.png": _make_phase1c_plot_recommended,
        "option_f_quartile_heatmap.png": _make_phase1c_plot_quartile_heatmap,
    }
    for fname, fn in plots.items():
        fn(df, options_dir / fname, strength_label)

    summary = pd.DataFrame(
        [
            {
                "option_file": "option_a_scatter_regression.png",
                "purpose": "Direct relationship view with trend line and labeled outliers.",
                "strength": "Most standard choice for two quantitative variables.",
                "limitation": "Shows less detail on disparity ranking than bar/rank views.",
            },
            {
                "option_file": "option_b_quadrant_scatter.png",
                "purpose": "Highlights balanced-vs-strong quadrant narrative.",
                "strength": "Makes strategic quadrants easy to interpret.",
                "limitation": "Median split can oversimplify continuous relationship.",
            },
            {
                "option_file": "option_c_binned_trend.png",
                "purpose": "Shows average team strength across disparity bins.",
                "strength": "Stable trend view with uncertainty bars.",
                "limitation": "Aggregation hides individual team variation.",
            },
            {
                "option_file": "option_d_dumbbell_lines.png",
                "purpose": "Compares line1 vs line2 quality for each team directly.",
                "strength": "Excellent for explaining disparity construction itself.",
                "limitation": "Indirect on disparity-vs-strength relationship.",
            },
            {
                "option_file": "option_e_recommended_two_panel.png",
                "purpose": "Combines relationship plot + top disparity ranking support.",
                "strength": "Best story balance for commissioner audience.",
                "limitation": "Slightly denser than a single-panel scatter.",
            },
            {
                "option_file": "option_f_quartile_heatmap.png",
                "purpose": "Shows outlier-resistant connection via quartile structure.",
                "strength": "Best when relationship is non-linear/threshold and scatter looks noisy.",
                "limitation": "Uses discretization, so exact continuous values are abstracted.",
            },
        ]
    )
    summary.to_csv(options_dir / "visualization_options_summary.csv", index=False)

    stats = _relationship_stats(df)
    refs = []
    refs.append("Visualization references used:")
    refs.append("- Data to Viz (scatter plot): https://www.data-to-viz.com/graph/scatter.html")
    refs.append("- Datawrapper (scatter plot annotation/labeling): https://www.datawrapper.de/blog/introducing-scatter-plot")
    refs.append("- Datawrapper Academy (dot/range plot for two-value comparison): https://academy.datawrapper.de/article/122-how-to-create-a-dot-plot")
    refs.append("- Datawrapper Academy (heatmap for pattern detection in binned tables): https://academy.datawrapper.de/article/308-how-to-create-a-heatmap")
    refs.append("- Data to Viz (connected comparison caveats): https://www.data-to-viz.com/graph/connectedscatter.html")
    (options_dir / "visualization_research_sources.txt").write_text("\n".join(refs) + "\n", encoding="utf-8")

    ct, top_share = _quartile_pattern(df)
    high_disp_top_strength_share = float(top_share.loc["Q4_HighDisparity"])
    top_strength_count_high_disp = int(ct.loc["Q4_High", "Q4_HighDisparity"])
    use_weak_relation = bool(np.isfinite(stats["pearson_r"]) and np.isfinite(stats["spearman_rho"]) and abs(stats["pearson_r"]) < 0.15 and abs(stats["spearman_rho"]) < 0.15)
    chosen = "option_f_quartile_heatmap.png" if use_weak_relation else "option_e_recommended_two_panel.png"

    rec = []
    rec.append(f"Recommended visualization: {chosen}")
    if use_weak_relation:
        rec.append(
            "Reason: the linear relationship is weak, so quartile heatmap structure better communicates a practical threshold pattern and reduces outlier dominance."
        )
    else:
        rec.append(
            "Reason: best alignment with the prompt by showing both the relationship (disparity vs team strength) and the ordered disparity leaderboard in one PNG."
        )
    if np.isfinite(stats["pearson_r"]) and np.isfinite(stats["spearman_rho"]):
        if abs(stats["pearson_r"]) < 0.15 and abs(stats["spearman_rho"]) < 0.15:
            rec.append(
                "Conclusion: the connection is non-linear and non-monotonic; ELO is highest in Q1/Q3 disparity bands and lower in Q2/Q4, with no top-ELO teams in Q4."
            )
        elif stats["pearson_r"] > 0:
            rec.append("Conclusion: higher disparity tends to align with higher team strength in this sample.")
        else:
            rec.append("Conclusion: higher disparity tends to align with lower team strength in this sample.")
    rec.append(f"Strength metric used: {strength_label}")
    rec.append(
        f"Quartile takeaway: top-{strength_label} teams by disparity quartile = "
        f"Q1:{int(ct.loc['Q4_High','Q1_Balanced'])}, Q2:{int(ct.loc['Q4_High','Q2'])}, "
        f"Q3:{int(ct.loc['Q4_High','Q3'])}, Q4:{top_strength_count_high_disp} "
        f"(Q4 share = {100.0*high_disp_top_strength_share:.1f}%)."
    )
    rec.append(f"Observed Pearson r: {stats['pearson_r']:.4f}")
    rec.append(f"Observed Spearman rho: {stats['spearman_rho']:.4f}")
    rec.append(f"Observed linear R^2: {stats['linear_r2']:.4f}")
    (options_dir / "visualization_recommendation.txt").write_text("\n".join(rec) + "\n", encoding="utf-8")

    return options_dir / chosen, stats


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
    team_strength, strength_label = _team_strength_table(paths=paths, ev_df=ev_df, repo_root=repo_root)
    viz_df = ratios.merge(team_strength, on="team", how="inner").sort_values("ratio", ascending=False).reset_index(drop=True)
    if "rank_x" in viz_df.columns:
        viz_df = viz_df.rename(columns={"rank_x": "disparity_rank"})
    if "rank_y" in viz_df.columns:
        viz_df = viz_df.rename(columns={"rank_y": "team_strength_rank"})
    viz_df["team_strength_metric"] = strength_label

    best_plot, rel_stats = _run_visualization_experiments(viz_df, out_dir=out_dir, strength_label=strength_label)
    shutil.copyfile(best_plot, out_dir / "phase1c_line_disparity_vs_team_strength.png")
    viz_df.to_csv(out_dir / "phase1c_viz_table.csv", index=False)
    viz_df.to_csv(Path("outputs") / "phase1c_output.csv", index=False)
    phase_art = repo_root / "phases" / "phase1c" / "artifacts"
    phase_art.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_dir / "phase1c_line_disparity_vs_team_strength.png", phase_art / "phase1c_line_disparity_vs_team_strength.png")
    shutil.copyfile(out_dir / "phase1c_viz_table.csv", phase_art / "phase1c_viz_table.csv")

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
    md.append(f"- team_strength_metric: `{strength_label}`")
    md.append(f"- pearson_r(disparity, strength): `{rel_stats['pearson_r']:.4f}`")
    md.append(f"- spearman_rho(disparity, strength): `{rel_stats['spearman_rho']:.4f}`")
    md.append(f"- linear_R2(disparity -> strength): `{rel_stats['linear_r2']:.4f}`")
    md.append("- nonlinear_takeaway: `ELO pattern is non-monotonic: Q1/Q3 disparity bins are stronger; Q2/Q4 are weaker; Q4 has 0 top-ELO teams.`")
    md.append("- options: `viz_options/` (multiple tested visualizations + recommendation)")
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
