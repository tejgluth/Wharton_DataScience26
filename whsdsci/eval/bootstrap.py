from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from whsdsci.strength import compute_disparity_ratios, compute_standardized_strengths


LOGGER = logging.getLogger(__name__)


@dataclass
class BootstrapResult:
    team_stats: pd.DataFrame
    stability_score: float
    n_boot: int


def bootstrap_rank_stability(
    model_factory,
    ev_df: pd.DataFrame,
    full_top10_teams: list[str],
    n_boot: int = 300,
    random_state: int = 0,
    show_progress: bool = False,
) -> BootstrapResult:
    rng = np.random.default_rng(random_state)
    games = ev_df["game_id"].astype(str).unique()
    if len(games) < 2:
        team_stats = pd.DataFrame(columns=["team", "rank_std", "top10_rate"])
        return BootstrapResult(team_stats=team_stats, stability_score=float("nan"), n_boot=0)

    idx_by_game = {
        g: ev_df.index[ev_df["game_id"].astype(str) == g].to_numpy(dtype=int) for g in games
    }

    team_ranks: dict[str, list[float]] = {}
    team_top10: dict[str, int] = {}

    iterator = range(n_boot)
    if show_progress:
        iterator = tqdm(iterator, desc="bootstrap", leave=False)

    for _ in iterator:
        sampled_games = rng.choice(games, size=len(games), replace=True)
        sampled_idx = np.concatenate([idx_by_game[g] for g in sampled_games])
        boot_df = ev_df.loc[sampled_idx].reset_index(drop=True)

        model = model_factory()
        try:
            model.fit(boot_df)
            strengths = compute_standardized_strengths(model=model, train_ev_df=boot_df)
            ratios = compute_disparity_ratios(strengths)
            ratios = ratios.sort_values("ratio", ascending=False).reset_index(drop=True)
            ratios["rank"] = np.arange(1, len(ratios) + 1)
        except Exception as exc:
            LOGGER.warning("Bootstrap replicate failed: %s", exc)
            continue

        top10 = set(ratios.head(10)["team"].astype(str))
        for _, row in ratios.iterrows():
            team = str(row["team"])
            team_ranks.setdefault(team, []).append(float(row["rank"]))
            if team in top10:
                team_top10[team] = team_top10.get(team, 0) + 1

    rows = []
    actual_boot = max([len(v) for v in team_ranks.values()], default=0)
    for team, ranks in sorted(team_ranks.items()):
        if not ranks:
            continue
        rstd = float(np.std(ranks))
        top10_rate = float(team_top10.get(team, 0) / max(1, len(ranks)))
        rows.append({"team": team, "rank_std": rstd, "top10_rate": top10_rate})

    team_stats = pd.DataFrame(rows)
    if team_stats.empty:
        stability_score = float("nan")
    else:
        mask = team_stats["team"].isin(full_top10_teams)
        stability_score = float(team_stats.loc[mask, "rank_std"].mean()) if mask.any() else float("nan")

    return BootstrapResult(team_stats=team_stats, stability_score=stability_score, n_boot=actual_boot)
