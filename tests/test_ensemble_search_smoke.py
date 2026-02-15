from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from whsdsci.ensemble.search import run_ensemble_search


def _tiny_ev_df(n_games: int = 30, segments_per_game: int = 8, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    teams = [f"T{i}" for i in range(1, 11)]
    rows = []
    for g in range(1, n_games + 1):
        for s in range(segments_per_game):
            o = teams[(g + s) % len(teams)]
            d = teams[(g + s + 3) % len(teams)]
            off = "first_off" if s % 2 == 0 else "second_off"
            deff = "first_def" if (s // 2) % 2 == 0 else "second_def"
            toi_hr = rng.uniform(0.01, 0.08)
            base = 0.8 if off == "first_off" else 0.62
            rate = np.clip(base + rng.normal(0, 0.09), 0.01, None)
            xg = rate * toi_hr
            shots = rng.poisson(max(0.1, xg * 7.0))
            rows.append(
                {
                    "game_id": f"game_{g}",
                    "offense_team": o,
                    "defense_team": d,
                    "off_line": off,
                    "def_pair": deff,
                    "off_unit": f"{o}__{off}",
                    "def_unit": f"{d}__{deff}",
                    "is_home": s % 2,
                    "toi_hr": toi_hr,
                    "xg_for": xg,
                    "shots_for": shots,
                }
            )
    return pd.DataFrame(rows)


def test_ensemble_search_smoke(tmp_path: Path):
    ev_df = _tiny_ev_df()

    out = run_ensemble_search(
        ev_df=ev_df,
        outputs_dir=tmp_path,
        random_state=0,
        screen_target=10,
        full_target=5,
        deep_target=2,
        base_model_limit=4,
        logger=None,
    )

    assert "best_config" in out
    assert "top20" in out
    assert "top10" in out

    # output files exist
    for p in [
        tmp_path / "ensemble_search_results.csv",
        tmp_path / "ensemble_best_config.json",
        tmp_path / "final_top10.csv",
        tmp_path / "submission_phase1b.csv",
        tmp_path / "oof_predictions_ev.parquet",
    ]:
        assert p.exists(), str(p)

    best = json.loads((tmp_path / "ensemble_best_config.json").read_text())
    assert "combiner_family" in best
    assert best["mean_cv_poisson_deviance"] > 0

    sub = pd.read_csv(tmp_path / "submission_phase1b.csv")
    assert not sub.empty
    assert np.all(np.isfinite(sub["ratio"]))
    assert np.all(sub["ratio"] > 0)
