from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from phases.phase1b.run import run_phase1b_best


def _tiny_raw_df(n_games: int = 12, segments_per_game: int = 8, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    teams = ["A", "B", "C", "D", "E", "F"]
    rows = []
    rid = 0
    for g in range(1, n_games + 1):
        for s in range(segments_per_game):
            rid += 1
            home = teams[(g + s) % len(teams)]
            away = teams[(g + s + 2) % len(teams)]
            off_h = "first_off" if s % 2 == 0 else "second_off"
            off_a = "second_off" if s % 2 == 0 else "first_off"
            def_h = "first_def" if (s // 2) % 2 == 0 else "second_def"
            def_a = "second_def" if (s // 2) % 2 == 0 else "first_def"
            toi = rng.uniform(30.0, 220.0)
            hxg = np.clip((0.8 if off_h == "first_off" else 0.65) * (toi / 3600.0) + rng.normal(0, 0.01), 0.001, None)
            axg = np.clip((0.82 if off_a == "first_off" else 0.62) * (toi / 3600.0) + rng.normal(0, 0.01), 0.001, None)
            rows.append(
                {
                    "game_id": f"game_{g}",
                    "record_id": f"r_{rid}",
                    "home_team": home,
                    "away_team": away,
                    "home_off_line": off_h,
                    "away_off_line": off_a,
                    "home_def_pairing": def_h,
                    "away_def_pairing": def_a,
                    "toi": toi,
                    "home_xg": hxg,
                    "away_xg": axg,
                    "home_shots": max(0, int(rng.poisson(max(1, hxg * 10)))),
                    "away_shots": max(0, int(rng.poisson(max(1, axg * 10)))),
                    "home_goals": max(0, int(rng.poisson(max(0.1, hxg)))),
                    "away_goals": max(0, int(rng.poisson(max(0.1, axg)))),
                }
            )
    return pd.DataFrame(rows)


def test_phase1b_best_smoke(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_path = data_dir / "whl_2025.csv"
    _tiny_raw_df().to_csv(raw_path, index=False)

    cfg_payload = {
        "config_id": "cfg_test",
        "combiner_family": "tree_poisson",
        "base_pool_id": "tiny",
        "base_models": ["POISSON_GLM_OFFSET", "DEFENSE_ADJ_TWO_STEP"],
        "calibration_type": "none",
        "hyperparams": {"backend": "auto", "max_depth": 2, "learning_rate": 0.1, "n_estimators": 50},
    }
    outputs = tmp_path / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "confirmed_best_config.json").write_text(json.dumps(cfg_payload), encoding="utf-8")

    def _fake_discover_paths(repo_root: Path, outputs_dir: Path):
        return {
            "whl_2025": str(raw_path),
            "repo_root": str(tmp_path),
            "data_dir": str(data_dir),
            "official_zip": None,
            "official_zip_member_whl_2025": None,
        }

    monkeypatch.setattr("phases.phase1b.run.discover_paths", _fake_discover_paths)
    top10 = run_phase1b_best(repo_root=tmp_path, outputs_dir=outputs)

    assert not top10.empty
    assert (outputs / "submission_phase1b.csv").exists()
    assert (outputs / "final_top10.csv").exists()
    assert (outputs / "best_method.txt").exists()
    assert (outputs / "phase1b_run.log").exists()
    assert np.all(np.isfinite(top10["ratio"]))
    assert np.all(top10["ratio"] > 0)
    assert float(np.std(np.log(np.clip(top10["ratio"].to_numpy(dtype=float), 1e-9, None)))) > 0
