from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from phases.phase1c.run import run_phase1c
from whsdsci.best_config import BestConfig


def _tiny_ev_df(n_games: int = 12, segments_per_game: int = 8, random_state: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    teams = [f"T{i}" for i in range(1, 9)]
    rows = []
    for g in range(1, n_games + 1):
        for s in range(segments_per_game):
            o = teams[(g + s) % len(teams)]
            d = teams[(g + s + 2) % len(teams)]
            off = "first_off" if s % 2 == 0 else "second_off"
            deff = "first_def" if (s // 2) % 2 == 0 else "second_def"
            toi_hr = rng.uniform(0.015, 0.08)
            rate = np.clip((0.9 if off == "first_off" else 0.7) + rng.normal(0, 0.08), 0.01, None)
            xg = rate * toi_hr
            xga = np.clip(xg * rng.uniform(0.6, 1.3), 0.001, None)
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
                    "log_toi_hr": np.log(max(toi_hr, 1e-9)),
                    "xg_for": xg,
                    "xg_diff": xg - xga,
                    "shots_for": max(0, int(rng.poisson(max(0.1, xg * 7.0)))),
                }
            )
    return pd.DataFrame(rows)


def test_phase1c_small_mode_smoke(tmp_path: Path, monkeypatch):
    ev_df = _tiny_ev_df()
    cfg = BestConfig(
        config_id="cfg_test_phase1c",
        combiner_family="tree_poisson",
        base_pool_id="tiny_pool",
        base_models=["POISSON_GLM_OFFSET", "DEFENSE_ADJ_TWO_STEP"],
        calibration_type="none",
        hyperparams={"backend": "auto", "max_depth": 2, "learning_rate": 0.1, "n_estimators": 60},
        metadata={},
    )

    def _fake_load_ev_dataset(repo_root: Path, outputs_dir: Path):
        return ev_df.copy(), {"league_table": None}

    def _fake_resolve_phase1b_config(config_name: str | None, outputs_dir: Path):
        return cfg, {
            "config_id": cfg.config_id,
            "combiner_family": cfg.combiner_family,
            "base_pool_id": cfg.base_pool_id,
            "base_models": cfg.base_models,
            "calibration_type": cfg.calibration_type,
            "hyperparams": cfg.hyperparams,
        }

    monkeypatch.setattr("phases.phase1c.run.load_ev_dataset", _fake_load_ev_dataset)
    monkeypatch.setattr("phases.phase1c.run.resolve_phase1b_config", _fake_resolve_phase1b_config)

    out_dir = tmp_path / "phase1c"
    result = run_phase1c(
        config_name="cfg_test_phase1c",
        seed=1,
        out_dir=out_dir,
        small=True,
    )

    assert result["config_id"] == "cfg_test_phase1c"
    assert result["rows_ev"] > 0

    required = [
        "phase1c_summary.md",
        "best_config.json",
        "best_config.yaml",
        "phase1c_line_disparity_vs_team_strength.png",
        "phase1c_viz_table.csv",
        "run.log",
    ]
    for rel in required:
        assert (out_dir / rel).exists(), rel
