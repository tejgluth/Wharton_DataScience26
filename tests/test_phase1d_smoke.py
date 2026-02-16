from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.olqd_report import run_olqd_report
from whsdsci.best_config import BestConfig


def _tiny_ev_df(n_games: int = 10, segments_per_game: int = 6, random_state: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    teams = [f"T{i}" for i in range(1, 9)]
    rows = []
    for g in range(1, n_games + 1):
        for s in range(segments_per_game):
            o = teams[(g + s) % len(teams)]
            d = teams[(g + s + 3) % len(teams)]
            off = "first_off" if s % 2 == 0 else "second_off"
            deff = "first_def" if (s // 2) % 2 == 0 else "second_def"
            toi = rng.uniform(0.02, 0.08)
            xg = np.clip((0.9 if off == "first_off" else 0.65) * toi + rng.normal(0, 0.01), 0.001, None)
            xga = np.clip(xg * rng.uniform(0.6, 1.4), 0.001, None)
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
                    "toi_hr": toi,
                    "log_toi_hr": np.log(max(toi, 1e-9)),
                    "xg_for": xg,
                    "xg_diff": xg - xga,
                    "shots_for": max(0, int(rng.poisson(max(1.0, xg * 9)))),
                }
            )
    return pd.DataFrame(rows)


def test_phase1d_smoke(tmp_path: Path, monkeypatch):
    ev_df = _tiny_ev_df()
    cfg = BestConfig(
        config_id="cfg_test_p1d",
        combiner_family="tree_poisson",
        base_pool_id="tiny",
        base_models=["POISSON_GLM_OFFSET", "DEFENSE_ADJ_TWO_STEP"],
        calibration_type="none",
        hyperparams={"backend": "auto", "max_depth": 2, "learning_rate": 0.1, "n_estimators": 50},
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

    monkeypatch.setattr("analysis.olqd_report.load_ev_dataset", _fake_load_ev_dataset)
    monkeypatch.setattr("analysis.olqd_report.resolve_phase1b_config", _fake_resolve_phase1b_config)

    out_dir = tmp_path / "reports"
    result = run_olqd_report(config_name="cfg_test_p1d", out_dir=out_dir, seed=1, small=True)
    assert result["config_id"] == "cfg_test_p1d"
    assert (out_dir / "phase1d_offensive_line_quality_disparity.md").exists()
    assert (out_dir / "olqd_ablation_summary.csv").exists()
    assert (out_dir / "olqd_team_table.csv").exists()
    assert (out_dir / "figures" / "olqd_ratio_vs_team_strength.png").exists()
    assert (out_dir / "figures" / "olqd_ablation_by_fold.png").exists()
    payload = json.loads((out_dir / "phase1d_olqd_result.json").read_text(encoding="utf-8"))
    assert "mean_delta_poisson_deviance" in payload

