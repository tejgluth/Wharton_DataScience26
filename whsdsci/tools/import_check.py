from __future__ import annotations

import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.olqd_report import run_olqd_report
from phases.phase1c.run import run_phase1c
from whsdsci.best_config import resolve_best_config
from whsdsci.build_long import build_canonical_long
from whsdsci.io import discover_paths
from whsdsci.models.tree_poisson_best import TreePoissonBestModel


def _tiny_ev(n_games: int) -> pd.DataFrame:
    paths = discover_paths(repo_root=Path.cwd(), outputs_dir=Path("outputs"))
    raw = pd.read_csv(paths["whl_2025"])
    _, ev_df, _ = build_canonical_long(raw_df=raw, outputs_dir=Path("outputs"))
    games = sorted(ev_df["game_id"].astype(str).unique())[:n_games]
    tiny = ev_df[ev_df["game_id"].astype(str).isin(games)].reset_index(drop=True)
    if tiny.empty:
        raise RuntimeError("Tiny EV subset is empty")
    return tiny


def main() -> None:
    try:
        cfg = resolve_best_config(outputs_dir=Path("outputs"))
        tiny = None
        model = None
        fit_error: Exception | None = None
        for n_games in [20, 40, 80, 160]:
            try:
                tiny = _tiny_ev(n_games=n_games)
                model = TreePoissonBestModel(config=cfg, outputs_dir=Path("outputs"), random_state=1).fit(tiny)
                fit_error = None
                break
            except Exception as exc:
                fit_error = exc
                continue
        if model is None or tiny is None:
            raise RuntimeError(f"Could not complete tiny smoke fit for TreePoissonBestModel: {fit_error}")
        pred = model.predict_total(tiny)
        if not (np.isfinite(pred).all() and (pred > 0).all()):
            raise RuntimeError("Invalid predictions from TreePoissonBestModel smoke fit")

        run_phase1c(config_name=cfg.config_id, seed=1, out_dir=Path("outputs/_importcheck_phase1c"), small=True, features="baseline")
        run_olqd_report(config_name=cfg.config_id, out_dir=Path("outputs/_importcheck_phase1d"), seed=1, small=True)
        print("IMPORT_CHECK PASS")
    except Exception as exc:
        print("IMPORT_CHECK FAIL")
        print(str(exc))
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
