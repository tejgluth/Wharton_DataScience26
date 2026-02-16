from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from phases.phase1b.system import BestConfig, resolve_best_config
from whsdsci.build_long import build_canonical_long
from whsdsci.io import discover_paths


LOGGER = logging.getLogger(__name__)


def setup_logger(name: str, log_path: Path) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    sh = logging.StreamHandler()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def get_git_commit(repo_root: Path) -> str | None:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return proc.stdout.strip() or None
    except Exception:
        return None
    return None


def load_ev_dataset(repo_root: Path, outputs_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    paths = discover_paths(repo_root=repo_root, outputs_dir=outputs_dir)
    raw = pd.read_csv(paths["whl_2025"])
    _, ev_df, _ = build_canonical_long(raw_df=raw, outputs_dir=outputs_dir)
    if ev_df.empty:
        raise RuntimeError("EV subset is empty")
    return ev_df, paths


def resolve_phase1b_config(config_name: str | None, outputs_dir: Path) -> tuple[BestConfig, dict[str, Any]]:
    cfg = resolve_best_config(outputs_dir=outputs_dir, config_name=config_name)
    payload = {
        "config_id": cfg.config_id,
        "combiner_family": cfg.combiner_family,
        "base_pool_id": cfg.base_pool_id,
        "base_models": cfg.base_models,
        "calibration_type": cfg.calibration_type,
        "hyperparams": cfg.hyperparams,
        **cfg.metadata,
    }
    return cfg, payload


def write_simple_yaml(path: Path, payload: dict[str, Any]) -> None:
    def _fmt(v: Any) -> str:
        if isinstance(v, bool):
            return "true" if v else "false"
        if v is None:
            return "null"
        if isinstance(v, (int, float)):
            return str(v)
        return json.dumps(v)

    lines: list[str] = []
    for k, v in payload.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for kk, vv in v.items():
                lines.append(f"  {kk}: {_fmt(vv)}")
        elif isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {_fmt(item)}")
        else:
            lines.append(f"{k}: {_fmt(v)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
