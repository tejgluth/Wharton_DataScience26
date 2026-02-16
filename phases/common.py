from __future__ import annotations

import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from whsdsci.build_long import build_canonical_long
from whsdsci.ensemble.search import Config
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


def _row_to_config(row: pd.Series) -> Config:
    return Config(
        config_id=str(row["config_id"]),
        combiner_family=str(row["combiner_family"]),
        base_pool_id=str(row["base_pool_id"]),
        base_models=list(json.loads(row["base_models"])),
        calibration_type=str(row["calibration_type"]),
        hyperparams=dict(json.loads(row["hyperparams"])),
    )


def _from_payload(payload: dict[str, Any]) -> Config:
    return Config(
        config_id=str(payload["config_id"]),
        combiner_family=str(payload["combiner_family"]),
        base_pool_id=str(payload["base_pool_id"]),
        base_models=list(payload["base_models"]),
        calibration_type=str(payload["calibration_type"]),
        hyperparams=dict(payload["hyperparams"]),
    )


def resolve_phase1b_config(config_name: str | None, outputs_dir: Path) -> tuple[Config, dict[str, Any]]:
    """Resolve a phase1b config from artifacts, defaulting to confirmed best."""
    candidate_payload_files = [
        outputs_dir / "confirmed_best_config.json",
        outputs_dir / "ensemble_best_config.json",
    ]
    if config_name and config_name.endswith(".json"):
        payload = json.loads(Path(config_name).read_text(encoding="utf-8"))
        cfg = _from_payload(payload)
        return cfg, payload

    payloads: list[dict[str, Any]] = []
    for p in candidate_payload_files:
        if p.exists():
            payloads.append(json.loads(p.read_text(encoding="utf-8")))

    if config_name:
        for payload in payloads:
            if str(payload.get("config_id")) == str(config_name):
                return _from_payload(payload), payload

    # Find by config id in search results.
    if config_name:
        search_path = outputs_dir / "ensemble_search_results.csv"
        if search_path.exists():
            res = pd.read_csv(search_path)
            sub = res[
                (res["status"] == "OK")
                & (res["config_id"].astype(str) == str(config_name))
                & (res["mean_cv_poisson_deviance"].notna())
            ].copy()
            if not sub.empty:
                # prioritize full/deep over screen when present
                stage_order = {"deep": 0, "full": 1, "screen": 2}
                sub["stage_order"] = sub["stage"].astype(str).map(stage_order).fillna(9)
                row = sub.sort_values(["stage_order", "mean_cv_poisson_deviance"]).iloc[0]
                payload = {
                    "config_id": str(row["config_id"]),
                    "combiner_family": str(row["combiner_family"]),
                    "base_pool_id": str(row["base_pool_id"]),
                    "base_models": list(json.loads(row["base_models"])),
                    "calibration_type": str(row["calibration_type"]),
                    "hyperparams": dict(json.loads(row["hyperparams"])),
                    "mean_cv_poisson_deviance": float(row["mean_cv_poisson_deviance"]),
                    "std_cv_poisson_deviance": float(row["std_cv_poisson_deviance"]) if pd.notna(row["std_cv_poisson_deviance"]) else None,
                    "source": "ensemble_search_results.csv",
                    "stage": str(row["stage"]),
                }
                return _row_to_config(row), payload

    # If explicitly asked for cfg_xxx and not found, error.
    if config_name and re.match(r"^cfg_[A-Za-z0-9_\\.-]+$", str(config_name)):
        raise FileNotFoundError(f"Could not resolve config id: {config_name}")

    # Default best.
    if payloads:
        payload = payloads[0]
        cfg = _from_payload(payload)
        return cfg, payload

    # Fallback from best_method text.
    best_txt = outputs_dir / "best_method.txt"
    if best_txt.exists():
        txt = best_txt.read_text(encoding="utf-8")
        m = re.search(r"config_id=(cfg_[A-Za-z0-9_\\.-]+)", txt)
        if m:
            return resolve_phase1b_config(m.group(1), outputs_dir)
    raise FileNotFoundError("Could not resolve phase1b best config from outputs artifacts")


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

