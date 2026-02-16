from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class BestConfig:
    config_id: str
    combiner_family: str
    base_pool_id: str
    base_models: list[str]
    calibration_type: str
    hyperparams: dict[str, Any]
    metadata: dict[str, Any]


def _coerce_payload(payload: dict[str, Any]) -> BestConfig:
    return BestConfig(
        config_id=str(payload["config_id"]),
        combiner_family=str(payload.get("combiner_family", "tree_poisson")),
        base_pool_id=str(payload.get("base_pool_id", "unknown_pool")),
        base_models=[str(x) for x in payload.get("base_models", [])],
        calibration_type=str(payload.get("calibration_type", "none")),
        hyperparams=dict(payload.get("hyperparams", {})),
        metadata={k: v for k, v in payload.items() if k not in {"config_id", "combiner_family", "base_pool_id", "base_models", "calibration_type", "hyperparams"}},
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_best_config(outputs_dir: Path, config_name: str | None = None) -> BestConfig:
    outputs_dir = Path(outputs_dir)
    if config_name and config_name.endswith(".json"):
        return _coerce_payload(_read_json(Path(config_name)))

    candidate_files = [
        outputs_dir / "confirmed_best_config.json",
        outputs_dir / "ensemble_best_config.json",
    ]
    payloads = [(_read_json(p), p) for p in candidate_files if p.exists()]

    if config_name:
        for payload, _ in payloads:
            if str(payload.get("config_id")) == str(config_name):
                return _coerce_payload(payload)

        search_path = outputs_dir / "ensemble_search_results.csv"
        if search_path.exists():
            df = pd.read_csv(search_path)
            sub = df[
                (df["status"] == "OK")
                & (df["config_id"].astype(str) == str(config_name))
                & (df["mean_cv_poisson_deviance"].notna())
            ].copy()
            if not sub.empty:
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
                    "source": "ensemble_search_results.csv",
                    "stage": str(row.get("stage", "")),
                    "mean_cv_poisson_deviance": float(row["mean_cv_poisson_deviance"]),
                }
                return _coerce_payload(payload)

        if re.match(r"^cfg_[A-Za-z0-9_\\.-]+$", str(config_name)):
            raise FileNotFoundError(f"Could not resolve best config id: {config_name}")

    if payloads:
        return _coerce_payload(payloads[0][0])

    best_txt = outputs_dir / "best_method.txt"
    if best_txt.exists():
        m = re.search(r"config_id=(cfg_[A-Za-z0-9_\\.-]+)", best_txt.read_text(encoding="utf-8"))
        if m:
            return resolve_best_config(outputs_dir=outputs_dir, config_name=m.group(1))
    raise FileNotFoundError("No best config artifact found in outputs/")

