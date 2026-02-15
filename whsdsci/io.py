from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)


def _resolve_optional(path: Path | None) -> str | None:
    if path is None:
        return None
    return str(path.resolve())


def _zip_contains_member(zip_path: Path, member_name: str) -> str | None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            if name.endswith(member_name):
                return name
    return None


def discover_paths(repo_root: Path, outputs_dir: Path) -> dict[str, Any]:
    """Discover required and optional project inputs, including official zip content."""
    data_dir = repo_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    zip_candidates: list[tuple[Path, str]] = []
    for zp in sorted(data_dir.glob("*.zip")):
        try:
            member = _zip_contains_member(zp, "whl_2025.csv")
            if member is not None:
                zip_candidates.append((zp, member))
        except zipfile.BadZipFile:
            LOGGER.warning("Skipping unreadable zip: %s", zp)

    official_zip: Path | None = None
    official_member: str | None = None
    unzipped_whl: Path | None = None
    cache_dir = repo_root / ".cache" / "unzipped"
    cache_dir.mkdir(parents=True, exist_ok=True)

    if zip_candidates:
        official_zip, official_member = zip_candidates[0]
        target_dir = cache_dir / official_zip.stem
        target_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(official_zip, "r") as zf:
            zf.extractall(target_dir)
        unzipped_whl = target_dir / official_member
        LOGGER.info("Using official zip source: %s :: %s", official_zip, official_member)

    whl_external = data_dir / "whl_2025.csv"
    whl_path = unzipped_whl if (unzipped_whl is not None and unzipped_whl.exists()) else None
    if whl_path is None and whl_external.exists():
        whl_path = whl_external
    if whl_path is None:
        raise FileNotFoundError("Could not locate whl_2025.csv in official zip or data directory")

    whl_game_summary = data_dir / "whl_game_summary.csv"
    league_table = data_dir / "league_table.csv"

    pdf_candidates = list(data_dir.glob("*.pdf")) + list(repo_root.glob("*.pdf"))
    pdf_candidates = sorted({p.resolve() for p in pdf_candidates})

    guideline_pdf: Path | None = None
    research_pdf: Path | None = None
    for pdf in pdf_candidates:
        name = pdf.name.lower()
        if guideline_pdf is None and ("workbook" in name or "guideline" in name or "glossary" in name):
            guideline_pdf = pdf
        if research_pdf is None and ("estimating offensive line strength disparity" in name or "model selection and approach" in name):
            research_pdf = pdf

    paths = {
        "repo_root": str(repo_root.resolve()),
        "data_dir": str(data_dir.resolve()),
        "official_zip": _resolve_optional(official_zip),
        "official_zip_member_whl_2025": official_member,
        "unzipped_whl_2025": _resolve_optional(unzipped_whl),
        "whl_2025": str(whl_path.resolve()),
        "whl_game_summary": _resolve_optional(whl_game_summary if whl_game_summary.exists() else None),
        "league_table": _resolve_optional(league_table if league_table.exists() else None),
        "guideline_pdf": _resolve_optional(guideline_pdf),
        "research_pdf": _resolve_optional(research_pdf),
        "all_pdfs": [str(p) for p in pdf_candidates],
    }

    outputs_dir.mkdir(parents=True, exist_ok=True)
    with (outputs_dir / "paths.json").open("w", encoding="utf-8") as f:
        json.dump(paths, f, indent=2)

    return paths
