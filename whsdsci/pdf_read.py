from __future__ import annotations

import logging
from pathlib import Path
from typing import Any


LOGGER = logging.getLogger(__name__)
KEYWORDS = [
    "Phase 1b",
    "expected goals",
    "ratio",
    "disparity",
    "TOI",
    "offset",
    "line 1",
    "line 2",
]


def _extract_with_pypdf(pdf_path: Path, max_pages: int = 2) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    out = []
    for i, page in enumerate(reader.pages[:max_pages]):
        text = page.extract_text() or ""
        out.append(f"[Page {i + 1}]\n{text[:3500]}")
    return "\n\n".join(out)


def write_pdf_notes(paths: dict[str, Any], out_path: Path) -> None:
    pdf_paths = []
    for key in ("guideline_pdf", "research_pdf"):
        p = paths.get(key)
        if p:
            pdf_paths.append(Path(p))

    lines: list[str] = []
    lines.append("PDF Notes (best effort)\n")

    if not pdf_paths:
        lines.append("No target PDFs discovered.\n")
        out_path.write_text("\n".join(lines), encoding="utf-8")
        return

    for pdf in pdf_paths:
        lines.append(f"=== {pdf} ===")
        if not pdf.exists():
            lines.append("Status: missing")
            lines.append("")
            continue

        try:
            extracted = _extract_with_pypdf(pdf)
            status = "parsed"
        except ModuleNotFoundError:
            extracted = ""
            status = "pypdf not installed"
        except Exception as exc:
            extracted = ""
            status = f"parse failed: {exc}"
            LOGGER.warning("Failed to parse PDF %s: %s", pdf, exc)

        lines.append(f"Status: {status}")
        if extracted:
            lowered = extracted.lower()
            hits = {kw: lowered.count(kw.lower()) for kw in KEYWORDS}
            lines.append("Keyword hits:")
            lines.extend([f"- {k}: {v}" for k, v in hits.items()])
            lines.append("Extract (first ~2 pages, truncated):")
            lines.append(extracted)
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
