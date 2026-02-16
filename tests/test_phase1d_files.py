from __future__ import annotations

from pathlib import Path

from whsdsci.run_phase1d_relevant import main as run_phase1d_relevant_main


def test_phase1d_text_file_exists():
    p = Path("phases/phase1d/offensive_line_quality_disparity.md")
    assert p.exists()
    txt = p.read_text(encoding="utf-8")
    assert "Offensive Line Quality Disparity" in txt
    assert len(txt.strip()) > 100


def test_phase1d_copy_script_writes_output():
    run_phase1d_relevant_main()
    out = Path("outputs/phase1d/phase1d_offensive_line_quality_disparity.md")
    flat = Path("outputs/phase1d_output.md")
    assert out.exists()
    assert flat.exists()
