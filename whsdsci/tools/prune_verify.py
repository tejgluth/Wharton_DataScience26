from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out


def main() -> None:
    checks: list[tuple[str, list[str]]] = [
        ("tests", [sys.executable, "-m", "pytest", "-q"]),
        ("phase1b_best", [sys.executable, "-m", "whsdsci.run_phase1b_best"]),
        ("phase1c", [sys.executable, "-m", "whsdsci.run_phase1c"]),
        ("phase1d_relevant", [sys.executable, "-m", "whsdsci.run_phase1d_relevant"]),
    ]

    failures = []
    for name, cmd in checks:
        code, out = _run(cmd)
        print(f"== {name} ==")
        print(out.strip()[-2000:])
        if code != 0:
            failures.append(name)

    required = [
        Path("outputs/submission_phase1b.csv"),
        Path("outputs/phase1c_output.csv"),
        Path("outputs/phase1d_output.csv"),
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        failures.append(f"missing_outputs:{','.join(missing)}")

    if failures:
        print("PRUNE_VERIFY FAIL")
        print("Failed checks:", failures)
        raise SystemExit(1)
    print("PRUNE_VERIFY PASS")


if __name__ == "__main__":
    main()

