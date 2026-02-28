from __future__ import annotations

import shutil
from pathlib import Path


def main() -> None:
    src = Path("phases/phase1d/phase1d_full_response.txt")
    if not src.exists():
        raise FileNotFoundError(f"Missing Phase 1d text file: {src}")
    out_dir = Path("outputs/phase1d")
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "phase1d_full_response.txt"
    shutil.copyfile(src, dst)
    flat_dst = Path("outputs/phase1d_output.txt")
    shutil.copyfile(src, flat_dst)
    print(f"Phase 1d narrative copied to {dst}")
    print(f"Phase 1d narrative copied to {flat_dst}")


if __name__ == "__main__":
    main()
