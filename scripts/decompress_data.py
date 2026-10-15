"""Run after the pinned .zst download completes; retain the source archive."""
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
source = root / "data/pusht_expert_train.h5.zst"
dest = root / "data/datasets/pusht_expert_train.h5"
dest.parent.mkdir(parents=True, exist_ok=True)
if dest.exists():
    raise SystemExit(f"Already exists: {dest}")
temporary = dest.with_suffix(".h5.partial")
with temporary.open("wb") as handle:
    subprocess.run(["zstd", "-dc", str(source)], stdout=handle, check=True)
os.replace(temporary, dest)
print(dest, dest.stat().st_size, flush=True)
