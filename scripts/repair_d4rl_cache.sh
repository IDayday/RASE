#!/usr/bin/env bash
set -euo pipefail

# Delete corrupted D4RL HDF5 files from the local cache. Safe to run before
# prefetching. It only removes files that h5py cannot open.
export D4RL_SUPPRESS_IMPORT_ERROR=1

python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path
import h5py

roots = []
for key in ["D4RL_DATASET_DIR", "D4RL_DATASETDIR"]:
    if os.environ.get(key):
        roots.append(Path(os.environ[key]).expanduser())
roots.extend([Path.home() / ".d4rl" / "datasets", Path.home() / ".d4rl"])

seen = set()
checked = deleted = 0
for root in roots:
    root = root.resolve()
    if root in seen or not root.exists():
        continue
    seen.add(root)
    for path in root.rglob("*.hdf5"):
        checked += 1
        try:
            with h5py.File(path, "r") as f:
                _ = list(f.keys())
        except Exception as exc:
            print(f"[repair] deleting corrupt HDF5: {path} :: {exc}")
            path.unlink(missing_ok=True)
            deleted += 1
print(f"[repair] checked={checked}, deleted={deleted}")
PY
