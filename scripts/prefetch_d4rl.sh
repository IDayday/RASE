#!/usr/bin/env bash
set -euo pipefail

# Download and validate D4RL datasets sequentially before multi-GPU training.
# This prevents concurrent first-run downloads from producing truncated HDF5s.
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}

DEFAULT_ENVS=(
  halfcheetah-medium-replay-v2
  hopper-medium-replay-v2
  walker2d-medium-replay-v2
  antmaze-umaze-v2
)

if [[ $# -gt 0 ]]; then
  ENVS=("$@")
else
  ENVS=("${DEFAULT_ENVS[@]}")
fi

bash scripts/repair_d4rl_cache.sh

for env_name in "${ENVS[@]}"; do
  echo "[prefetch] ${env_name}"
  python - "${env_name}" <<'PY'
from __future__ import annotations

import os
import sys
from pathlib import Path

import gym
import d4rl  # noqa: F401
import h5py

os.environ.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")

env_name = sys.argv[1]

def cache_roots():
    roots = []
    for key in ["D4RL_DATASET_DIR", "D4RL_DATASETDIR"]:
        if os.environ.get(key):
            roots.append(Path(os.environ[key]).expanduser())
    roots.extend([Path.home() / ".d4rl" / "datasets", Path.home() / ".d4rl"])
    seen = set()
    for root in roots:
        root = root.resolve()
        if root.exists() and root not in seen:
            seen.add(root)
            yield root

def matching_hdf5():
    env_token = env_name.lower().replace("_", "-")
    base_token = env_token.rsplit("-v", 1)[0]
    out = []
    for root in cache_roots():
        for path in root.rglob("*.hdf5"):
            name = path.name.lower().replace("_", "-")
            if env_token in name or base_token in name:
                out.append(path)
    return sorted(set(out))

def delete_matches(reason: str):
    for path in matching_hdf5():
        print(f"[prefetch] deleting {path} ({reason})")
        path.unlink(missing_ok=True)

try:
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)
except OSError as exc:
    # The common failure is: truncated file: eof=..., stored_eof=...
    delete_matches(str(exc))
    env = gym.make(env_name)
    dataset = d4rl.qlearning_dataset(env)

print("[prefetch] loaded keys:", {k: tuple(v.shape) for k, v in dataset.items() if hasattr(v, "shape")})
for path in matching_hdf5():
    with h5py.File(path, "r") as f:
        print(f"[prefetch] validated {path} keys={list(f.keys())[:8]}")
try:
    env.close()
except Exception:
    pass
PY
done
