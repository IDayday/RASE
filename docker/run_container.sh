#!/usr/bin/env bash
set -euo pipefail

IMAGE=${IMAGE:-rase-phase0:cu118}
PROJECT_DIR=${PROJECT_DIR:-$(pwd)}
DATA_DIR=${DATA_DIR:-$HOME/.d4rl}

mkdir -p "${DATA_DIR}" "${PROJECT_DIR}/outputs" "${PROJECT_DIR}/logs"

docker run --rm -it \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e MUJOCO_GL=osmesa \
  -e PYOPENGL_PLATFORM=osmesa \
  -e D4RL_SUPPRESS_IMPORT_ERROR=1 \
  -v "${PROJECT_DIR}":/workspace/rase_phase0 \
  -v "${DATA_DIR}":/root/.d4rl \
  -w /workspace/rase_phase0 \
  "${IMAGE}" \
  bash
