#!/usr/bin/env bash
set -euo pipefail

# Independent jobs are better than DDP for this offline-RL pre-experiment.
# Override GPUs without editing the file, e.g.:
#   RASE_GPU_IDS="0 1 2 3 4 5 6 7" bash scripts/run_sweep_3090.sh bc
export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}

read -r -a GPU_IDS <<< "${RASE_GPU_IDS:-0 1 2 3}"
ENVS=(
  halfcheetah-medium-replay-v2
  hopper-medium-replay-v2
  walker2d-medium-replay-v2
  antmaze-umaze-v2
)
read -r -a SEEDS <<< "${RASE_SEEDS:-41 42 43}"
SOURCE=${1:-bc}

mkdir -p logs

# Very important: D4RL first-run downloads must be serialized. If several jobs
# download the same .hdf5 simultaneously, h5py may later report "truncated file".
if [[ "${RASE_SKIP_PREFETCH:-0}" != "1" ]]; then
  bash scripts/prefetch_d4rl.sh "${ENVS[@]}"
fi

JOBS=()
for env in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    JOBS+=("${env},${seed}")
  done
done

run_worker() {
  local gpu="$1"
  local worker_idx="$2"
  local stride="$3"
  local j env seed
  for ((j=worker_idx; j<${#JOBS[@]}; j+=stride)); do
    IFS=',' read -r env seed <<< "${JOBS[$j]}"
    echo "[worker ${worker_idx}] env=${env} seed=${seed} gpu=${gpu} source=${SOURCE}"
    CUDA_VISIBLE_DEVICES=${gpu} python run_pipeline.py \
      --config configs/phase0_d4rl.yaml \
      --env_name "${env}" \
      --seed "${seed}" \
      --device cuda:0 \
      --candidate_source "${SOURCE}" \
      > "logs/${env}_seed${seed}_${SOURCE}.log" 2>&1
  done
}

for i in "${!GPU_IDS[@]}"; do
  run_worker "${GPU_IDS[$i]}" "$i" "${#GPU_IDS[@]}" &
done
wait
