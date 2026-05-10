#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:-bc}
GPUS=(${RASE_GPU_IDS:-0 1 2 3})
ENVS=(
  halfcheetah-medium-replay-v2
  hopper-medium-replay-v2
  walker2d-medium-replay-v2
)
SEEDS=(${RASE_SEEDS:-41 42 43})
mkdir -p logs

job_id=0
for env in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    gpu=${GPUS[$((job_id % ${#GPUS[@]}))]}
    echo "[rollout] env=${env} seed=${seed} source=${SOURCE} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES=${gpu} python run_rollout_diagnostic.py \
      --config configs/phase0_d4rl.yaml \
      --env_name "${env}" \
      --seed "${seed}" \
      --device cuda:0 \
      --candidate_source "${SOURCE}" \
      > "logs/rollout_${SOURCE}_${env}_seed${seed}.log" 2>&1 &
    job_id=$((job_id + 1))
    if (( job_id % ${#GPUS[@]} == 0 )); then
      wait
    fi
  done
done
wait
