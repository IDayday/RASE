#!/usr/bin/env bash
set -euo pipefail

# Independent jobs are better than DDP for this offline-RL pre-experiment.
# Edit GPU_IDS to match visible devices in `nvidia-smi`.
GPU_IDS=(0 1 2 3)
ENVS=(
  halfcheetah-medium-replay-v2
  hopper-medium-replay-v2
  walker2d-medium-replay-v2
  antmaze-umaze-v2
)
SEEDS=(0 1 2)
SOURCE=${1:-bc}

job_id=0
for env in "${ENVS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    gpu=${GPU_IDS[$((job_id % ${#GPU_IDS[@]}))]}
    echo "Launching env=${env} seed=${seed} gpu=${gpu} source=${SOURCE}"
    CUDA_VISIBLE_DEVICES=${gpu} python run_pipeline.py \
      --config configs/phase0_d4rl.yaml \
      --env_name "${env}" \
      --seed "${seed}" \
      --device cuda:0 \
      --candidate_source "${SOURCE}" \
      > "logs/${env}_seed${seed}_${SOURCE}.log" 2>&1 &
    job_id=$((job_id + 1))
    if (( job_id % ${#GPU_IDS[@]} == 0 )); then
      wait
    fi
  done
done
wait
