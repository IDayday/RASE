#!/usr/bin/env bash
set -euo pipefail

SOURCE=${1:-bc}
OUT_DIR=${RASE_OUT_DIR:-}
POLICY_SQUASH=${RASE_POLICY_SQUASH:-auto}
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
    echo "[proxy] env=${env} seed=${seed} source=${SOURCE} gpu=${gpu}"
    cmd=(python run_proxy_alignment.py
      --config configs/phase0_d4rl.yaml
      --env_name "${env}"
      --seed "${seed}"
      --device cuda:0
      --candidate_source "${SOURCE}"
      --policy_squash "${POLICY_SQUASH}")
    if [[ -n "${OUT_DIR}" ]]; then
      cmd+=(--out_dir "${OUT_DIR}")
    fi
    CUDA_VISIBLE_DEVICES=${gpu} "${cmd[@]}" > "logs/proxy_${SOURCE}_${env}_seed${seed}.log" 2>&1 &
    job_id=$((job_id + 1))
    if (( job_id % ${#GPUS[@]} == 0 )); then
      wait
    fi
  done
done
wait
