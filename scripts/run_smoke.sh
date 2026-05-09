#!/usr/bin/env bash
set -euo pipefail

# Smoke test: verifies install and pipeline logic. Not for reporting.
python run_pipeline.py \
  --config configs/phase0_d4rl.yaml \
  --env_name halfcheetah-medium-replay-v2 \
  --seed 0 \
  --device cuda:0 \
  --iql_steps 2000 \
  --bc_steps 1000 \
  --fqe_steps 2000 \
  --n_eval_states 512 \
  --candidate_source bc
