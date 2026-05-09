# RASE Phase-0 Pre-Experiment

RASE = **Risk-Controlled Advantage Set Extraction**. This code implements the first diagnostic stage: whether enlarging a candidate action pool increases predicted high-Q selection while creating false-positive policy improvements.

## What this repo tests

For each dataset state `s`, sample candidate actions from a behavior model, IQL actor, random actions, or local perturbations. For each candidate-pool size `M`, select `argmax_a Q_IQL(s,a)` and compare:

- predicted advantage from IQL: `Q_IQL(s,a*) - V_IQL(s)`;
- empirical proxy advantage from FQE: `Q_FQE(s,a*) - Q_FQE(s,a_data)`;
- false-positive improvement: predicted advantage positive, FQE advantage non-positive;
- RASE-v1 risk-coverage curve with `Z = predicted_advantage - lambda * behavior_NLL`.

This is a Phase-0 diagnostic, not a final RASE algorithm.

## Recommended pre-experiment environments

Start with D4RL v2 state-based tasks:

```text
halfcheetah-medium-replay-v2
hopper-medium-replay-v2
walker2d-medium-replay-v2
antmaze-umaze-v2
```

Minimum: 4 tasks x 3 seeds. Add `antmaze-medium-play-v2` and `antmaze-medium-diverse-v2` after smoke tests.

## Build Docker image

```bash
cd rase_phase0
docker build -t rase-phase0:cu118 .
```

## Start Docker container

```bash
cd rase_phase0
bash docker/run_container.sh
conda activate rase
```

## Smoke test

```bash
bash scripts/run_smoke.sh
```

## Full multi-GPU sweep

Inside the container:

```bash
mkdir -p logs
bash scripts/run_sweep_3090.sh bc
bash scripts/run_sweep_3090.sh iql
bash scripts/run_sweep_3090.sh perturb
```

The script launches independent env/seed jobs across GPUs via `CUDA_VISIBLE_DEVICES`. This is preferable to DDP because each offline RL seed is small and independent.

## Outputs

Each run writes to:

```text
outputs/rase_phase0/<env_name>/seed<seed>/
  iql.pt
  bc.pt
  fqe_iql_ref.pt
  sweep_<source>.csv
  risk_coverage_<source>.csv
  plots/*.png
```

Main plots:

- `pred_adv_mean.png`
- `fqe_adv_mean.png`
- `pred_empirical_gap.png`
- `fpi_rate_cond_pred_positive.png`
- `risk_coverage_M*.png`

## Go / no-go signal

Continue RASE if at least two tasks show:

1. predicted advantage increases with candidate-pool size `M`;
2. FQE advantage does not increase at the same rate, or decreases;
3. conditional false-positive improvement rate increases with `M`;
4. RASE score improves risk-coverage over raw high-Q selection.

If this does not happen, move the project toward subgoal/trajectory false-positive stitching rather than action-level RASE.
