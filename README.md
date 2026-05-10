# RASE Phase-0 to Phase-1 Experiment Code

RASE = **Risk-Controlled Advantage Set Extraction**.

This repository implements the current RASE-only experimental line. It is **not** PCAR and does not contain offline-to-online action replacement. The goal is to study and control false-positive policy improvement induced by high-Q candidate selection in offline RL.


## v5 checkpoint-compatibility fix

This release fixes the common error:

```text
RuntimeError: FQE checkpoint is from an older incompatible version.
```

Cause: early Phase-0 outputs under `outputs/rase_phase0/` used a legacy single-Q FQE checkpoint and a clipped Gaussian policy. The current audited diagnostics require TwinQ/min-Q FQE. v5 now:

1. detects legacy `outputs/rase_phase0/.../fqe_iql_ref.pt`;
2. preserves legacy policy semantics with `--policy_squash auto` by selecting `clip`;
3. backs up the old FQE checkpoint;
4. retrains a compatible TwinQ/min-Q FQE checkpoint automatically unless `--no_auto_retrain_fqe` is passed.

To diagnose your existing Phase-0 outputs without rerunning IQL/BC:

```bash
python run_rollout_diagnostic.py \
  --config configs/phase0_d4rl.yaml \
  --out_dir outputs/rase_phase0 \
  --env_name walker2d-medium-replay-v2 \
  --seed 41 \
  --device cuda:0 \
  --candidate_source bc \
  --policy_squash auto
```

For a fresh audited rerun, use the default output directory `outputs/rase_phase0_v5` and keep `policy_squash: tanh`.

## Current research question

For a dataset state `s`, sample a candidate action pool `A_M(s)` from BC / IQL / random / perturb proposals. Raw selection chooses

```text
a* = argmax_a Q_IQL(s, a), a in A_M(s)
```

RASE asks whether this high-Q action is actually a reliable improvement over the data action `a_data`, and whether we can extract an accepted action set with controlled false-positive risk.

Main pairwise quantities:

```text
predicted_gap = Q_IQL(s, a*) - Q_IQL(s, a_data)
fqe_gap       = Q_FQE(s, a*) - Q_FQE(s, a_data)
FPI_FQE       = predicted_gap > 0 and fqe_gap <= 0
```

The old `Q_IQL(s,a*) - V_IQL(s)` statistic is logged as `iql_adv_vs_v_mean` but is not the main false-positive label.

## What changed in v4

v4 keeps the audited Phase-0 sweep and adds the next experimental stages:

1. **Phase-0.5 rollout diagnostic**: short-horizon action replacement rollouts validate whether FQE labels agree with simulator evidence.
2. **Phase-1 cross-fitted critic diagnostic**: train fold-specific IQL critics and evaluate candidate selection on held-out states.
3. **Phase-1/2 proxy alignment**: evaluate support NLL, kNN support distance, critic/FQE disagreement, action distance, and composite RASE scores against FQE-positive labels.
4. **Calibrated risk-coverage**: compare precision at fixed coverage instead of relying on global absolute thresholds.

## Build Docker image

```bash
docker build -t rase-phase0:cu118 .
bash docker/run_container.sh
conda activate rase
```

## Data cache sanity

Before multi-GPU runs, prefetch D4RL serially:

```bash
bash scripts/repair_d4rl_cache.sh
bash scripts/prefetch_d4rl.sh \
  halfcheetah-medium-replay-v2 \
  hopper-medium-replay-v2 \
  walker2d-medium-replay-v2 \
  antmaze-umaze-v2
```

## Phase-0: reproduce candidate-pool sweep

```bash
bash scripts/run_smoke.sh
bash scripts/run_sweep_3090.sh bc
RASE_SKIP_PREFETCH=1 bash scripts/run_sweep_3090.sh iql
RASE_SKIP_PREFETCH=1 bash scripts/run_sweep_3090.sh perturb
```

Outputs:

```text
outputs/rase_phase0_v4/<env>/seed<seed>/
  sweep_<source>.csv
  risk_coverage_<source>.csv
  plots/*.png
```

## Phase-0.5: validate FQE with short rollouts

Run on locomotion tasks first. AntMaze state restoration is less reliable for this action-level diagnostic.

```bash
python run_rollout_diagnostic.py \
  --config configs/phase0_d4rl.yaml \
  --env_name hopper-medium-replay-v2 \
  --seed 41 \
  --device cuda:0 \
  --candidate_source bc \
  --n_eval_states 256 \
  --rollout_horizon 50 \
  --rollout_repeats 3 \
  --max_pairs_per_m 128
```

Outputs:

```text
diagnostics/rollout_pairs_<source>_<continuation>.csv
diagnostics/rollout_summary_<source>_<continuation>.csv
```

Key checks:

```text
fqe_rollout_corr
pred_rollout_corr
rollout_fpi_rate
pred_rollout_gap_mean
```

## Phase-1: cross-fitted critic diagnostic

```bash
python run_crossfit_iql.py \
  --config configs/phase0_d4rl.yaml \
  --env_name hopper-medium-replay-v2 \
  --seed 41 \
  --device cuda:0 \
  --candidate_source bc \
  --num_folds 3 \
  --fold_iql_steps 100000 \
  --heldout_eval_states 2048
```

Outputs:

```text
crossfit/crossfit_sweep_<source>.csv
crossfit/fold*/iql_fold.pt
```

This tests whether the selection-induced predicted/FQE gap persists when the selecting critic is trained out-of-fold.

## Phase-1/2: proxy alignment and calibrated risk-coverage

```bash
python run_proxy_alignment.py \
  --config configs/phase0_d4rl.yaml \
  --env_name hopper-medium-replay-v2 \
  --seed 41 \
  --device cuda:0 \
  --candidate_source bc \
  --n_eval_states 4096 \
  --knn_ref_size 20000
```

Outputs:

```text
diagnostics/proxy_details_<source>.csv
diagnostics/proxy_alignment_<source>.csv
diagnostics/calibrated_risk_coverage_<source>.csv
```

Main metrics:

```text
AUROC/AUPRC for FQE-positive improvement
precision_at_cov_0.2 / 0.3 / 0.5 / 0.7
calibrated risk-coverage rows for pred_pair_gap, rase_score, rase_score_v2
```

## Recommended next run order

Start with locomotion replay tasks where Phase-0 gave strong evidence:

```text
hopper-medium-replay-v2
walker2d-medium-replay-v2
halfcheetah-medium-replay-v2
```

Then run:

```bash
bash scripts/run_phase05_rollout.sh bc
bash scripts/run_phase1_proxy.sh bc
bash scripts/run_phase1_crossfit.sh bc
```

Repeat with `iql` source after BC is verified.

## Unit tests

```bash
bash scripts/run_unit_tests.sh
```

The unit tests do not require D4RL.
