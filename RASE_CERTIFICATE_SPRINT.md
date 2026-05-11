# RASE Action-Level Certificate Sprint

This package is a focused diagnostic sprint. It is not another clean Phase-0 rerun.
It reuses existing `outputs/rase_phase0/{env}/seed{seed}/iql.pt` and `bc.pt`, refreshes
legacy FQE checkpoints if needed, then evaluates whether any certificate can lower
short-rollout false-positive risk for selected candidate actions.

## Core question

Can action-level RASE produce a calibrated accepted-set certificate that is materially
better than raw max-Q selection?

Go criteria used for decision:

- AUROC against `rollout_positive` should be at least roughly 0.60 for a certificate.
- `precision@coverage=0.3` should improve over raw `pred_pair_gap` by at least about 10 percentage points.
- rollout false-positive rate should drop without collapsing coverage.
- the effect should hold on at least two of HalfCheetah / Hopper / Walker2d.

## Experiments run by the full script

1. FQE checkpoint refresh: train audited TwinQ/min-Q FQE only if an old incompatible checkpoint is found.
2. Same-pair rollout certificate diagnostic:
   - envs: HalfCheetah, Hopper, Walker2d medium-replay
   - seeds: 41, 42, 43
   - sources: BC and IQL
   - M: 16, 64, 256
   - continuation: IQL
   - rollout labels are saved together with proxy features on the exact same candidate pairs.
3. Cross-fitted verifier scoring:
   - 3-fold IQL critics are trained per env/seed.
   - exact rollout pairs are rescored by the held-out fold critic.
4. Aggregated analysis:
   - AUROC / AUPRC against `rollout_positive`
   - precision at target coverage
   - risk-coverage rows
   - calibration-split thresholds at target false-positive rates

## One-command run

```bash
cd /root/remote/RASE
conda activate gcrlo

RASE_GPU_IDS="0 1 2 3 4 5 6" \
RASE_OUT_DIR=outputs/rase_phase0 \
RASE_POLICY_SQUASH=auto \
bash scripts/run_rase_certificate_sprint_full.sh
```

If your available GPUs are `1..7`, use:

```bash
RASE_GPU_IDS="1 2 3 4 5 6 7" \
RASE_OUT_DIR=outputs/rase_phase0 \
RASE_POLICY_SQUASH=auto \
bash scripts/run_rase_certificate_sprint_full.sh
```

## Faster fallback

For a faster first pass, reduce rollout states and crossfit steps:

```bash
RASE_GPU_IDS="0 1 2 3 4 5 6" \
RASE_OUT_DIR=outputs/rase_phase0 \
RASE_POLICY_SQUASH=auto \
RASE_ROLLOUT_N_STATES=256 \
RASE_ROLLOUT_MAX_PAIRS=96 \
RASE_FOLD_IQL_STEPS=50000 \
bash scripts/run_rase_certificate_sprint_full.sh
```

## Important outputs

The master script prints paths at the end. The most important files are:

```text
outputs/rase_phase0/certificate_sprint_analysis_<RUN_TAG>/
  RASE_certificate_sprint_report.md
  certificate_pairs_all.csv
  certificate_score_alignment.csv
  certificate_risk_coverage.csv
  certificate_calibration_split.csv
  certificate_gap_summary.csv

outputs/rase_phase0/<env>/seed<seed>/diagnostics/
  rollout_pairs_bc_iql.csv
  rollout_pairs_iql_iql.csv
  crossfit_verified_pairs_bc_iql.csv
  crossfit_verified_pairs_iql_iql.csv
```

The generated tarball is named:

```text
rase_certificate_sprint_<RUN_TAG>.tar.gz
```

Upload that tarball for analysis.

## What changed relative to v5

- `run_rollout_diagnostic.py` supports `--candidate_ms`, `--add_proxy_features`, `--save_action_dims`.
- `rollout_pairs_*_iql.csv` now saves proxy features and selected action vectors for the exact rollout-labeled pairs.
- `run_crossfit_verify_pairs.py` trains fold-specific IQL verifier critics and rescoring the exact rollout pairs.
- `run_certificate_analysis.py` aggregates rollout/crossfit results and reports certificate quality.
- `scripts/run_rase_certificate_sprint_full.sh` schedules the full sprint on 7 GPUs.
