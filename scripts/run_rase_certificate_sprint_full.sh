#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

# RASE action-level certificate sprint.
# Goal: decide whether action-level RASE can produce a certificate that lowers
# rollout false-positive risk beyond raw max-Q selection.

export D4RL_SUPPRESS_IMPORT_ERROR=1
export MUJOCO_GL=${MUJOCO_GL:-osmesa}
export PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-osmesa}

export RASE_GPU_IDS="${RASE_GPU_IDS:-0 1 2 3 4 5 6}"
export RASE_OUT_DIR="${RASE_OUT_DIR:-outputs/rase_phase0}"
export RASE_POLICY_SQUASH="${RASE_POLICY_SQUASH:-auto}"
export RASE_CONFIG="${RASE_CONFIG:-configs/phase0_d4rl.yaml}"

# Core sprint design: no AntMaze, no perturb, no clean Phase-0 rerun.
export RASE_ENVS="${RASE_ENVS:-halfcheetah-medium-replay-v2 hopper-medium-replay-v2 walker2d-medium-replay-v2}"
export RASE_SEEDS="${RASE_SEEDS:-41 42 43}"
export RASE_SOURCES="${RASE_SOURCES:-bc iql}"

# Rollout-label experiment.  Keep M compact and put compute into simulator labels.
export RASE_ROLLOUT_CANDIDATE_MS="${RASE_ROLLOUT_CANDIDATE_MS:-16,64,256}"
export RASE_ROLLOUT_N_STATES="${RASE_ROLLOUT_N_STATES:-384}"
export RASE_ROLLOUT_MAX_PAIRS="${RASE_ROLLOUT_MAX_PAIRS:-128}"
export RASE_ROLLOUT_HORIZON="${RASE_ROLLOUT_HORIZON:-50}"
export RASE_ROLLOUT_REPEATS="${RASE_ROLLOUT_REPEATS:-3}"
export RASE_KNN_REF_SIZE="${RASE_KNN_REF_SIZE:-20000}"
export RASE_KNN_BATCH_SIZE="${RASE_KNN_BATCH_SIZE:-512}"

# FQE refresh is only needed if old single-Q FQE checkpoints remain.
export RASE_FQE_STEPS="${RASE_FQE_STEPS:-200000}"

# Crossfit verifier.  Full sprint default covers all three locomotion tasks.
# To shorten: export RASE_CROSSFIT_ENVS="hopper-medium-replay-v2 walker2d-medium-replay-v2"
export RASE_CROSSFIT_ENVS="${RASE_CROSSFIT_ENVS:-halfcheetah-medium-replay-v2 hopper-medium-replay-v2 walker2d-medium-replay-v2}"
export RASE_NUM_FOLDS="${RASE_NUM_FOLDS:-3}"
export RASE_FOLD_IQL_STEPS="${RASE_FOLD_IQL_STEPS:-100000}"

RUN_TAG="${RASE_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="logs/certificate_sprint_${RUN_TAG}"
JOB_DIR="tmp/certificate_sprint_${RUN_TAG}"
mkdir -p "${LOG_DIR}" "${JOB_DIR}"

echo "============================================================"
echo "RASE certificate sprint"
echo "RUN_TAG=${RUN_TAG}"
echo "RASE_GPU_IDS=${RASE_GPU_IDS}"
echo "RASE_OUT_DIR=${RASE_OUT_DIR}"
echo "RASE_ENVS=${RASE_ENVS}"
echo "RASE_SEEDS=${RASE_SEEDS}"
echo "RASE_SOURCES=${RASE_SOURCES}"
echo "RASE_ROLLOUT_CANDIDATE_MS=${RASE_ROLLOUT_CANDIDATE_MS}"
echo "RASE_CROSSFIT_ENVS=${RASE_CROSSFIT_ENVS}"
echo "LOG_DIR=${LOG_DIR}"
echo "============================================================"

echo "[1/6] Unit tests and Python syntax checks"
bash scripts/run_unit_tests.sh

echo "[2/6] Serial D4RL prefetch / cache validation"
bash scripts/prefetch_d4rl.sh ${RASE_ENVS}

echo "[3/6] Refresh audited FQE checkpoints if needed"
REFRESH_JOBS="${JOB_DIR}/refresh_fqe.jobs"
: > "${REFRESH_JOBS}"
for env_name in ${RASE_ENVS}; do
  for seed in ${RASE_SEEDS}; do
    cat >> "${REFRESH_JOBS}" <<EOF
python run_refresh_fqe.py --config ${RASE_CONFIG} --out_dir ${RASE_OUT_DIR} --env_name ${env_name} --seed ${seed} --device cuda:0 --policy_squash ${RASE_POLICY_SQUASH} --fqe_steps ${RASE_FQE_STEPS}
EOF
  done
done
python scripts/run_job_queue.py --job_file "${REFRESH_JOBS}" --gpu_ids "${RASE_GPU_IDS}" --log_dir "${LOG_DIR}" --prefix refresh_fqe --stop_on_failure

echo "[4/6] Rollout-labeled certificate pairs with same-pair proxy features"
ROLLOUT_JOBS="${JOB_DIR}/rollout_certificate.jobs"
: > "${ROLLOUT_JOBS}"
for source in ${RASE_SOURCES}; do
  for env_name in ${RASE_ENVS}; do
    for seed in ${RASE_SEEDS}; do
      cat >> "${ROLLOUT_JOBS}" <<EOF
python run_rollout_diagnostic.py --config ${RASE_CONFIG} --out_dir ${RASE_OUT_DIR} --env_name ${env_name} --seed ${seed} --device cuda:0 --candidate_source ${source} --policy_squash ${RASE_POLICY_SQUASH} --candidate_ms ${RASE_ROLLOUT_CANDIDATE_MS} --n_eval_states ${RASE_ROLLOUT_N_STATES} --max_pairs_per_m ${RASE_ROLLOUT_MAX_PAIRS} --rollout_horizon ${RASE_ROLLOUT_HORIZON} --rollout_repeats ${RASE_ROLLOUT_REPEATS} --continuation_policy iql --add_proxy_features --knn_ref_size ${RASE_KNN_REF_SIZE} --knn_batch_size ${RASE_KNN_BATCH_SIZE} --save_action_dims -1
EOF
    done
  done
done
python scripts/run_job_queue.py --job_file "${ROLLOUT_JOBS}" --gpu_ids "${RASE_GPU_IDS}" --log_dir "${LOG_DIR}" --prefix rollout_cert --stop_on_failure

echo "[5/6] Cross-fitted verifier scoring on the exact rollout pairs"
CROSSFIT_JOBS="${JOB_DIR}/crossfit_verify.jobs"
: > "${CROSSFIT_JOBS}"
for env_name in ${RASE_CROSSFIT_ENVS}; do
  for seed in ${RASE_SEEDS}; do
    sources_csv="$(echo ${RASE_SOURCES} | tr ' ' ',')"
    cat >> "${CROSSFIT_JOBS}" <<EOF
python run_crossfit_verify_pairs.py --config ${RASE_CONFIG} --out_dir ${RASE_OUT_DIR} --env_name ${env_name} --seed ${seed} --device cuda:0 --sources ${sources_csv} --continuation_policy iql --policy_squash ${RASE_POLICY_SQUASH} --num_folds ${RASE_NUM_FOLDS} --fold_iql_steps ${RASE_FOLD_IQL_STEPS}
EOF
  done
done
python scripts/run_job_queue.py --job_file "${CROSSFIT_JOBS}" --gpu_ids "${RASE_GPU_IDS}" --log_dir "${LOG_DIR}" --prefix crossfit_verify --stop_on_failure

echo "[6/6] Aggregate certificate metrics"
analysis_dir="${RASE_OUT_DIR}/certificate_sprint_analysis_${RUN_TAG}"
python run_certificate_analysis.py \
  --out_dir "${RASE_OUT_DIR}" \
  --analysis_dir "${analysis_dir}" \
  --envs "$(echo ${RASE_ENVS} | tr ' ' ',')" \
  --seeds "$(echo ${RASE_SEEDS} | tr ' ' ',')" \
  --sources "$(echo ${RASE_SOURCES} | tr ' ' ',')" \
  --continuation_policy iql

echo "[packaging]"
RESULT_TAR="rase_certificate_sprint_${RUN_TAG}.tar.gz"
tar -czf "${RESULT_TAR}" \
  "${LOG_DIR}" \
  "${analysis_dir}" \
  ${RASE_OUT_DIR}/*/seed*/diagnostics/rollout_pairs_*_iql.csv \
  ${RASE_OUT_DIR}/*/seed*/diagnostics/rollout_summary_*_iql.csv \
  ${RASE_OUT_DIR}/*/seed*/diagnostics/crossfit_verified_pairs_*_iql.csv \
  ${RASE_OUT_DIR}/*/seed*/diagnostics/crossfit_verification_summary_*_iql.csv \
  ${RASE_OUT_DIR}/*/seed*/diagnostics/rollout_*_config.json \
  ${RASE_OUT_DIR}/*/seed*/crossfit/crossfit_verify_config.json

ls -lh "${RESULT_TAR}"

echo "============================================================"
echo "DONE: RASE certificate sprint"
echo "analysis_dir=${analysis_dir}"
echo "result_tar=${RESULT_TAR}"
echo "Inspect report: ${analysis_dir}/RASE_certificate_sprint_report.md"
echo "============================================================"
