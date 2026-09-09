#!/usr/bin/env bash
# Run one collaborator NUTS chain in one flexible CANFAR headless session.
set -euo pipefail

TARGET="${1:?target required}"
CHAIN_ID="${2:?chain id required}"
DURABLE_ROOT="${3:?durable root required}"
CODE_COMMIT="${4:?code commit required}"
PROJECT="/arc/projects/KILOGAS/analysis/toby_sandbox"
SOURCE_REPO="${PROJECT}/kinUV"
MAP_ROOT="${PROJECT}/results/incoming/collaborator-delivery-20260908/map"
VENV="/arc/home/thbrown/kinuv-venv-recovery"
SESSION_ID="${SKAHA_SESSION_ID:-${skaha_sessionid:-${HOSTNAME:-unknown}}}"
SCRATCH_ROOT="/scratch/kinuv-${USER:-thbrown}/collaborator-${SESSION_ID}-${TARGET}-c${CHAIN_ID}"
CODE_ROOT="${SCRATCH_ROOT}/repo"
LOG_ROOT="${SCRATCH_ROOT}/logs"
ARC_LOG_ROOT="${DURABLE_ROOT}/logs/${TARGET}/chain-${CHAIN_ID}"
SEED_BASE=9100
if [[ "${TARGET}" == "KGAS007" ]]; then SEED_BASE=9200; fi
SEED=$((SEED_BASE + CHAIN_ID))

mkdir -p "${CODE_ROOT}" "${LOG_ROOT}" "${ARC_LOG_ROOT}" \
  "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/jax-cache" "${SCRATCH_ROOT}/xdg"
git -C "${SOURCE_REPO}" archive "${CODE_COMMIT}" | tar -x -C "${CODE_ROOT}"

export TMPDIR="${SCRATCH_ROOT}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg"
export JAX_COMPILATION_CACHE_DIR="${SCRATCH_ROOT}/jax-cache"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export TERM=dumb
export PYTHONPATH="${CODE_ROOT}/src:${CODE_ROOT}/scripts"
export KINUV_WORKSPACE="${PROJECT}"
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1
export JAX_NUM_THREADS=4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"

sync_logs() {
  cp -a "${LOG_ROOT}/." "${ARC_LOG_ROOT}/" 2>/dev/null || true
}
(
  while true; do
    sleep 60
    sync_logs
  done
) &
SYNC_PID=$!

finish() {
  code=$?
  kill "${SYNC_PID}" 2>/dev/null || true
  wait "${SYNC_PID}" 2>/dev/null || true
  sync_logs
  exit_file="${DURABLE_ROOT}/headless_exit_${TARGET}_chain${CHAIN_ID}.json"
  printf '{"target":"%s","chain_id":%s,"session_id":"%s","exit_code":%d}\n' \
    "${TARGET}" "${CHAIN_ID}" "${SESSION_ID}" "${code}" > "${exit_file}.tmp"
  mv "${exit_file}.tmp" "${exit_file}"
}
trap finish EXIT

cd "${CODE_ROOT}"
"${VENV}/bin/python" scripts/run_collaborator_nuts_chain.py \
  --selected-map "${MAP_ROOT}/${TARGET}/selected_map.json" \
  --chain-id "${CHAIN_ID}" \
  --seed "${SEED}" \
  --scratch "${SCRATCH_ROOT}/chains/${TARGET}" \
  --durable "${DURABLE_ROOT}/${TARGET}" \
  --log-dir "${LOG_ROOT}" \
  --warmup 1000 \
  --samples 1000 \
  --chunk 100 \
  --target-accept 0.90 \
  --max-tree-depth 10
