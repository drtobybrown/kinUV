#!/usr/bin/env bash
# Run one target's four collaborator NUTS chains in a CANFAR headless session.
set -euo pipefail

TARGET="${1:?target required}"
DURABLE_ROOT="${2:?durable root required}"
CODE_COMMIT="${3:?code commit required}"
PROJECT="/arc/projects/KILOGAS/analysis/toby_sandbox"
SOURCE_REPO="${PROJECT}/kinUV"
MAP_ROOT="${PROJECT}/results/incoming/collaborator-delivery-20260908/map"
VENV="/arc/home/thbrown/kinuv-venv-recovery"
SESSION_ID="${SKAHA_SESSION_ID:-${skaha_sessionid:-${HOSTNAME:-unknown}}}"
SCRATCH_ROOT="/scratch/kinuv-${USER:-thbrown}/collaborator-${SESSION_ID}-${TARGET}"
CODE_ROOT="${SCRATCH_ROOT}/repo"
LOG_ROOT="${SCRATCH_ROOT}/logs"
ARC_LOG_ROOT="${DURABLE_ROOT}/logs/${TARGET}"

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
export PYTHONPATH="${CODE_ROOT}/src"
export KINUV_WORKSPACE="${PROJECT}"

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
  printf '{"target":"%s","session_id":"%s","exit_code":%d}\n' \
    "${TARGET}" "${SESSION_ID}" "${code}" \
    > "${DURABLE_ROOT}/headless_exit_${TARGET}.json.tmp"
  mv "${DURABLE_ROOT}/headless_exit_${TARGET}.json.tmp" \
    "${DURABLE_ROOT}/headless_exit_${TARGET}.json"
}
trap finish EXIT

cd "${CODE_ROOT}"
"${VENV}/bin/python" scripts/run_collaborator_nuts_controller.py \
  --map-root "${MAP_ROOT}" \
  --scratch-root "${SCRATCH_ROOT}/chains" \
  --durable-root "${DURABLE_ROOT}" \
  --python "${VENV}/bin/python" \
  --targets "${TARGET}" \
  --max-workers 4 \
  --cpu-sets 0-3,4-7,8-11,12-15 \
  --status-file "${DURABLE_ROOT}/controller_status_${TARGET}.json" \
  --controller-log "${LOG_ROOT}/controller.log" \
  --worker-log-root "${LOG_ROOT}"
