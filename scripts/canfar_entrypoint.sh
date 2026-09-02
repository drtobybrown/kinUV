#!/usr/bin/env bash
# Headless entrypoint (DEC-067-RUNNER). Compute on /scratch; durable I/O on project /arc.
# Platform canfar logs expire in ~1 hour. Verbose stdout stays on scratch; worker.log
# is overwrite-copied to /arc every 60 s and on exit (not a per-sample NFS tee).
set -euo pipefail

RUN_ID="${1:-kgas066-nuts}"
USER_NAME="${USER:-thbrown}"
PROJECT="${KINUV_PROJECT:-/arc/projects/KILOGAS/analysis/toby_sandbox}"
REPO="${KINUV_REPO:-${PROJECT}/kinUV}"
VENV="${KINUV_VENV:-/arc/home/thbrown/kinuv-venv-recovery}"
# Default: /arc/projects/KILOGAS/analysis/toby_sandbox/kinuv_runs
RUNS_ROOT="${KINUV_RUNS:-${PROJECT}/kinuv_runs}"
RUN_DIR="${RUNS_ROOT}/${RUN_ID}"
SESSION="${SKAHA_SESSION_ID:-${HOSTNAME:-local}}"
SCRATCH_JOB="/scratch/kinuv-${USER_NAME}/${SESSION}"
mkdir -p "${SCRATCH_JOB}/tmp" "${SCRATCH_JOB}/jax-cache" "${SCRATCH_JOB}/xdg" \
  "${RUN_DIR}/logs" "${RUN_DIR}/checkpoints"
export TMPDIR="${SCRATCH_JOB}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export JAX_COMPILATION_CACHE_DIR="${SCRATCH_JOB}/jax-cache"
export XDG_CACHE_HOME="${SCRATCH_JOB}/xdg"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export KINUV_RUNS="${RUNS_ROOT}"
export KINUV_PROJECT="${PROJECT}"

SCRATCH_LOG="${SCRATCH_JOB}/worker.log"
ARC_LOG="${RUN_DIR}/worker.log"
# High-frequency stdout (NumPyro tqdm) stays on scratch. Overwrite-copy to /arc
# every 60 s and on exit — do not tee every sample onto NFS.
if command -v stdbuf >/dev/null 2>&1; then
  exec > >(stdbuf -oL -eL tee -a "${SCRATCH_LOG}") 2>&1
else
  exec > >(tee -a "${SCRATCH_LOG}") 2>&1
fi
copy_worker_log() {
  if [[ -f "${SCRATCH_LOG}" ]]; then
    cp -f "${SCRATCH_LOG}" "${ARC_LOG}.copying" 2>/dev/null || return 0
    mv -f "${ARC_LOG}.copying" "${ARC_LOG}" 2>/dev/null || true
  fi
}
(
  while true; do
    sleep 60
    copy_worker_log
  done
) &
LOG_SYNC_PID=$!

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "=== kinuv entrypoint start utc=$(utc) host=$(hostname) pid=$$ run_id=${RUN_ID} ==="
echo "repo=${REPO} venv=${VENV} scratch=${SCRATCH_JOB}"
echo "runs=${RUN_DIR} session=${SESSION} user=${USER_NAME}"
echo "ulimit -v=$(ulimit -v 2>/dev/null || echo na) nproc=$(nproc 2>/dev/null || echo na)"
if command -v free >/dev/null 2>&1; then
  free -h || true
fi
env | grep -E '^(SKAHA_|JAX_|KINUV_|PYTHON)' | sort || true

# shellcheck disable=SC1091
source "${VENV}/bin/activate"
cd "${REPO}"
export PYTHONPATH="${REPO}/src"
if [[ "${KINUV_SKIP_PULL:-0}" != "1" ]]; then
  git fetch origin dev >/dev/null 2>&1 || true
  git checkout dev >/dev/null 2>&1 || true
  git pull --ff-only origin dev >/dev/null 2>&1 || true
fi
git rev-parse --short=12 HEAD || true
python -c "import jax; print('jax', jax.__version__, 'devices', jax.devices())" || true

flush_logs() {
  ec=$?
  echo "=== kinuv entrypoint exit utc=$(utc) code=${ec} ==="
  kill "${LOG_SYNC_PID}" 2>/dev/null || true
  wait "${LOG_SYNC_PID}" 2>/dev/null || true
  copy_worker_log
  sync -f "${ARC_LOG}" 2>/dev/null || sync || true
}
trap flush_logs EXIT
PA_ARGS=()
if [[ -n "${KINUV_PA_INIT:-}" ]]; then
  PA_ARGS+=(--pa-init "${KINUV_PA_INIT}")
fi
python "${REPO}/scripts/run_kgas066_nuts_headless.py" --run-id "${RUN_ID}" "${PA_ARGS[@]}"
