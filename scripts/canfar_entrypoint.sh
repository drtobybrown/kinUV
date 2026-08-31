#!/usr/bin/env bash
# Headless entrypoint (DEC-067-RUNNER). Activate recovery venv; TMP on /scratch.
# Tee stdout/stderr onto /arc: platform canfar logs expire in ~1 hour.
set -euo pipefail

RUN_ID="${1:-kgas066-nuts}"
USER_NAME="${USER:-thbrown}"
REPO="${KINUV_REPO:-/arc/projects/KILOGAS/analysis/toby_sandbox/kinUV}"
VENV="${KINUV_VENV:-/arc/home/thbrown/kinuv-venv-recovery}"
RUNS_ROOT="${KINUV_RUNS:-/arc/home/thbrown/kinuv_runs}"
RUN_DIR="${RUNS_ROOT}/${RUN_ID}"
SCRATCH_BASE="/scratch/kinuv-${USER_NAME}"
mkdir -p "${SCRATCH_BASE}" "${RUN_DIR}/logs"
export TMPDIR="${SCRATCH_BASE}/tmp"
export TEMP="${TMPDIR}"
export TMP="${TMPDIR}"
export JAX_COMPILATION_CACHE_DIR="${SCRATCH_BASE}/jax-cache"
export XDG_CACHE_HOME="${SCRATCH_BASE}/xdg"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"
export JAX_ENABLE_X64=1
export PYTHONUNBUFFERED=1
mkdir -p "${TMPDIR}" "${JAX_COMPILATION_CACHE_DIR}" "${XDG_CACHE_HOME}"

WORKER_LOG="${RUN_DIR}/worker.log"
# Line-buffered tee so a kill still leaves the last lines on NFS.
if command -v stdbuf >/dev/null 2>&1; then
  exec > >(stdbuf -oL -eL tee -a "${WORKER_LOG}") 2>&1
else
  exec > >(tee -a "${WORKER_LOG}") 2>&1
fi

utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
echo "=== kinuv entrypoint start utc=$(utc) host=$(hostname) pid=$$ run_id=${RUN_ID} ==="
echo "repo=${REPO} venv=${VENV} scratch=${SCRATCH_BASE}"
echo "session=${SKAHA_SESSION_ID:-unset} user=${USER_NAME}"
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

trap 'ec=$?; echo "=== kinuv entrypoint exit utc=$(utc) code=${ec} ==="; sleep 0.2 || true' EXIT
python "${REPO}/scripts/run_kgas066_nuts_headless.py" --run-id "${RUN_ID}"
