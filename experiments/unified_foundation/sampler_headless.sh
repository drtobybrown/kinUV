#!/usr/bin/env bash
# One bounded sampler-spike process in one flexible CANFAR headless session.
set -euo pipefail

TARGET="${1:?target required}"
SAMPLER="${2:?sampler required}"
RUN_ROOT="${3:?durable /arc run root required}"
PROJECT="/arc/projects/KILOGAS/analysis/toby_sandbox"
REPO="${PROJECT}/kinUV"
VENV="/arc/home/thbrown/kinuv-venv-recovery"
VENDOR="${RUN_ROOT}/vendor"
SESSION_ID="${SKAHA_SESSION_ID:-${skaha_sessionid:-${HOSTNAME:-unknown}}}"
SCRATCH_ROOT="/scratch/kinuv-${USER:-thbrown}/unified-sampler-${SESSION_ID}-${TARGET}-${SAMPLER}"

mkdir -p "${SCRATCH_ROOT}/tmp" "${SCRATCH_ROOT}/jax-cache" "${RUN_ROOT}/${TARGET}/${SAMPLER}"
export TMPDIR="${SCRATCH_ROOT}/tmp"
export XDG_CACHE_HOME="${SCRATCH_ROOT}/xdg"
export JAX_COMPILATION_CACHE_DIR="${SCRATCH_ROOT}/jax-cache"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 TERM=dumb
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 JAX_NUM_THREADS=4
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
export PYTHONPATH="${VENDOR}:${REPO}/src:${REPO}/scripts"

OUTPUT="${RUN_ROOT}/${TARGET}/${SAMPLER}/result.json"
LOG="${RUN_ROOT}/${TARGET}/${SAMPLER}/worker.log"
STARTED="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
set +e
"${VENV}/bin/python" "${REPO}/experiments/unified_foundation/sampler_baseline.py" \
  --target "${TARGET}" --sampler "${SAMPLER}" --output "${OUTPUT}" \
  --profile-repeats 5 --warmup 8 --samples 8 --max-tree-depth 6 \
  --nlive 20 --slices 2 --maxiter 2 --maxcall 80 >"${LOG}" 2>&1
CODE=$?
set -e
printf '{"target":"%s","sampler":"%s","session_id":"%s","started_utc":"%s","completed_utc":"%s","exit_code":%d}\n' \
  "${TARGET}" "${SAMPLER}" "${SESSION_ID}" "${STARTED}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${CODE}" \
  >"${RUN_ROOT}/${TARGET}/${SAMPLER}/exit.json.tmp"
mv "${RUN_ROOT}/${TARGET}/${SAMPLER}/exit.json.tmp" "${RUN_ROOT}/${TARGET}/${SAMPLER}/exit.json"
exit "${CODE}"
