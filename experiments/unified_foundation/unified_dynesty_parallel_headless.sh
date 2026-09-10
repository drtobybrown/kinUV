#!/usr/bin/env bash
# One shared-JIT thread-pool unified Dynesty replicate in one CANFAR session.
set -uo pipefail

target="${1:?target required}"
replicate="${2:?replicate required}"
seed="${3:?seed required}"
durable_root="${4:?durable root required}"
code_commit="${5:?code commit required}"
map_result="${6:?MAP result required}"
map_commit="${7:?MAP commit required}"
workers="${8:-4}"
n_effective="${9:-2000}"
slices="${10:-17}"
dlogz_init="${11:-0.01}"
resume_root="${12:-}"
case "${workers}" in
  4|8|16|32) ;;
  *) printf 'workers must be 4, 8, 16, or 32\n' >&2; exit 2 ;;
esac
session_id="${SKAHA_SESSION_ID:-${skaha_sessionid:-${HOSTNAME:-unknown}}}"
workspace="/arc/projects/KILOGAS/analysis/toby_sandbox"
vendor="${workspace}/results/incoming/unified-sampler-spike-20260909/vendor"
scratch="/scratch/kinuv-${USER:-$(id -un)}/unified-dynesty-parallel-${session_id}"
repo="${scratch}/repo"
log="${scratch}/worker.log"
durable="${durable_root}/${target}/replicate-${replicate}"
mkdir -p "${repo}" "${scratch}/work" "${scratch}/tmp" "${scratch}/jax-cache" "${scratch}/vendor" "${durable}"
git -C "${workspace}/kinUV" archive "${code_commit}" | tar -x -C "${repo}"
cp -a "${vendor}/dynesty" "${vendor}/dynesty-2.1.5.dist-info" "${scratch}/vendor/"

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS=0
export XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONUNBUFFERED=1 TERM=dumb
export JAX_COMPILATION_CACHE_DIR="${scratch}/jax-cache" TMPDIR="${scratch}/tmp"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 JAX_NUM_THREADS=1
export PYTHONPATH="${scratch}/vendor:${repo}/src:${repo}/experiments/unified_foundation"
export KINUV_WORKSPACE="${workspace}"

resume_args=()
if [ -n "${resume_root}" ]; then
  resume_args=(--resume-root "${resume_root}")
fi
set +e
/arc/home/thbrown/kinuv-venv-recovery/bin/python \
  "${repo}/experiments/unified_foundation/unified_dynesty_parallel.py" \
  --target "${target}" --replicate "${replicate}" --seed "${seed}" \
  --map-result "${map_result}" --map-commit "${map_commit}" --code-commit "${code_commit}" \
  --scratch "${scratch}/work" --durable "${durable_root}" \
  --nlive 500 --n-effective "${n_effective}" --workers "${workers}" --slices "${slices}" \
  --dlogz-init "${dlogz_init}" "${resume_args[@]}" \
  >"${log}" 2>&1
code=$?
set -e
cp "${log}" "${durable}/worker.log.tmp" && mv "${durable}/worker.log.tmp" "${durable}/worker.log"
printf '{"target":"%s","replicate":%s,"session_id":"%s","code_commit":"%s","map_commit":"%s","workers":%s,"n_effective":%s,"slices":%s,"dlogz_init":%s,"exit_code":%d,"completed_utc":"%s"}\n' \
  "${target}" "${replicate}" "${session_id}" "${code_commit}" "${map_commit}" "${workers}" "${n_effective}" "${slices}" "${dlogz_init}" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"${durable}/headless_exit.json.tmp"
mv "${durable}/headless_exit.json.tmp" "${durable}/headless_exit.json"
exit "${code}"
