#!/usr/bin/env bash
# One independent unified dynesty replicate in one flexible CANFAR session.
set -uo pipefail

target="${1:?target required}"
replicate="${2:?replicate required}"
seed="${3:?seed required}"
durable_root="${4:?durable root required}"
code_commit="${5:?code commit required}"
map_result="${6:?MAP result required}"
map_commit="${7:?MAP commit required}"
session_id="${SKAHA_SESSION_ID:-${skaha_sessionid:-${HOSTNAME:-unknown}}}"
workspace="/arc/projects/KILOGAS/analysis/toby_sandbox"
vendor="${workspace}/results/incoming/unified-sampler-spike-20260909/vendor"
scratch="/scratch/kinuv-${USER:-$(id -un)}/unified-dynesty-${session_id}"
repo="${scratch}/repo"
log="${scratch}/worker.log"
durable="${durable_root}/${target}/replicate-${replicate}"
mkdir -p "${repo}" "${scratch}/work" "${scratch}/tmp" "${scratch}/jax-cache" "${scratch}/vendor" "${durable}"
git -C "${workspace}/kinUV" archive "${code_commit}" | tar -x -C "${repo}"
cp -a "${vendor}/dynesty" "${vendor}/dynesty-2.1.5.dist-info" "${scratch}/vendor/"

export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 TERM=dumb
export JAX_COMPILATION_CACHE_DIR="${scratch}/jax-cache" TMPDIR="${scratch}/tmp"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 JAX_NUM_THREADS=4
export PYTHONPATH="${scratch}/vendor:${repo}/src:${repo}/experiments/unified_foundation"
export KINUV_WORKSPACE="${workspace}"

set +e
/arc/home/thbrown/kinuv-venv-recovery/bin/python \
  "${repo}/experiments/unified_foundation/unified_dynesty.py" \
  --target "${target}" --replicate "${replicate}" --seed "${seed}" \
  --map-result "${map_result}" --map-commit "${map_commit}" --code-commit "${code_commit}" \
  --scratch "${scratch}/work" --durable "${durable_root}" --nlive 500 --n-effective 2000 \
  >"${log}" 2>&1
code=$?
set -e
cp "${log}" "${durable}/worker.log.tmp" && mv "${durable}/worker.log.tmp" "${durable}/worker.log"
printf '{"target":"%s","replicate":%s,"session_id":"%s","code_commit":"%s","map_commit":"%s","exit_code":%d,"completed_utc":"%s"}\n' \
  "${target}" "${replicate}" "${session_id}" "${code_commit}" "${map_commit}" "${code}" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  >"${durable}/headless_exit.json.tmp"
mv "${durable}/headless_exit.json.tmp" "${durable}/headless_exit.json"
exit "${code}"
