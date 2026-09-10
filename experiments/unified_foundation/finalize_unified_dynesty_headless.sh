#!/usr/bin/env bash
# Aggregate two complete dynesty replicates into one immutable incoming packet.
set -uo pipefail

target="${1:?target required}"
dynesty_root="${2:?dynesty root required}"
map_result="${3:?MAP result required}"
map_commit="${4:?MAP commit required}"
code_commit="${5:?code commit required}"
output="${6:?immutable output required}"
session_id="${SKAHA_SESSION_ID:-${skaha_sessionid:-${HOSTNAME:-unknown}}}"
workspace="/arc/projects/KILOGAS/analysis/toby_sandbox"
scratch="/scratch/kinuv-${USER:-$(id -un)}/unified-dynesty-finalize-${session_id}"
repo="${scratch}/repo"
mkdir -p "${repo}" "${scratch}/tmp" "${scratch}/jax-cache" "$(dirname "${output}")"
git -C "${workspace}/kinUV" archive "${code_commit}" | tar -x -C "${repo}"
export JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONUNBUFFERED=1 TERM=dumb
export JAX_COMPILATION_CACHE_DIR="${scratch}/jax-cache" TMPDIR="${scratch}/tmp"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=4"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=1 JAX_NUM_THREADS=4
export PYTHONPATH="${repo}/src:${repo}/experiments/unified_foundation"
export KINUV_WORKSPACE="${workspace}"

log="${scratch}/worker.log"
set +e
/arc/home/thbrown/kinuv-venv-recovery/bin/python \
  "${repo}/experiments/unified_foundation/finalize_unified_dynesty.py" \
  --target "${target}" --dynesty-root "${dynesty_root}" --map-result "${map_result}" \
  --map-commit "${map_commit}" --code-commit "${code_commit}" --output "${output}" \
  >"${log}" 2>&1
code=$?
set -e
cp "${log}" "$(dirname "${output}")/finalize-${target}.log.tmp" && \
  mv "$(dirname "${output}")/finalize-${target}.log.tmp" "$(dirname "${output}")/finalize-${target}.log"
printf '{"target":"%s","session_id":"%s","code_commit":"%s","exit_code":%d}\n' \
  "${target}" "${session_id}" "${code_commit}" "${code}" \
  >"$(dirname "${output}")/finalize-${target}-exit.json.tmp"
mv "$(dirname "${output}")/finalize-${target}-exit.json.tmp" "$(dirname "${output}")/finalize-${target}-exit.json"
exit "${code}"
