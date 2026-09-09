#!/bin/bash
set -uo pipefail

target="$1"
output_root="$2"
maxiter="${3:-3}"
family="${4:-}"
repo="/arc/projects/KILOGAS/analysis/toby_sandbox/kinUV"
venv="/arc/home/thbrown/kinuv-venv-recovery"
run_user="${USER:-$(id -un)}"
run_tag="unified-representation-${target}-$$"
scratch="/scratch/kinuv-${run_user}/${run_tag}"
durable="${output_root}/${target}"

mkdir -p "$scratch" "$durable"
export JAX_PLATFORMS=cpu
export JAX_ENABLE_X64=1
export JAX_COMPILATION_CACHE_DIR="${scratch}/jax-cache"
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TMPDIR="$scratch"
export PYTHONPATH="${repo}/src"
export PYTHONUNBUFFERED=1

cd "$repo"
family_args=()
if [ -n "$family" ]; then
  family_args=(--family "$family")
fi
"${venv}/bin/python" experiments/unified_foundation/representation_worker.py \
  --target "$target" --output-root "$output_root" --maxiter "$maxiter" "${family_args[@]}" \
  >"${scratch}/worker.log" 2>&1
rc=$?
cp "${scratch}/worker.log" "${durable}/worker.log.tmp"
mv "${durable}/worker.log.tmp" "${durable}/worker.log"
printf '{"target":"%s","exit_code":%d,"scratch":"%s"}\n' \
  "$target" "$rc" "$scratch" >"${durable}/headless_exit.json.tmp"
mv "${durable}/headless_exit.json.tmp" "${durable}/headless_exit.json"
exit "$rc"
