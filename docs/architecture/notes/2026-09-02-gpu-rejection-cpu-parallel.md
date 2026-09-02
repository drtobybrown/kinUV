# GPU rejection and CPU-parallel NUTS (architecture note, not ADR)

Date: 2026-09-02. Existing ids only; no new `DEC-*`. Official MAP unchanged.

## Empirical result (066, 881×95, frozen s=0.5136098555284736)

| Path | eval/s | chi2 @ MAP | grad(U) six-axis |
|---|---:|---:|---|
| CPU jax-finufft (`JAX_PLATFORMS=cpu`) | **3.01** | 168675.6 | 0.43 s |
| CUDA jax-finufft (H100 MIG 1g.12gb) | **0.55** | 168675.6 | 2.80 s |

CUDA identity passed (`|chi2 − 168675.6| < 1`) but the kernel is **≈5.5× slower** than CPU. The proposed 10× wall target (17440 s → 1744 s) failed before mixing: four GPU 1-chain jobs were ≈4–11% through warmup after ~45 min with multi-hour ETAs.

## Why GPU loses on this problem

1. **Small array** (881×95 vis): kernel launch and dispatch dominate; NUTS leapfrog is sequential.
2. **MIG 1g.12gb**: partitioned memory bandwidth; platform GPU usage read 0% during slow warmup (accounting artifact or idle dispatch).
3. **CPU jax-finufft** uses host AVX paths and cache-friendly sequential NUFFT on a problem sized for CPU SIMD locality.

## Canonical architecture (formalized)

**Production NUTS = CPU only**, `kinuv-venv-recovery`, `skaha/astroml:latest`, flexible headless by default (`DEC-067-RUNNER`).

Speed path when needed: **four independent flexible CPU sessions**, one chain each (`--chain-id 1..4`), host merge via `scripts/merge_nuts_chains.py`. Serial 4-chain single session remains valid (receding product `sd3ckpf2`).

## Actions taken

- GPU CANFAR sessions terminated; CUDA venv and GPU run dirs purged.
- GPU runner code and probe scripts removed from `origin/dev`.
- Negative benchmark numbers preserved here; partial GPU chain outputs discarded.

## Still running

Approaching PA 25.2 NUTS (`xgepg7qy`, run `KGAS066-20260902T085027Z-nuts-pa25`) is the active CPU science dependency for the conjugate mode at PA ≈ 25.2°. Do not interrupt.

## Do not

- Re-open GPU NUTS without a new propose + dual review and a kernel that beats CPU eval/s on the official MAP chi2 path.
- Quote GPU timing as production guidance.
- Start G4.
