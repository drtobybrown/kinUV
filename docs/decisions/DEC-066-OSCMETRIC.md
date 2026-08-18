---
id: DEC-066-OSCMETRIC
status: accepted
generation: 2
date: 2026-08-18
owner: 066-4-rings
supersedes: generation 1 (Matérn-∞ / SE kernel claim; 100-mock grid; hardcoded Δv=10.6)
---
# Quantitative oscillation metric for ring V_c regularisation

**Question:** How to quantify rotation-curve ringing for the `λ_reg` calibration?

## Answer

Discrete curvature ratio, using the **npz channel width**, not a hardcoded 10.6 km/s:

`Ω_k = |V_{k+1} − 2 V_k + V_{k-1}| / Δv_chan`

`Δv_chan` is the channel width of the visibilities being fit (replica bin-4 ≈ 5.3 km/s; YAML bin-8 ≈ 10.6 km/s). Record it with the npz in 066-0.

## Bayesian content (what it is, not a GP slogan)

The curvature penalty `λ_reg Σ_k (V_{k+1} − 2 V_k + V_{k-1})²` is a Gaussian prior on second differences: a discrete **cubic-spline / integrated-Wiener** regulariser. It is **not** a squared-exponential / Matérn-∞ kernel. Do not implement a GP; implement the sum of squares.

## 066 calibration protocol

1. Truth = arctan `(V₀ = 200 km/s, r_t = 3″)` at real 066 `(u,v,ν,w)` plus Gaussian noise.
2. **066 budget:** 20 mocks × ~5 values of `λ_reg`. Densify only if the two criteria below conflict.
3. Accept the **smallest** `λ_reg` such that:
   - `max_k Ω_k < 0.3` in ≥95% of mocks
   - **and** recovered `(V₀, r_t)` lie within 1σ of truth in ≥68% of mocks
4. If no `λ` satisfies both, increase ring count. If they conflict at all ring counts, that is a model-specification failure, not a licence to drop the metric.

## Monotonicity (066 only)

`Φ_mono = μ_mono Σ_k max(0, V_k − V_{k+1})²`

with `μ_mono = λ_reg`. This equates a first-difference penalty to a second-difference penalty and is a **heuristic**, not a derivation. Soft quadratic, differentiable, no hard bound. Applies inside the last knot; the outermost ring may decline without penalty.

## Model selection

If Stage B (rings) does not beat Stage A (arctan) by more than `ΔAIC = 2 k_extra` with `k_extra = N_rings − 2`, **keep Stage A**.

AIC not BIC: BIC with `N_vis ~ 10⁵` rejects every non-trivial curve; the effective independent sample is ~10³; `λ_reg` already regularises.
