---
id: DEC-066-OSCMETRIC
status: accepted
generation: 4
date: 2026-08-18
owner: 066-4-rings
supersedes: generation 3 (no outer/inner evaluation BC)
---
# Quantitative oscillation metric for ring V_c regularisation

**Question:** How to quantify rotation-curve ringing for the `λ_reg` calibration?

## Answer

Discrete curvature ratio, using the **npz channel width**, not a hardcoded 10.6 km/s:

`Ω_k = |V_{k+1} − 2 V_k + V_{k-1}| / Δv_chan`

`Δv_chan` is the channel width of the visibilities being fit (replica bin-4 ≈ 5.3 km/s; YAML bin-8 ≈ 10.6 km/s). Record it with the npz in 066-0.

## Knot placement (066)

Innermost knot `r_0 ≥ 0.5 × BMAJ`. For the Ico restoring beam BMAJ ≈ 1.30″ that is **`r_0 ≥ 0.65″`**. Uniform rings down to `r = 0` overfit beam-smeared inner gradients into a central spike. Subsequent knots: `Δr ≈ 1.0–1.3″` over the ~7.5″ CO disk, `N_rings = 6–8` (DEC-066-VC).

## Evaluation outside the knots (Stage B rings only)

- **Outer:** for all `R > r_last`, `V_c(R) = V_c(r_last)` (flat). No polynomial or spline extrapolation. Quadrature nodes and mask pixels out to ~10″ must use this.
- **Inner:** for `R < r_0`, **solid body** `V_c(R) = V_c(r_0) · (R / r_0)`, not a flat `V_c(r_0)`. Flat inner BC leaves a derivative jump at `r_0` that can still spike the centre. Stage A arctan is defined for all `R` and is **not** flattened.

Unit test: evaluate `V_c` on a grid to 15″ with `r_last = 7.5″`; `dV/dR = 0` for `R > r_last`; `V(0) = 0`.

## Bayesian content (what it is, not a GP slogan)

The curvature penalty `λ_reg Σ_k (V_{k+1} − 2 V_k + V_{k-1})²` is a Gaussian prior on second differences: a discrete **cubic-spline / integrated-Wiener** regulariser. It is **not** a squared-exponential / Matérn-∞ kernel. Do not implement a GP; implement the sum of squares.

## 066 calibration protocol

1. Truth = arctan `(V₀ = 200 km/s, r_t = 3″)` at real 066 `(u,v,ν,w)` plus Gaussian noise.
2. **066 budget:** 20 mocks × ~5 values of `λ_reg`. Densify only if the three criteria below conflict.
3. Accept the **smallest** `λ_reg` such that:
   - `max_k Ω_k < 0.3` in ≥95% of mocks
   - **and** recovered `(V₀, r_t)` lie within 1σ of truth in ≥68% of mocks
   - **and** mean recovered `V₀` is not biased low relative to Stage A (arctan-only) by more than the 1σ scatter. This guards the heuristic `μ_mono = λ_reg` near turnover `r_t`.
4. If no `λ` satisfies all three, increase ring count (keeping `r_0 ≥ 0.5 BMAJ`). If they conflict at all ring counts, that is a model-specification failure, not a licence to drop the metric.

## Monotonicity (066 only)

`Φ_mono = μ_mono Σ_k max(0, V_k − V_{k+1})²`

with `μ_mono = λ_reg`. This equates a first-difference penalty to a second-difference penalty and is a **heuristic**, not a derivation. Soft quadratic, differentiable, no hard bound. Applies inside the last knot; the outermost ring may decline without penalty.

## Model selection

If Stage B (rings) does not beat Stage A (arctan) by more than `ΔAIC = 2 k_extra` with `k_extra = N_rings − 2`, **keep Stage A**.

AIC not BIC: BIC with `N_vis ~ 10⁵` rejects every non-trivial curve; the effective independent sample is ~10³; `λ_reg` already regularises.
