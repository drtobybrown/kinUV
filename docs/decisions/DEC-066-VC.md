---
id: DEC-066-VC
status: accepted
date: 2026-08-18
owner: 066-4-rings
---
# Rotation curve representation

**Question:** How is V_c(r) parameterised for 066?

**Answer:** Two-stage.

1. Stage A: arctan `V_c(r) = V_0 (2/π) arctan(r/r_t)` plus flux, PA, vsys, σ, (dx, dy).
2. Stage B: 6–8 rings initialised to that arctan. Curvature penalty `λ_reg Σ (V_{k+1}−2V_k+V_{k−1})²`. Soft monotonicity on (066 only). L-BFGS bounds `V_k ∈ [0, 400]` km/s.

Calibrate `λ_reg` per DEC-066-OSCMETRIC **before** real visibilities. If Stage B does not beat Stage A on the AIC gate, keep Stage A. No unconstrained cold-start rings. No γ in this vector.
