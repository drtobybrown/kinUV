---
id: DEC-066-SHIFT
status: accepted
date: 2026-08-18
owner: 066-8-map
---
# Phase centre (dx, dy)

**Question:** Are (dx, dy) frozen at zero?

**Answer:** No. They are MAP parameters, implemented as `V ↦ V exp(−2πi (u dx + v dy))` (analytic ramp, not an image shift). Gaussian prior σ = 0.5″, support ±2″. Mock inject 0.3″ and require recovery. After MAP, freeze for NUTS only if both are consistent with 0 at <1σ — that freeze is a result, not an input. YAML `[0,0]` is a seed.
