---
id: DEC-066-ZEROMODEL
status: accepted
generation: 1
date: 2026-08-18
owner: planner
---
# Zero model for the Δχ² gate

**Question:** What is the zero model against which MAP improvement is measured?

## Answer

`V_model(u,v,ν) = 0` for all visibilities.

`χ²_zero = Σ_k w_k |d_k|²`

`Δχ² = χ²_zero − χ²_MAP = 2 Re[m† W d] − m† W m`

Chosen because it has no parameters, is the natural detection null, and is comparable across galaxies.

Reduced χ² is **not** the gate. Hanning/binning plus WEIGHT mis-scale inflate χ²/N even for a perfect model (066 forensic: 1.111 at MAP vs 1.120 at zero). Report `Δχ²` vs zero.
