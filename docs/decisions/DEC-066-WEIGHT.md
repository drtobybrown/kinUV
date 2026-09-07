---
id: DEC-066-WEIGHT
status: accepted
generation: 2
date: 2026-08-18
owner: 066-6-likelihood
---
# χ² weight scale

**Question:** What replaces `weight_scale_factor: 0.5`?

**Answer:** The empirical per-SPW scale in DEC-066-SPECRESP: `s = 2 / ⟨w|V|²⟩_line-free` on the fit npz, sanity `0.3 < s < 1.5`. Do not use YAML 0.5. Do not use theoretical `12/29` unless that npz actually shows Hanning+bin overcount (`⟨w|V|²⟩ ~ 4.8` for N=8). Full banded GLS is after 066 MAP.

For `ms2kinuv-npz-v2`, `sigma_complex² = E[|noise|²]` and the stored
likelihood weight is `w = 2/sigma_complex² = 1/sigma_component²`. Standard
Measurement Set `WEIGHT` and `WEIGHT_SPECTRUM` already use the latter
component convention and are retained without another factor of two. Flags
set the corresponding stored weight to zero. The schema records this identity
in `weight_convention` so ingestion fails closed on ambiguity.
