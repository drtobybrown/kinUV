---
id: DEC-066-INFER
status: accepted
date: 2026-08-18
owner: 066-8 / 066-10
---
# Inference order

**Question:** MAP or NUTS first?

**Answer:** MAP (L-BFGS on the smooth model) first. NUTS only if MAP Δχ² vs the zero model is real and vsys/PA/flux mocks recover. Do not sample a likelihood that cannot beat the noise pedestal. Do not switch to dynesty because MAP failed.
