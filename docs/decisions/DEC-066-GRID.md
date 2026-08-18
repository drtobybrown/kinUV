---
id: DEC-066-GRID
status: accepted
date: 2026-08-18
owner: 066-5-nufft
---
# Image / NUFFT grid

**Question:** What sky grid?

**Answer:** Nyquist vs 066 max baseline (~305 kλ) with margin: `1/(2 · cell_rad) > max_baseline_λ`. Assert; do not silently override from an imaging-cube header (production uvkin replaced YAML 0.1″ with 0.4″). Size the grid to CO extent plus PB, not blindly 256²@0.1″. If one likelihood eval exceeds ~0.5 s on this M1, the grid is wrong.
