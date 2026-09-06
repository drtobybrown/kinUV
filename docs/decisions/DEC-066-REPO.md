---
id: DEC-066-REPO
status: accepted
date: 2026-08-18
owner: planner
---
# Repository strategy

**Question:** kinUV standalone or an external fitting backend?

**Answer:** kinUV is a standalone fitter with no legacy fitting backend. KinMS is not a runtime dependency. Measurement Set extraction belongs to the separate `ms2kinuv` ETL companion and ends at the versioned NPZ contract. First git commit is Field Guide + these ADRs + `.gitignore` on branch `kgas066-slice`. Untracked `src/` scaffolding is discarded, not adopted.
