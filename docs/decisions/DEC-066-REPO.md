---
id: DEC-066-REPO
status: accepted
date: 2026-08-18
owner: planner
---
# Repository strategy

**Question:** kinUV standalone or uvfit backend?

**Answer:** Standalone 066 fitter in kinUV. uvfit selectable backend is deferred until 066 MAP exists — wiring emcee back in is how the last stack failed. KinMS is not a runtime dependency. First git commit is Field Guide + these ADRs + `.gitignore` on branch `kgas066-slice`. Untracked `src/` scaffolding is discarded, not adopted.
