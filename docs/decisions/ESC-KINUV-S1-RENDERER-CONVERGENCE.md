---
id: ESC-KINUV-S1-RENDERER-CONVERGENCE
status: resolved-by-astra-pending-executable-amendment
date: 2026-09-06
authority_requested: Astra
implementation_commit: 0c95240b2f7e238ba6c0666f61871c65558494b0
artifact: results/validation/crossdomain-recovery-s1-20260906-r2
resolution: DEC-KINUV-S1-CONTINUUM-AND-S2-CONTRACTS
---
# S1 intrinsic-renderer convergence escalation

## Decision requested

Freeze an anti-aliased intrinsic KinMS rendering contract, or provide another
renderer prescription that can satisfy the existing target-path resolution
gate without changing its `|Delta chi2| <= 0.1` threshold. The Implementer has
stopped before altering KinMS cloud deposition because that changes the
comparison transform governed by
[`DEC-KINUV-CROSSDOMAIN-RECOVERY`](DEC-KINUV-CROSSDOMAIN-RECOVERY.md).

The recommended contract is to use KinMS 3.0.13 to generate continuous
projected cloud positions and line-of-sight velocities, deposit those clouds
with a conservative anti-aliased spatial/spectral kernel, and apply a
flux-audited analytic constant-dispersion LOSVD. The resulting intrinsic cube
would still pass through kinUV's shared PB, NUFFT, and Hann/bin operators once.
The revised decision must state whether this is an eligible KinMS comparator,
which deposition kernel is frozen, and how its spatial and LOSVD convergence
axes are defined.

## Evidence

Exact commit `0c95240b2f7e238ba6c0666f61871c65558494b0`
repairs every bounded contract defect identified by both S1 reviewers:

- KinMS receives `posAng=(360-PA)%360`; no completed cube is reflected.
- A signed asymmetric worker test recovers PA to `0.001397 deg`, its east/north
  offset to `1.94e-5 arcsec`, and its spectral centroid to `1.34e-4` native
  channel without post-render interpolation.
- The checksum-bound NPZ metadata and sidecar must agree exactly; flux is
  recomputed from the loaded cube.
- The runner enforces the accepted S0 manifest, target and data hashes, the
  isolated KinMS environment lock, a clean exact `dev` commit, atomic
  publication, and read-only sealing.
- Both nominal/common-high and independent/common-high comparisons are gated.
  They pass for KGAS066 at `0.01901` and `0.02640` chi-square and for KGAS007
  at `0.003332` and `0.000458`.
- Descriptive runtime provenance records `1196.34 s` wall time and `2,634,220
  KiB` as the maximum observed single-process peak RSS on the recorded
  CPU/JAX-FINUFFT environment. This is not a concurrent process-tree memory
  measurement.
- The revised code passed the complete repository suite: `239 passed, 8
  skipped`.

The new per-axis target test exposes unresolved numerical dependence:

| Doubled axis | KGAS066 absolute Delta chi-square | KGAS007 absolute Delta chi-square | Gate |
|---|---:|---:|---:|
| Image grid | 37.3989 | 0.3865 | <= 0.1 |
| Radial quadrature | 4.5185 | 0.5779 | <= 0.1 |
| Azimuth quadrature | 0.02791 | 0.001360 | <= 0.1 |
| Dispersion quadrature | 896.7242 | 143.1000 | <= 0.1 |
| Native spectral sub-sampling | 0.0 | 0.0 | <= 0.1 |

Source inspection and the axis-sensitivity matrix identify KinMS nearest-cell
cloud deposition and the finite Gauss-Hermite approximation used to make the
stochastic API deterministic as the leading candidate mechanisms. The dossier
does not uniquely isolate causality. Raising sample counts further repeats a
discretized operation whose target-path chi-square has not converged; replacing
that deposition or LOSVD is a transform-level decision.
The frozen threshold has not been relaxed and the failed dossier has not been
promoted. Its read-only record contains 77 total files (76 manifest-indexed
payloads plus the manifest) and occupies 49.03 MiB (51,415,691 bytes). It is
indexed by manifest SHA-256
`97b4d4f526680fbfaad9fbb47147ab47aef8d9803cf923a4123b5c7cc674c45d`.

The Senior Registrar independently verified all 76 payload hashes and sizes,
the absence of extra or missing payloads, the source snapshots and input
hashes, the gate arithmetic, and this stop condition. Two reproducibility
limits remain explicit: the KinMS lock records an ordered `pip freeze`, package
version, and `kinms/__init__.py` hash but no artifact hashes or complete
distribution-tree hash; the parent record contains six package versions rather
than a complete environment lock, and both interpreter paths point into
ephemeral `/scratch`. Derived binned visibility arrays were not retained, so a
fully independent gate recomputation requires rerunning the snapshotted
operator against the hash-bound inputs. The scalar values themselves can be
recomputed arithmetically from the retained component metrics.

## Debugging history and stop condition

The first S1 attempt used stochastic clouds and failed render convergence.
The first deterministic debugging pass removed stochastic noise but retained
large pixel-deposition aliasing. The phase-ensemble pass brought the originally
declared scalar gates below threshold, after which independent review exposed
the PA error and missing integration axes. The `r2` run repaired those defects
and demonstrated that the full target-path resolution gate still fails by up
to three orders of magnitude. Further work now requires changing the frozen
comparator transform, so the autonomous protocol's architectural stop condition
applies.

## Decisions also required before S2

S2 cannot be promoted from the current inputs even after S1 is resolved:

1. KGAS007's historical visibility NPZ has no time, baseline, or antenna
   metadata. Five correlation-aware real holdout folds cannot be constructed.
   The DEC already requires that claim to remain `BLOCKED`; a provenance-complete
   `ms2kinuv` re-export is required for real held-out superiority.
2. The candidate model requires an independently sourced prior in `cos(i)` with
   quoted uncertainty and source. KGAS007 has only a fixed inclination value,
   while no approved uncertainty/source is registered. The Implementer will not
   invent this physical prior.
3. Astra must freeze the BMAJ product used for the turnover grid, LOS-amplitude
   bounds, covariance candidate family and selection rule, projected-gradient
   norm convention, and whether every fold repeats the complete 72-fit
   turnover/start matrix.

No S2 implementation or run has begun.
