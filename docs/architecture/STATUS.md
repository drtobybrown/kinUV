---
generation: 24
phase: s4-paired-benchmark-active
code_freeze: false
next_role: implementer-sol
board: crossdomain-recovery-s4
build_licensed: true
pending:
  - nonrotation-null-bootstrap
  - stage-b-smoothness-recalibration
  - s4-paired-selection
  - exact-workflow-posterior-calibration
last_propose: docs/decisions/DEC-KINUV-S1-CONTINUUM-AND-S2-CONTRACTS.md
last_review: docs/reviews/2026-09-07-code-review-b-crossdomain-s3.md
last_review_a: docs/reviews/2026-09-07-code-review-a-crossdomain-s3.md
last_review_b: docs/reviews/2026-09-07-code-review-b-crossdomain-s3.md
user_review: docs/reviews/artifacts/2026-09-05-kgas007-nuts/
open_questions: []
deadlocks: []
canon_generation: 24
---

## Agent Run Status

* **Phase:** S0--S3 are closed; S4 paired recovery and real-cube comparison are active
* **Last Action:** Closed S3 after both review seats accepted exact implementation commit `b2ac6bc` and its checksum-bound two-target dossier
* **Decisions Made:** The likelihood stays on native TOPO frequencies; the measured TOPO-to-LSRK drift is 0.037 km/s for KGAS066 and 0.064 km/s for KGAS007, below the PI-authorized 1.0 km/s reporting gate. The PI override in `DEC-HUMAN-OVERRIDE-RIGHTSIZED-GATES` replaces universal multi-start identity with consensus in the top likelihood cluster.
* **Blockers / Gates:** S4 must show at least 10% lower projected-velocity RMSE than frozen stock KinMS and improve overall reduced chi-square on both real targets. Failure or KGAS066 regression is an explicit escalation boundary.
* **Next Step:** Render the two S3-selected models through the frozen diagnostic operator, score them and stock KinMS on the same official cubes/masks, and run the structured visibility-residual diagnostics required by the S2 reviewers.

The extraction implementation is on `dev` at `246bc19` in ms2kinuv and its
kinUV ingestion boundary is at `d177f58`. The casacore-backed ms2kinuv suite
passes all 16 tests. Standard MS weights are retained as
`w=1/sigma_component^2=2/E[|complex noise|^2]`; they are not doubled during
export.

## S2 input and covariance evidence

The canonical inputs are `visibilities/KILOGAS066.v2.npz` and
`visibilities/KILOGAS007.v2.npz`; their checksums and Measurement Set lineage
are recorded in `visibilities/MANIFEST.json`. Both exports retain native rows,
standard partition IDs, antenna pairs, timestamps, channel centers and edges,
flags, weights, and XX/YY-to-Stokes-I combination lineage.

The immutable covariance dossier is
`results/validation/crossdomain-recovery-s2-20260907-r1/`. Five disjoint folds
contain 8 time blocks for KGAS066 and 9 for KGAS007, with a 12.096 s embargo.
C1 is selected for both targets: native adjacent-channel rho is 0.29770 and
0.29769, respectively. Whitened mean, variance, and lag-one gates pass; maximum
remaining lag-one correlation is 0.02922 for KGAS066 and 0.02930 for KGAS007.

The Field Guide and Review Board charter govern the active S4 implementation.
The S1 closure did not alter the physical brightness profile, velocity
prescription, target parameters, or frozen thresholds.

## S2 geometry and covariance closure

S2 is **CLOSED** at exact implementation commit `8248a05`. Reviewer A accepted
the science and numerical evidence in
`docs/reviews/2026-09-07-code-review-a-crossdomain-s2.md`; Reviewer B accepted
the software and reproducibility evidence in
`docs/reviews/2026-09-07-code-review-b-crossdomain-s2.md`. The immutable
geometry dossiers are
`results/validation/crossdomain-recovery-s2-geometry-20260907-r3-066/` and
`results/validation/crossdomain-recovery-s2-geometry-20260907-r3-007/`.

Both targets retain all 72 fixed-turnover and 12 released-turnover fits.
KGAS066 has eight mutually consistent starts in the top likelihood cluster;
KGAS007 has all twelve. The selected projected speeds are 184.003 and
96.263 km/s, the turnover ratios are 0.28619 and 0.42123 BMAJ, and the
per-complex projected gradients are `7.044e-5` and `7.003e-5`. No selected
parameter is on a bound. The reported intrinsic speeds and inclinations remain
diagnostics because the registered inclination prior is isotropic.

C1 is selected in every held-out group for both targets. Native adjacent-channel
correlations are 0.297702 and 0.297688; the maximum whitened residual lag is
0.02930. The independently reproduced within-run TOPO-to-LSRK drifts are
0.037134 and 0.064040 km/s, safely below the PI-authorized 1 km/s gate. S4 must
add baseline-, antenna-, time-, and real/imaginary cross-component residual
checks before any held-out superiority claim.

## S3 joint-ablation closure

S3 is **CLOSED** at exact implementation commit `b2ac6bc`. Reviewer A accepted
the science and numerical evidence in
`docs/reviews/2026-09-07-code-review-a-crossdomain-s3.md`; Reviewer B accepted
the software and reproducibility evidence in
`docs/reviews/2026-09-07-code-review-b-crossdomain-s3.md`. The checksum-bound
dossier is `results/validation/crossdomain-recovery-s3-20260907-r3/`, and the
full suite passed with 259 tests and 8 skips.

KGAS066 gains 223.193 chi-square from the positive rank-three emissivity basis,
rejects four velocity knots at a 4.821 gain, and retains a two-zone dispersion
addition with a further 152.500 gain. Its selected projected arctan amplitude
is 183.633 km/s, with inner and outer dispersions 7.473 and 9.878 km/s.
KGAS007 gains 12.692 from emissivity, retains four supported projected-speed
knots at a 32.875 gain, and rejects the two-zone dispersion addition. Its knots
are 59.083, 88.334, 83.139, and 95.502 km/s. Both knot Hessian blocks are full
rank on the registered local scale. The model-family chi-square differences
are retention criteria, not likelihood-ratio significance claims; S4 decides
predictive superiority.

## S1 operator/comparator closure

The first dossier at `results/validation/crossdomain-recovery-s1-20260906/`
is retained as pre-review evidence. The revised read-only dossier is
`results/validation/crossdomain-recovery-s1-20260906-r2/`; it is a failed-gate
record and is not promoted. No fit, bootstrap, posterior, NUTS, or G4 campaign
was run.

| Gate | KGAS066 | KGAS007 | Limit |
|---|---:|---:|---:|
| Analytic complex-visibility relative L2 | 4.70e-14 | 4.70e-14 | <=1e-6 |
| Analytic noise-normalized component RMS | 1.04e-12 | 1.04e-12 | <=0.001 |
| Zero-baseline flux relative error | 3.70e-15 | 3.70e-15 | <=0.1% |
| Native centroid error (channel) | <1e-9 | <1e-9 | <=0.02 |
| Doubled-sampling delta chi2 | 2.40e-5 | 2.40e-5 | <=0.1 |
| Nominal/high render RMS (thermal SD) | 6.73e-5 | 1.58e-5 | <=0.1 |
| Independent high-repeat RMS (thermal SD) | 4.58e-5 | 1.13e-5 | <=0.1 |
| Nominal/high absolute chi2 change | 0.01901 | 0.003332 | <=0.1 |
| Independent/high absolute chi2 change | 0.02640 | 0.000458 | <=0.1 |
| Maximum per-axis doubled-sampling absolute chi2 change | **896.724** | **143.100** | <=0.1; **FAIL** |
| Signed worker PA error | 0.001397 deg | 0.001397 deg | <=3 deg |
| S0 baseline replay absolute chi2 error | 0.0 | 0.0 | <=0.1 |

The pragmatic continuum-adapter runs are retained as failed numerical evidence.
The second and final bounded iteration is
`results/validation/crossdomain-recovery-s1-20260906-r4/` at exact commit
`c55c985c131a48e1516a1e189920be5a67694074`.

| r4 refinement axis | KGAS066 absolute Delta chi-square / relative L2 | KGAS007 absolute Delta chi-square / relative L2 | Status |
|---|---:|---:|---|
| Spatial grid | 0.06773 / 9.26e-5 | 0.003810 / 9.38e-5 | pass |
| Radial quadrature | **0.7568 / 9.27e-4** | **1.0361 / 1.26e-3** | **fail** |
| Azimuth quadrature | 3.72e-7 / 2.12e-6 | 7.58e-6 / 2.50e-7 | pass |
| Analytic spectral subdivision | 0.0 / 8.65e-15 | 0.0 / 2.37e-14 | pass |

The analytic LOSVD eliminated the previous 896.7/143.1 dispersion-order
failure. Cubic assignment-window compensation reduced the spatial-grid changes
below both thresholds. Gauss-Legendre radial refinement improved but did not
close the unchanged `|Delta chi2| <= 0.1` and relative-L2 `<=1e-4` gates.
Per the two-iteration stop rule, no S2 work followed.

The r5 implementation replaces the single-interval radial rule with composite
three-point Gauss-Legendre quadrature split at every piecewise-linear
surface-brightness knot and at uniform radial refinement edges. The nominal
and doubled radial subdivision counts are 256 and 512. The immutable passing
dossier is
`results/validation/crossdomain-recovery-s1-20260907-r5/`.

| r5 refinement axis | KGAS066 absolute Delta chi-square / relative L2 | KGAS007 absolute Delta chi-square / relative L2 | Status |
|---|---:|---:|---|
| Spatial grid | 0.07771 / 6.17e-5 | 0.002735 / 5.35e-5 | pass |
| Radial quadrature | 1.37e-4 / 5.55e-7 | 3.94e-6 / 1.01e-7 | pass |
| Azimuth quadrature | 1.98e-5 / 1.34e-6 | 3.68e-6 / 1.39e-7 | pass |
| Analytic spectral subdivision | 0.0 / 8.66e-15 | 0.0 / 2.37e-14 | pass |

All flux, independent-phase, signed-coordinate, and spectral-centroid gates
also pass. S1 is **CLOSED** and S2 began on 2026-09-07.

### Briefing note for Astra

The earlier spatial, azimuthal, and spectral refinements closed at relative
visibility scales from approximately `1e-5` to `1e-15`. The r4 radial residual
was caused by the steep turnover and piecewise-linear brightness knots and was
already below `0.13%`, negligible against the project-authorized `2--10%`
telescope calibration scale and the measured thermal noise. The lightweight
composite rule reduced the radial residual to `5.55e-7` for KGAS066 and
`1.01e-7` for KGAS007, so S1 closed under its original stricter thresholds;
the authorized pragmatic relaxation was not needed. This closure protects
engineering velocity without weakening physical fidelity. The additive
rationale record is
`results/validation/crossdomain-recovery-s1-closure-20260907/metrics.json`.

S2 readiness evidence is
`results/validation/crossdomain-recovery-s2-readiness-20260907/metrics.json`.
The missing grouping variables make the required C0/C1 covariance comparison
and held-out scores non-identifiable. Inferring them from row order or uv
coordinates would fabricate independence, so the autonomous cascade stops at
an authorized S2 identifiability boundary.

## Corrected prospective rotation accounting

| Quantity | Definition | Threshold/status |
|---|---|---|
| `chi2_blank` | Data against zero complex visibilities | Emission-detection diagnostic only |
| `chi2_nonrot` | Best-fit emitting disk with circular speed fixed to zero; flux, systemic velocity, dispersion, and center refitted | Required comparator |
| `chi2_rot` | Best-fit rotating visibility model, prior excluded | Required comparator |
| `delta_chi2_blank` | `chi2_blank - chi2_rot` | No rotation claim |
| `delta_chi2_nonrot` | `chi2_nonrot - chi2_rot` | At least 25 prospectively |
| Null bootstrap | `(k+1)/(M+1)` after complete refits with actual covariance | At least 199 trials and `p <= 0.01` |
| Stage B `max_omega_dimensionless` | `max(abs(Delta2(V_ring - V_reference)) / abs(Delta v_chan))` | Blocked until target/regime-specific mock calibration is registered |

The S0 MAP-only accounting check used the audited Ico template noise from the
official propagated error map, unchanged target physical parameters and
optimizer budgets, the version-2 gate schema, and both PA starts. These values
are prospective diagnostics,
not promoted rotation detections:

| Target | `chi2_blank` | fitted `chi2_nonrot` | refitted `chi2_rot` | `delta_chi2_nonrot` | Status |
|---|---:|---:|---:|---:|---|
| KGAS066 | 204228.248 | 200023.444 | 168526.073 | 31497.371 | Delta gate passes; bootstrap pending |
| KGAS007 | 128282.392 | 126567.540 | 122144.496 | 4423.045 | Delta gate passes; bootstrap pending |

Both non-rotating fits reached the allowed 50 km/s dispersion ceiling;
KGAS066 also reached the lower systemic-velocity bound and the +2 arcsec
declination-offset bound. This boundary pressure is retained as diagnostic
evidence and does not authorize wider
bounds. The durable run record and reproducibility manifest are in
`results/validation/crossdomain-recovery-s0-20260906/`.

## Current products

| Target | Product | Status |
|---|---|---|
| KGAS066 | `results/production/KGAS066/kinuv-KGAS066-4c1bc4-milestone1/` | Sealed historical engineering baseline; reported delta chi2 is versus blank complex signal and intervals are conditional/uncalibrated |
| KGAS007 | `results/production/KGAS007/kinuv-KGAS007-e1ee1a-milestone1/` | Sealed historical engineering baseline; delta chi2 6211.629 is versus blank complex signal, not non-rotation; intervals are conditional/uncalibrated |

The authoritative artifact index is `/arc/projects/KILOGAS/analysis/toby_sandbox/results/MANIFEST.md`.

# Architecture mailbox

Closed review and experiment history is synthesized in [`../PRODUCTION_RECORD.md`](../PRODUCTION_RECORD.md). Add only current state here; completed cards should be incorporated into the production record and removed from `docs/reviews/` after closure.
