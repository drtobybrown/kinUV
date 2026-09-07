---
generation: 33
phase: crossdomain-recovery-s5-closed
code_freeze: true
next_role: consultant-astra
board: production-layout-and-diagnostics
build_licensed: false
pending:
  - nonrotation-null-bootstrap
  - stage-b-smoothness-recalibration
  - exact-workflow-posterior-calibration
last_propose: human-directive-production-cleanup-20260907
last_review: docs/reviews/2026-09-07-production-layout-qa.md
last_review_a: docs/reviews/2026-09-07-astra-production-figure-completeness.md
last_review_b: docs/reviews/2026-09-07-code-review-b-crossdomain-s4-recovery.md
user_review: docs/reviews/artifacts/2026-09-05-kgas007-nuts/
open_questions:
  - How broadly does the demonstrated advantage extend beyond the tested thin axisymmetric arctan family?
deadlocks: []
canon_generation: 33
---

## Agent Run Status

* **Phase:** S0--S5 are closed under the PI-authorized cross-domain recovery contract. The accepted S4 evidence commit is `354bbad`; S5 sealed it at `98f7730`.
* **Last Action:** Commit `e3a54fcdd2d61e9bf283374cd4b35043c9921fb7` replaced the run-ID layout with canonical `best_model/`, `plots/`, and `benchmarks/` directories and expanded the modern diagnostic suite. No fit, sampler, or scientific parameter changed.
* **Decisions Made:** Standard practice is now the binding comparison: stock KinMS consumes the canonical full-data pipeline cube, while kinUV consumes and predicts calibrated visibilities. Training-fold `tclean` products are not required. Synthetic truth uses an analytic Python cube and native Fourier visibilities with no CASA dependency.
* **Gates:** Every grouped visibility fold favors kinUV. Aggregate lower 95 percent gains are +0.03178592 chi-square/component for KGAS066 and +0.00290304 for KGAS007. Synthetic projected-velocity RMSE ratios are 0.01180 and 0.04569, respectively, against the required maximum 0.90. KGAS066 therefore satisfies the non-regression guard.
* **Verification:** Each target manifest verifies 43 files, including 20 modern PDF/PNG figures. KGAS066 and KGAS007 manifest SHA-256 values are `9a0fa7f569fa526dd4d23f8e6f4c9aaad6fbb21aafd204ca32534fc3557a285b` and `a96e9927c9b201ad8bd2949f5141ee262807c75a9e00d17ef2eb9e56fd8a4951`. The generator reproduced the hierarchy from standardized inputs at the exact implementation commit. The deterministic suite passes 261 tests with 5 skips.
* **Next Step:** Astra may define the next scientific expansion. Posterior calibration, the non-rotation bootstrap, and broader synthetic families remain separate future campaigns.

## Production layout and diagnostic closure

The only active target-level directories are `best_model/`, `plots/`, and
`benchmarks/`. `best_model/` retains selected parameters, native and matched
model cubes, the posterior checkpoint, and a local manifest. `plots/` contains
five PDF/PNG pairs for moments, major/minor PVDs, rotation, aperture-integrated
spectra, and sampled-parameter covariance. `benchmarks/` contains five
PDF/PNG pairs comparing kinUV with KinMS in moments, major/minor PVDs,
integrated spectra, real rotation profiles, and synthetic sub-beam recovery.

Moment framing is determined from the 5-percent moment-0 support plus one BMAJ,
capped at 12 arcsec. The registered crop is 12.0 arcsec for KGAS066 and
7.5111 arcsec for KGAS007. The posterior plot shows the five sampled primary
dimensions; inclination is stated as fixed because it has no posterior
dimension. Intervals remain explicitly uncalibrated.

The complete pre-layout bundles were archived before removal. KGAS066's
12,814,556-byte archive has SHA-256
`fe7e653a8535a3243163873b378bd22e3f97d847a5c8a216af39a7e1defa7e0a`;
KGAS007's 10,122,422-byte archive has SHA-256
`8bcb012c2891c50afc915cda5dc17469fe35086517b267e848a60b84c7344f99`.
No legacy PNG or PDF remains anywhere under `results/production/`.

## S5 sub-beam turnover diagnostic

Both accepted synthetic truths meet the declared `R_turn < BMAJ` condition.
For KGAS066, `R_turn/BMAJ=0.28846`; kinUV's mean absolute turnover error is
0.000996 arcsec versus 0.039977 arcsec for KinMS, a ratio of 0.02493. The
inner-beam projected-velocity RMSE is 0.13830 versus 9.61877 km/s, a ratio of
0.01438. For KGAS007, `R_turn/BMAJ=0.43853`; turnover errors are 0.020044
versus 0.072452 arcsec, a ratio of 0.27665, and inner-beam RMSE values are
0.63702 versus 4.63336 km/s, a ratio of 0.13749.

The reporter uses only the accepted S4 fitted parameters and stored 24-bin
profile grid. It performs no optimization, changes no target parameter, and
does not modify the dual-approved S4 gate. Per-target inner profiles and the
combined summary figure are retained with the metrics and manifest. This
STATUS generation 31 is the deterministic in-repository handoff for code
commit `9f09bfa`.

## Governance workflow synchronization

The exact governance implementation commit is
`97babbbf6abb7865ea701a11495d22b7c37d8fd0`. It adds autonomous follow-through,
anti-ceremony testing, proportional subagent sizing, and deterministic
in-repository stage handshakes. The affected verification suite passed 34 of
34 tests in 17.75 seconds. No scientific model, likelihood, parameter,
threshold, target configuration, or accepted product changed. The next role is
Consultant Astra; the durable handoff is this STATUS generation 30.

## S4 scientific recovery and S5 seal

The binding PI correction is
[`DEC-PI-S4-STANDARD-USE-BENCHMARK`](../decisions/DEC-PI-S4-STANDARD-USE-BENCHMARK.md).
The final dossier is
`results/validation/crossdomain-recovery-s4-final-20260907/`; its top-level
`metrics.json` combines the exact grouped-prediction and synthetic-truth
records without changing either source dossier. The original grouped summary's
`promotion_eligible: false` field is explicitly superseded because it encoded
the now-rejected CASA reimaging prerequisite.

The first submitted dossier was rejected for reproducibility because resumed
KinMS checkpoints were not authenticated to the stamped commit and inputs.
The revised runner binds every checkpoint to the runner commit, realization
seed, worker hash, mock-cube hash, truth-cube hash, and mask hash. It also
records target configuration, covariance, visibility, diagnostic product, and
runner hashes, and serializes grouped bootstrap seed 4404. The old final
dossier was removed and all six external fits were rerun from an empty output
directory at `f2f6a22`; the scientific values reproduced exactly. Reviewer B's
second pass identified the still-unhashed fit-window cube and the lack of a
behavioral checkpoint test. Commit `baf7047` added both without changing the
science, and the final dossier was regenerated from empty output at stable
commit `354bbad`. Both reviewers then accepted it.

S5 reverified the complete S4 manifest, both final review verdicts, the clean
CASA boundary, and the deterministic suite. Its sealed metrics and manifest
are in `results/validation/crossdomain-recovery-s5-20260907/`. The standard-use
and exact-family claims are accepted; posterior calibration and performance on
unmodeled morpho-kinematic families are not implied.

The repaired replay established that the historical KGAS066 cube score was
inherited by the baseline and was not caused by S3 kinematic flexibility.
Smooth positive emissivity improved the visibility fit slightly but did not
materially change the restored-cube metric. This supports the PI's separation
of promotion evidence from reconstruction-dependent supporting diagnostics.
The old r3 failure remains immutable historical evidence.

## Historical S4 failed-gate evidence

The S4 attempt was **FAILED AND ESCALATED** at exact implementation commit `5d08507`. The
checksum-bound dossier is
`results/validation/crossdomain-recovery-s4-20260907-r3/`; all 23 manifest
entries match their registered sizes and SHA-256 hashes. The complete test
suite passed with 263 tests and 8 skips. The formal record is
`docs/decisions/ESC-KINUV-S4-CROSSDOMAIN-SUPERIORITY.md`.
Reviewer A and Reviewer B independently accepted the dossier as valid
failed-gate evidence at commits `a0b0f0a` and `4882189`, respectively; neither
verdict closes S4 or authorizes S5.

| Target | kinUV reduced chi-square | KinMS reduced chi-square | ratio | kinUV velocity RMSE | KinMS velocity RMSE | ratio | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| KGAS066 | 7.43475 | 6.50510 | 1.14291 | 16.3203 km/s | 12.8486 km/s | 1.27020 | fail both; regression |
| KGAS007 | 2.04657 | 2.27507 | 0.89956 | 2.5049 km/s | 4.0872 km/s | 0.61286 | chi-square pass; velocity ineligible with two bins |

The S3-selected KGAS066 model recovers 97.75 percent of the registered cube
flux, compared with 92.76 percent for KinMS, and materially improves on the
older milestone kinUV cube. Its common-domain residual RMS and projected
velocity profile nevertheless remain worse than KinMS, so flux conservation
alone cannot establish superiority. KGAS007 improves both the common cube RMS
and reduced chi-square; the restored cube does not provide the three common
moment-1 radial bins required for an eligible velocity claim.

Structured native line-free residual diagnostics are present for folds,
baselines, antennas, and real/imaginary components. Global component-mean
z-scores are within 0.13 in absolute value and global real/imaginary
correlations are below `2.4e-4`; localized baseline summaries remain available
for diagnosis. These diagnostics do not override the failed promoted metrics.

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
| KGAS066 | `results/production/KGAS066/best_model/` | Sealed historical engineering baseline; reported delta chi2 is versus blank complex signal and intervals are conditional/uncalibrated |
| KGAS007 | `results/production/KGAS007/best_model/` | Sealed historical engineering baseline; delta chi2 6211.629 is versus blank complex signal, not non-rotation; intervals are conditional/uncalibrated |

The authoritative artifact index is `/arc/projects/KILOGAS/analysis/toby_sandbox/results/MANIFEST.md`.

# Architecture mailbox

Closed review and experiment history is synthesized in [`../PRODUCTION_RECORD.md`](../PRODUCTION_RECORD.md). Add only current state here; completed cards should be incorporated into the production record and removed from `docs/reviews/` after closure.
