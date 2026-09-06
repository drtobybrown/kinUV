# kinUV production record

## 2026-09-06 scientific-gate audit supersession

MILESTONE-001 numerical artifacts remain sealed. Per
[`DEC-KINUV-CROSSDOMAIN-RECOVERY`](decisions/DEC-KINUV-CROSSDOMAIN-RECOVERY.md),
earlier rotation-null and omega interpretations are withdrawn pending S0. Neither
real target has an established truth-recovery ranking. Archived evidence is
preserved unchanged.

This document is the durable synthesis of completed review cards, architecture notes, and superseded diagnostic artifacts through 2026-09-06. It records conclusions needed to interpret or reproduce the current KGAS066 and KGAS007 products. Active coordination belongs in [`reviews/BOARD.md`](reviews/BOARD.md); binding model choices remain in [`decisions/`](decisions/).

## Current products

| Target | Product | Result | Interpretation |
|---|---|---|---|
| KGAS066 | MILESTONE-001, `results/production/KGAS066/kinuv-KGAS066-4c1bc4-milestone1/` | Selected Stage B chi2 167302.366; Stage A PA 199.729 deg, V0 267.670 km/s, rt 0.500 arcsec, delta chi2 versus V=0 35552.645; retained posterior max Rhat 1.00369 and min ESS 889 | Accepted immutable baseline. Stage B passes AIC, bound-pressure, and oscillation gates. Posterior intervals remain uncalibrated and the structured visibility residual flag remains set. |
| KGAS007 | MILESTONE-001, `results/production/KGAS007/kinuv-KGAS007-e1ee1a-milestone1/` | Selected Stage A chi2 122070.763; PA 151.601 deg, V0 195.979 km/s, rt 0.500 arcsec, delta chi2 versus blank visibilities 6211.629; retained posterior max Rhat 1.00214 and min ESS 1093 | Accepted immutable baseline. Stage B is retained but rejected because a ring reached 0 km/s. Historical `max_omega=76.329` is dimensionless; its old km/s threshold label was invalid. Posterior intervals remain uncalibrated. |

The production target set is KGAS066 plus KGAS007. G4 and population inference have not been authorized.

The independent KGAS007 fit assessment is recorded in
[`diagnostics/kgas007-fit-assessment.md`](diagnostics/kgas007-fit-assessment.md).
It finds a robust visibility-domain rotation detection and PA, but no current
evidence that kinUV outperforms KinMS for this target. The dominant mismatch is
the fixed-geometry, frozen-brightness model at 1--3 arcsec and in asymmetric
spectral channels; an equal visibility-likelihood comparison has not yet been run.

## Production method and invariants

- The likelihood is evaluated on complex ALMA visibilities: `chi2 = s * sum(w * |data - model|^2)`. A complex visibility contributes two real degrees of freedom.
- The fit arrays are 881 by 95 for KGAS066 (`N=4`, `dv=5.080 km/s`, `s=0.513610`) and 956 by 66 for KGAS007 (`N=4`, `dv=5.080 km/s`, `s=0.570735`).
- Native channels are Hann-smoothed with guard channels before binning. The production Fourier convention uses `NPZ_UV_SIGN=-1` with the negative-exponential kernel.
- Surface brightness is the 30 km/s Wiener-deconvolved Ico map, transformed so positive x is east. The primary beam is applied in the image plane after the positional shift.
- Stage A uses an arctangent rotation curve. Stage B fits seven ring velocities with Stage A geometry fixed and is selected only when AIC, bound-pressure, and oscillation gates pass.
- The unconstrained NUTS chart logs flux, gas dispersion, and rt, uses a stable softplus for V0, and leaves PA and systemic velocity on identity coordinates. Position offsets are frozen at the MAP during NUTS.
- Production sampling uses CPU JAX/FINUFFT and NumPyro in headless CANFAR jobs. Four independent one-chain jobs may be merged after per-parameter Rhat and ESS checks.

## Validation and benchmark evidence

The transform tests compare the type-2 NUFFT with analytic Gaussian and thin-ring cases. The S1 inject-and-recover experiment used the real KGAS066 uv coverage with truth `rt=0.25 arcsec`, `V0=250 km/s`, and `gas_sigma=8 km/s`. The visibility fit recovered `rt=0.254 arcsec` and the injected dispersion. CLEAN-beam moment estimates did not recover the same sub-beam structure: the M1 inner slope was about 94.7 rather than 236.7 km/s/arcsec, and M2 was about 56 rather than 8 km/s. This is a controlled-mock result, not permission to quote a real-galaxy inner slope.

S2 used independence Metropolis with a Laplace proposal. It mixed, but 20 exact-model noise realizations failed the binomial 68/95 percent coverage checks. Consequently, neither S2 Laplace intervals nor current real-data 16/50/84 percent intervals are calibrated science products. KGAS066 has `chi2/(2*n_vis) = 1.0077`, supporting the global visibility weight scale; its residual structure is stronger with velocity than baseline and points to surface-brightness mismatch.

The JAX likelihood reproduced the official KGAS066 MAP chi2 and ran at 3.01 evaluations/s on CPU versus 0.329 evaluations/s for the earlier finite-difference path. An H100 MIG CUDA trial reproduced the chi2 but achieved only 0.55 evaluations/s and a six-axis gradient in 2.80 s versus 0.43 s on CPU. GPU NUTS was therefore retired for this problem size.

The retained S3 comparison uses KinMS only as an image-plane comparator. Its corrected wrapper passes face-on disk coordinates to KinMS, lets KinMS project inclination and PA, writes cubes with transpose `(2, 1, 0)`, and applies systemic velocity through `vOffset`. The best real-data KinMS comparator found PA 198.70 deg, V0 266.92 km/s, rt 0.465 arcsec, inclination 48.41 deg, and gas dispersion 12.83 km/s. These cube-fit parameters do not replace the visibility likelihood or official kinUV parameters. On the controlled mock, KinMS returned `rt=0.395 arcsec` for truth 0.25 arcsec, while kinUV returned 0.253 arcsec.

The 2026-09-06 canonical downstream runner extended the same comparison contract to KGAS007 and regenerated a homogeneous diagnostic suite for both targets. The independent KGAS007 KinMS cube fit completed after 475 evaluations with PA 152.72 deg, V0 285.25 km/s, rt 1.095 arcsec, inclination 25.26 deg, systemic optical velocity 14203.71 km/s, and gas dispersion 15.56 km/s. MILESTONE-001 embeds the comparator in each accepted bundle. KinMS has lower residual RMS on the masked CLEAN cube for both targets, as expected for a model optimized in that image domain. That metric is not a visibility likelihood comparison. The retained controlled KGAS066 mock remains the evidence that kinUV recovers injected sub-beam kinematics more accurately than the restored-cube fit.

MILESTONE-001 fixed an inclination propagation defect in Stage B and model-cube serialization. The previous KGAS007 Stage A likelihood used 28.9 degrees correctly, but its exported cube inherited the KGAS066 default inclination. The milestone reran the fit and now propagates the configured target inclination through likelihood evaluation, Stage B, native cube creation, matched imaging products, and the KinMS comparison. Both products were generated from clean commits, reproduce their selected chi2 exactly, and contain complete SHA-256 manifests.

## Closed alternatives

- The pre-sign PA 21.9 deg solution and the 2026-08-27 image products are superseded. Correcting the stored uv sign yielded the official PA 199.73 deg solution.
- The PA 25.2 deg NUTS attempt failed mixing: chains occupied inconsistent regions, one chain diverged catastrophically, and PA Rhat was about 22. A recovery MAP returned to PA 199.728 deg and matched the official chi2 within 0.008. The approaching branch is closed.
- A 10 km/s Ico template gave conditional chi2 168701.213 versus 168675.596 for the 30 km/s template. Imaging weight, beam, cell, mask, and CLEAN-threshold differences confound that comparison. The 30 km/s Ico input remains locked.
- An optional axisymmetric-plus-m=2 surface-brightness experiment improved its axisymmetric analytic baseline by 99.5 in chi2 but remained 2399 worse than the production two-dimensional Ico template. It is not part of the production model.
- Exploratory dirty-adjoint, radial-gradient, PV-angle, parameter-slice, and SKA yield figures were publication experiments. The dirty residual used a script-local type-1 DFT adjoint, parameter contours were chi2 slices rather than posterior samples, and the SKA calculation was an analytic volume integral rather than a visibility simulation. None changes the KGAS066 or KGAS007 products.
- The early Gate 4 campaign and 2026-08-30 final-fit bundle were superseded by the current Stage A, Stage B, NUTS, and three-way comparison artifacts.

## Retained production evidence

| Path | Purpose |
|---|---|
| `docs/reviews/artifacts/2026-08-28-stage-b-imaging/` | Correct-sign Stage B versus 10 km/s image-plane check. |
| `docs/reviews/artifacts/2026-08-29-s1-mock/` | Real-uv injection and recovery evidence. |
| `docs/reviews/artifacts/2026-08-29-s2/` | Coverage experiment and interval-calibration failure. |
| `docs/reviews/artifacts/2026-08-30-g3-nuts/` | KGAS066 NUTS posterior and mixing diagnostics. |
| `docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes/` | Canonical Stage A, NUTS-mean, and Stage B residual/image comparisons. |
| `docs/reviews/artifacts/2026-09-05-kgas007-stage-a-map/` | KGAS007 MAP initialization and diagnostics. |
| `docs/reviews/artifacts/2026-09-05-kgas007-nuts/` | Merged KGAS007 NUTS product and mixing diagnostics. |
| `docs/reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark/` | KinMS live and controlled-mock comparator, excluding retired exploratory figures. |
| `MILESTONE-001.md` | Compact milestone receipt and acceptance summary. |
| `../../results/production/KGAS066/kinuv-KGAS066-4c1bc4-milestone1/` | Accepted KGAS066 visibility, imaging, benchmark, and posterior bundle. |
| `../../results/production/KGAS007/kinuv-KGAS007-e1ee1a-milestone1/` | Accepted KGAS007 visibility, imaging, benchmark, and posterior bundle. |

The compressed source bundle for pruned history is [`archives/kinuv_docs_legacy_20260906.tar.gz`](../../archives/kinuv_docs_legacy_20260906.tar.gz). It contains the closed review cards and removed artifacts exactly as they existed before cleanup.

## Remaining work

1. Calibrate the exact NUTS workflow with simulation-based calibration before publishing credible intervals.
2. Re-export the historical KGAS007 wavelength-coordinate NPZ with `ms2kinuv` when its source Measurement Set becomes available.
3. Investigate the KGAS066 velocity-structured residual and define an outer-ring support criterion before another Stage B campaign.
4. Define a target-selection contract before adding a multi-galaxy runner or hierarchical model.
5. Test richer surface-brightness models only when they preserve the visibility-plane likelihood and have an explicit comparison criterion.
