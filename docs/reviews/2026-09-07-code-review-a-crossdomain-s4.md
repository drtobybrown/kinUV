---
role: reviewer
seat: science-numerics
phase: implementation
date: 2026-09-07
reviewer: review-a-s4
canon_generation: 24
campaign_id: crossdomain-recovery-s4
proposal: docs/decisions/DEC-HUMAN-OVERRIDE-RIGHTSIZED-GATES.md
reviewed_commit: 5d08507d2b9fa048a7760d57eedd7cbff722238c
verdict: accept-failed-gate-evidence
---
# Independent code review A: failed cross-domain recovery S4 gate

I independently reviewed exact clean kinUV commit
`5d08507d2b9fa048a7760d57eedd7cbff722238c` and the sealed evidence at
`results/validation/crossdomain-recovery-s4-20260907-r3`. I accept the dossier
as valid evidence that the present S3 models **fail** the binding Project-PI S4
gate. This verdict does not accept S4, license S5, or support a kinUV
superiority claim.

## Scientific audit

1. **Common scoring domain — valid for this gate.** Both models are compared
   with the same official 10 km/s KILOGAS cube, three-dimensional source mask,
   celestial/spectral grid, restoring beam, brightness-temperature units, and
   scalar channel-noise normalization. The kinUV intrinsic model is converted
   from Jy/pixel, primary-beam corrected, overlap-rebinned spectrally, restored,
   regridded, and converted to kelvin before scoring. The frozen stock-KinMS
   cube is converted to the same kelvin representation. Neither model receives
   an amplitude refit during scoring. Visual inspection of the moment products
   shows common registration and no axis-reflection or gross frame-offset
   failure.

2. **Frame and noise handling — sufficient for the relative result.** The
   likelihood remains in native TOPO frequency. Diagnostic rendering applies
   the registered frequency-equivalent corrections (`+10.9851 km/s` for
   KGAS066 and `-16.7875 km/s` for KGAS007) before conversion to the official
   LSRK optical axis. The corresponding within-observation drifts remain only
   `0.0371` and `0.0640 km/s`, negligible relative to the 10 km/s diagnostic
   channels. Channel RMS values (`0.03963 K` and `0.03403 K`) are derived from
   the official integrated-error maps and applied identically to both models.
   Spatial and spectral covariance prevent interpreting the absolute reduced
   chi-square values as calibrated goodness-of-fit probabilities, but the
   shared mask and scalar normalization preserve the model ordering used here.

3. **Reduced chi-square calculation — reproduced.** KGAS066 uses 21,497 common
   voxels, 12 active kinUV parameters and 7 KinMS parameters. I reproduce
   `chi2_red=7.43475` for kinUV and `6.50510` for KinMS, a kinUV/KinMS ratio of
   `1.14291`; kinUV therefore regresses by 14.3%. KGAS007 uses 4,461 voxels,
   13 and 7 parameters, yielding `2.04657` and `2.27507`, a ratio of
   `0.89956`; kinUV improves that target by 10.0%. The same ordering appears
   in raw residual RMS, so it is not created by the small degrees-of-freedom
   difference. Flux ratios are also retained rather than renormalized:
   kinUV/KinMS recover `0.9775/0.9276` of masked KGAS066 flux and
   `1.0331/1.0015` for KGAS007.

4. **Projected-velocity metric — correctly restricted.** Both model profiles
   and the data proxy are extracted with the same PA, center, systemic
   velocity, inclination convention, beam, mask, and moment calculation. The
   final residual is multiplied by `sin(i)`, so the promoted comparison is in
   projected speed rather than an externally unsupported intrinsic circular
   speed. KGAS066 has 15 common radial bins and is eligible: kinUV RMSE is
   `16.3203 km/s` versus `12.8486 km/s` for KinMS, giving ratio `1.27020`
   instead of the required `<=0.90`. KGAS007 has only two common bins. Its
   diagnostic ratio of `0.61286` is therefore correctly marked ineligible and
   cannot rescue the gate.

5. **Provenance — pass.** All 23 manifest entries reproduce their byte counts
   and SHA-256 digests. The target records bind the exact S3 selections,
   configurations, official cubes, masks, error maps, KinMS cubes, and KinMS
   fit records. The top-level summary records the exact reviewed commit and a
   clean `dev` state.

## Gate result

The PI requires at least 10% lower eligible projected-velocity RMSE and better
overall reduced chi-square on both real targets. KGAS066 fails both conditions.
KGAS007 passes the cube-residual comparison but lacks enough common radial
support for the velocity gate. The conjunction therefore fails decisively,
and the explicit KGAS066 non-regression condition also fails. A 512-pair mock
campaign cannot promote the present candidate over this real-target failure;
the authorized action is escalation rather than S5 progression.

## Verdict

**Accept failed-gate evidence.** The comparison is sufficiently controlled to
establish that the current S3 candidate does not meet the Project-PI S4
superiority standard. Preserve this dossier as a failed benchmark, halt the
autonomous cascade, and return the KGAS066 regression plus KGAS007 profile
support limitation to Astra for a new physical or model-selection decision.
