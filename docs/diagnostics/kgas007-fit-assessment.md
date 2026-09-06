# KGAS007 MILESTONE-001 fit assessment

Date: 2026-09-06

## Verdict

KGAS007 is a secure visibility-domain rotating-disc detection and a reproducible
Stage A optimization result. It is not yet evidence that kinUV outperforms KinMS
for this target. On the official restored image cube, KinMS reproduces the data
substantially better. The comparison is informative but not an equal-likelihood
contest: kinUV was optimized against complex visibilities, while KinMS was
optimized against the same image cube used to score this diagnostic.

The main limitation is model adequacy. KGAS007 is compact, relatively face-on,
and spectrally asymmetric. The frozen surface-brightness template plus a fixed
inclination and one arctan rotation curve recovers the global gradient, but it
cannot reproduce the channel-dependent clumps or the observed radial velocity
shape. The seven-ring alternative extends well outside the detected CO emission
and is unregularized; its nominal chi-square improvement is therefore achieved by
an unstable, physically unsupported rotation curve.

## Quantitative evidence

| Diagnostic | kinUV | KinMS | Assessment |
|---|---:|---:|---|
| Visibility chi-square | 122070.763 | not evaluated | kinUV-only objective; exact serialized-model recomputation error is zero |
| Delta chi-square versus zero rotation | 6211.629 | not evaluated | strong evidence for ordered kinematics |
| Masked cube RMS | 0.08053 K | 0.05129 K | KinMS is 36% lower |
| Masked cube normalized RMSE | 0.67057 | 0.42714 | KinMS is 36% lower |
| Masked voxel correlation | 0.619 | 0.832 | KinMS follows image structure better |
| Aperture-spectrum RMSE | 4.508 K | 2.632 K | KinMS is 42% lower |
| Aperture-spectrum correlation | 0.882 | 0.959 | KinMS follows the asymmetric line profile better |
| Major-axis PVD RMSE | 0.02984 K | 0.02026 K | KinMS is 32% lower |
| Major-axis PVD correlation | 0.842 | 0.925 | KinMS follows the resolved gradient better |
| Model/data masked flux ratio | 0.963 | 1.002 | both are close; KinMS is closer |

There are 956 binned visibility rows and 66 fitted channels. Stage A reduces the
zero-rotation chi-square by 4.84%. The selected chi-square is 0.967 per real or
imaginary datum after the campaign weight scaling, so the residual level is
consistent with the adopted noise normalization. Both PA starts converge to
PA=151.60 degrees, differing by only 0.0016 degrees and 0.0002 in chi-square.
KinMS independently finds PA=152.72 degrees. This agreement is strong evidence
that kinUV has identified the projected kinematic axis.

## Where the fit succeeds

1. **Detection and numerical reproducibility.** The visibility likelihood strongly
   rejects zero rotation, both 180-degree-separated starts find the same solution,
   and the saved model reproduces the reported chi-square exactly.
2. **Global geometry.** The kinUV and KinMS position angles agree to 1.12 degrees.
   The accepted center displacement is small, and no monotonic residual trend is
   detected with baseline length or fitted visibility channel.
3. **Gross velocity field.** The model reproduces the orientation and overall
   S-shaped major-axis PVD. The retained posterior chains have maximum R-hat
   1.0021 and minimum ESS 1093, although their intervals have not passed coverage
   calibration and should not be presented as calibrated credible intervals.

## Where the fit fails

1. **Image-plane line profile.** The data spectrum is clumpy and asymmetric, with
   its strongest sampled peak near 14133.5 km/s. The kinUV spectrum is smoother
   and peaks near 14275.0 km/s. Its largest channel RMS errors occur around
   14100.8, 14264--14275, and 14307.7 km/s. This is not a radio/optical-axis
   conversion error: the aperture centroids are 14210.14 km/s for the data,
   14203.59 for kinUV, and 14203.31 for KinMS.
2. **Intermediate radii.** Between 2 and 3 arcsec, the kinUV cube RMS is 0.1003 K,
   compared with 0.0548 K for KinMS. Between 1 and 2 arcsec the values are 0.0890
   and 0.0554 K. These annuli contain the clearest morphology and rotation-shape
   mismatch.
3. **Rotation-profile shape.** At radii 1.42, 2.42, and 4.08 arcsec, the image
   moment profile is 79.4, 80.6, and 123.1 km/s. kinUV gives 55.6, 95.1, and
   92.2 km/s; KinMS gives 80.1, 77.7, and 124.0 km/s. The accepted arctan model
   undershoots the inner and outer points while overshooting the middle point.
4. **Stage A boundary pressure.** kinUV fixes inclination at 28.9 degrees and its
   turnover radius reaches the configured 0.5 arcsec lower bound. KinMS instead
   fits inclination 25.26 degrees, V0=285.3 km/s, and r_t=1.095 arcsec. Because
   KGAS007 is close to face-on, circular speed and inclination are strongly
   covariant; fixing inclination turns a modest geometry error into a large
   deprojected-speed error.
5. **Stage B identifiability.** The detected moment-0 extent is about 4.19 arcsec,
   but the seven knots span 0.65--7.5 arcsec with zero regularization. The 6.36
   arcsec knot reaches 0 km/s and the curve has max_omega=76.33 km/s. Stage B
   lowers chi-square by 24.26 and formally improves AIC, but much of its outer
   freedom has little surface-brightness support. Rejecting it was the correct
   decision.

## Why KGAS007 is harder than KGAS066

KGAS007 has only 16.5 Jy in the fitted visibility model versus 70.5 Jy for
KGAS066. Its visibility improvement over zero rotation is smaller by a factor of
5.7. The fixed inclination is also only 28.9 degrees, so the line-of-sight
rotation signal is suppressed by sin(i)=0.483, compared with 0.693 at the KGAS066
inclination. The source is only a few synthesized beams across and contains
channel-dependent structure that a single shifted, frozen integrated-intensity
template cannot represent. These effects create a strong V_c--inclination--surface
brightness degeneracy even though the global rotation axis is detectable.

KGAS007 additionally uses the historical reference-wavelength visibility export,
which lacks the complete time/baseline metadata of the `ms2kinuv-npz-v1`
contract. The compatibility reader is validated, and the flat residual-versus-uv
diagnostic gives no evidence that this caused the fit mismatch, but the missing
provenance prevents the strongest possible data-contract audit.

## Required comparison before claiming an improvement over KinMS

Fourier-sample the fitted KinMS sky cube at the observed uv coordinates, apply
the identical spectral response and weights, optimize only any explicitly shared
nuisance normalization, and score it with the kinUV visibility chi-square. The
present image metric cannot establish a visibility-domain winner because KinMS
was optimized in that image domain and the restored cube has correlated noise.

The next MAP-level experiments should use target-adaptive ring support ending at
the detected CO radius, a regularized four- or five-knot profile, an inclination
profile likelihood under an external morphological prior, and a surface-brightness
model that can represent asymmetric channel structure. KGAS007 should also be
re-exported from its Measurement Set through `ms2kinuv` when that source becomes
available. Posterior sampling should wait until these model-adequacy tests improve
held-out visibility residuals.
