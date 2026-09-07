---
role: reviewer
seat: science-numerics
phase: implementation
date: 2026-09-07
reviewer: review-a-s3
canon_generation: 23
campaign_id: crossdomain-recovery-s3
proposal: docs/decisions/DEC-KINUV-CROSSDOMAIN-RECOVERY.md
reviewed_commit: b2ac6bc44991766ab8c790de1f351b812df0db45
verdict: accept
---
# Independent code review A: cross-domain recovery S3

I reviewed exact clean kinUV commit
`b2ac6bc44991766ab8c790de1f351b812df0db45` independently against the
binding Project-PI right-sized gates. I audited the sealed dossier at
`results/validation/crossdomain-recovery-s3-20260907-r3`, recomputed its
manifest and input identities, checked the likelihood and penalty accounting,
and attempted to falsify the replay, positivity, sequential-selection,
identifiability, optimizer, and hot-path boundary claims. This verdict closes
only the S3 MAP ablation gate. It does not license posterior calibration,
rotation significance, cross-domain superiority, or production promotion.

## Findings

1. **Artifact and S2 replay integrity — pass.** Every file listed in the S3
   manifest matches its recorded byte count and SHA-256 digest, and the
   manifest identifies the exact reviewed commit with a clean `dev` tree.
   Both visibility inputs, Ico templates and errors, fit-window cubes, target
   configurations, and S2 summaries still match the hashes embedded in each
   target record. The recomputed S2 likelihood errors are
   `6.112e-10` for KGAS066 and `5.821e-11` for KGAS007, against the `0.1`
   gate. The expected chi-square values also match the accepted S2 summaries
   exactly: `166697.915358353` and `105467.02659189644`.

2. **Positive emissivity and flux accounting — pass.** The three radial
   basis images are constructed from the non-negative template, each is
   normalized to unit angular integral, and their natural weights reconstruct
   the clipped unit-integral template. A softmax enforces positive weights
   summing to one for every fitted candidate, while one global line-flux
   parameter carries the physical normalization. In the sealed preferred
   fits the minimum component weights are `0.30298` for KGAS066 and `0.30439`
   for KGAS007; both weight vectors sum to one at floating-point precision.
   The forward operator retains integrated line flux in Jy km/s and returns
   channel-average Jy without surviving-support renormalization.

3. **Sequential one-factor selection — pass.** The candidate sequence changes
   one physical component at a time while retaining joint optimization of
   flux, center, PA, systemic velocity, inclination, and the active kinematic
   and dispersion parameters. Emissivity gains are `223.1933` for KGAS066 and
   `12.69224` for KGAS007, both above the two-parameter AIC cost of four. The
   projected-knot gains are `4.82079` and `32.87504`; the registered
   `delta chi2 >= 10` rule therefore rejects rings for KGAS066 and retains
   them for KGAS007. Crucially, the dispersion test branches from the retained
   `joint_emissivity` parent for KGAS066 and the retained `supported_rings`
   parent for KGAS007. This repairs the earlier non-sequential comparison.

4. **Projected-knot support and identifiability — pass.** The four knots use
   the declared projected speed `u=v_c sin(i)`, begin at `0.5 BMAJ`, and end
   at the positive-emission `R95`. Their radii are
   `[0.5200, 4.9575, 7.4480, 11.6927] arcsec` for KGAS066 and
   `[0.6271, 2.3841, 3.5415, 5.3670] arcsec` for KGAS007. All separations
   exceed `0.2 BMAJ`. Both four-knot local Hessian blocks have rank four and
   no knot is on a bound. Their identified-subspace condition numbers are
   `29.79` and `14.29`. I bounded the curvature-penalty contribution to the
   per-complex knot Hessian by `0.00430` and `0.00627`; these are at most
   `2.1%` and `11.6%` of the respective smallest reported knot singular
   values. The data likelihood therefore supplies the retained rank rather
   than the regularizer alone. KGAS007's non-monotonic middle knots remain
   visible in the record and are permitted by the decision's prohibition on
   artificial monotonic coupling.

5. **Dispersion selection — pass.** The smooth two-zone model adds one
   positive outer-dispersion parameter only after brightness and the retained
   velocity family are selected. KGAS066 improves chi-square by `152.5002`
   relative to joint emissivity and selects interior values
   `sigma_inner=7.473 km/s`, `sigma_outer=9.878 km/s`. KGAS007 changes
   chi-square by `-0.000384` relative to supported rings, converges back to
   `11.212/11.211 km/s`, and correctly rejects the extra degree of freedom.

6. **Optimizer, boundaries, and accounting — pass.** All eight target/candidate
   fits report successful finite optimization and no active parameter on a
   bound. Per-complex projected-gradient infinity norms span
   `6.031e-5` to `2.367e-4` for KGAS066 and `7.003e-5` to `8.317e-5` for
   KGAS007, comfortably below `1e-3`. I reproduced each normalized gradient
   from its raw value and exact retained complex count (`83,695` and `60,192`),
   and reproduced every saved chi-square from objective, prior, and
   regularization terms. The preferred full local Hessians are full rank:
   `12/12` with condition `820.8` for KGAS066 and `13/13` with condition
   `368.2` for KGAS007.

7. **Physical and architectural invariance — pass.** The S3 records retain the
   accepted C1 covariance values and unchanged visibility selections from S2.
   The default forward path is unchanged when optional velocity or dispersion
   profiles are absent, which the near-machine-precision S2 replay confirms.
   A source and import scan of the production hot path finds no NFW, Burkert,
   dark-matter, halo-mass, baryonic-decomposition, `uvkin`, or `uvfit`
   dependency. S3 uses only angular radii, projected velocities, positive gas
   dispersion, emissivity weights, geometry, and visibility-domain chi-square.

## Claim boundaries and residual risks

The supported-ring candidate replaces the two-parameter arctan function with
a four-knot piecewise profile; it is a prespecified one-component model-family
comparison, not a formal nested likelihood-ratio significance test. The
registered ten-point chi-square threshold is explicitly a retention heuristic,
so this does not invalidate the S3 selection. S4 must judge predictive truth
recovery on preregistered outer folds and mocks rather than interpret these
training gains as evidence of superiority.

The Hessians are local identifiability diagnostics, not calibrated posterior
covariances. Inclination remains externally uncalibrated, so only projected
speed is eligible for promotion. S4 must also carry forward the structured
baseline, antenna, time, and real/imaginary residual diagnostics required by
the S2 review.

## Verdict

**Accept.** The sealed S3 evidence reproduces S2, conserves a positive
unit-integral emissivity model, applies the model-complexity sequence to the
correct retained parent, and promotes only interior, empirically converged,
locally identifiable parameters. The target-dependent selections are
scientifically coherent: KGAS066 needs brightness and radial dispersion
flexibility but lacks support for four velocity knots, while KGAS007 supports
projected-profile flexibility and provides no evidence for radial dispersion
complexity.
