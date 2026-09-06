---
id: DEC-KINUV-CROSSDOMAIN-RECOVERY
status: accepted
date: 2026-09-06
authority: Astra
implementation: licensed-s0-s1
proposal_reviews: authority-approved-2026-09-06
---
# Cross-domain truth-recovery architecture

## Decision and evidence boundary

Astra accepted this scientific strategy and licensed S0/S1 implementation on 2026-09-06. Each stage still requires independent Reviewer A science/numerics and Reviewer B software/reproducibility code verdicts before its gate can close. The sealed MILESTONE-001 artifacts remain historical engineering baselines; their accepted state is not retroactively changed, but it does not validate superiority or rotation significance.

kinUV will be judged on prospective ground-truth recovery across declared regimes. Both real targets currently have lower restored-cube NRMSE under KinMS (KGAS066: 0.536 versus 0.818; KGAS007: 0.427 versus 0.671). KGAS066 favors kinUV on one seed-66 exact-family mock (`r_t=0.2527` versus truth `0.25`, KinMS `0.3953`), but kinUV generated its own exact 2-D model while KinMS used radial brightness and free inclination. This is useful unmatched evidence, not a general superiority demonstration.

The KGAS007 `delta chi2=6211.629` is fitted emission versus blank complex visibilities, not zero rotation. Future reports distinguish parameter-free `chi2_blank`, `chi2_nonrot` with equal brightness complexity/nuisances, and `chi2_rot`. Rotation candidacy requires prior-free `delta chi2_nonrot>=25` and bootstrap `p=(k+1)/(M+1)<=0.01` from at least 199 preregistered null realizations with actual covariance and complete refits. Each null realization repeats any data-dependent template, mask, regularization, and initialization selection used on data unless inputs were independently frozen. Wilks/`sqrt(delta chi2)` shortcuts are prohibited: amplitude is bounded and kinematic direction can be unidentified; parameters still affecting morphology remain fitted or integrated.

The current Stage B `max_omega` values (76.33 for KGAS007; 10.18 for KGAS066) are dimensionless outputs of a normalization in `rotation.py`, although configuration labels the threshold in km/s and historical logic used 0.3. This is a blocking scientific-gate/provenance defect. S0 must document the formula with a unit test and recover the actual historical threshold provenance. The historical 0.3 may be retained only in its exact mock-calibrated scope; no universal or relabeled omega threshold is accepted. No new scientific promotion may depend on this gate until a mock-validated smoothness criterion is registered.

| Evidence class | Current items | Consequence |
|---|---|---|
| Proven defects/limitations | null semantics; omega unit/threshold; hard-coded brightness frequency/noise proxy; inverted mock-predicate label; restored comparator; incomplete grouping | Correct in S0/S1 |
| Plausible hypotheses | operator/response, covariance, joint brightness, inclination, radial support, frozen nuisances, basin, then dispersion | Test one factor at a time |
| No current evidence | thickness, warp, radial flow, intrinsic disturbances/clumps | Exclude; require later proposal |

## Fair comparator and covariance contract

Never Fourier sample or deconvolve the saved KinMS FITS cube: the worker `beamSize` has already restored it. Regenerate intrinsic pre-beam KinMS emissivity/cloud deposition at native spectral resolution, then apply the same primary beam, phase convention, channel frequencies, Fourier operator, and spectral response exactly once.

Maintain two labeled paired branches: a matched branch with the same emissivity, geometry, priors, spectral family, cells, and covariance isolates fitting domain; a native branch compares prospectively declared best practice. Both use identical mask rules, PB, WCS, velocity/LSRK convention, native response, selection, and covariance. Common-restoration closure is secondary. Report likelihood and prior terms separately. Covariance comparison uses held-out predictive log likelihood including `logdet(C)`; raw delta chi-square is comparable only on identical cells with fixed covariance.

Training-derived templates, masks, nuisance estimates, regularization, and image products are frozen before scoring. Real validation uses at least five correlation-aware folds with boundaries wider than Hann/bin support. The legacy KGAS007 export lacks time/antenna grouping; if valid folds cannot be constructed, label real holdout and superiority `BLOCKED`, never fabricate independent groups. Propagate `C' = B H C H^T B^T` in the appropriate complex block form and never apply Hann twice. The predicted adjacent-bin correlation is 0.1154, so `chi2/(2n)=0.967` after marginal scale normalization is neither an effective reduced chi-square nor covariance validation.

## Candidate model

- Keep angular kinematics and no dark-matter, cosmology, CASA, or Measurement Set runtime dependency. Retain exact baseline reproduction.
- Fit `u(R)=v_c(R) sin(i)` and profile inclination. Use an independently sourced prior in `cos(i)` with quoted uncertainty/source; never invent its width or learn it from KinMS. Same-data photometry is modeled jointly or training-only. Without identifiable external inclination information, promote only `u(R)`. The 28.9-to-25.26 degree difference changes speed at fixed LOS amplitude by about 1.13, insufficient alone to explain the approximately 1.46 V0 ratio.
- Audit `r_t/BMAJ=[0.05,0.1,0.2,0.4,0.8,1.6]`. Endpoint selection triggers an extended audit and forbids an inner-slope claim.
- With unit-integral `B_j`, use `I(x,y)=F sum_j[a_j B_j(x,y)]`, `a_j>=0`, `sum_j a_j=1`, small rank, and one common kinematic LOSVD; prohibit channel amplitudes. Retain the fixed Wiener 2-D template as an ablation. Start axisymmetric, then a positivity-preserving basis alternative for weak `m=1,m=2`, never raw sign-indefinite Fourier coefficients. Select regularization on training inner-validation and require `u(R)` stability across rank in the accepted identifiable subspace.
- Derive ring support from training intrinsic-emissivity cumulative R95, beam response, and sensitivity, not the clipped 4.19-arcsec sky radius. Start with at most four active angular-speed knots. Use Fisher/Jacobian diagnostics whitened by validated covariance to merge/remove unsupported knots. New nonuniform knots use spacing-aware curvature of normalized `u/Uref` versus `R/Rsupport`, quadrature weights, and train-only fixed scales/strength; do not reuse index-only differences or monotonic coupling. Inclination remains necessary for spatial deprojection even when `u` is reported.
- Jointly re-optimize center, PA, systemic velocity, flux, dispersion, `u(R)`, and allowed geometry. Conditional Stage B remains diagnostic. Use 12 deterministic starts spanning PA symmetry, LOS amplitude, turnover, and inclination; at least three must lie within 0.1 chi-square with projected scaled-gradient norm at most `1e-3`. Bound/unidentifiable parameters cannot be promoted.
- Use constant positive dispersion first; test a smooth positive two-zone `sigma(R)` only after operator and brightness parity. Thickness, warps, and radial flows require later residual-driven proposals.

## Blocking S0 audit ledger

As the first licensed stage before model expansion or production, correct and test: blank/non-rotating null semantics; omega label/threshold; `forward/sb.py` hard-coded 224.3 GHz and `sigma_empty=0.02*peak`; and inverted `mock_inner_slope_recovered` semantics. The KGAS007 posterior samples flux, PA, systemic velocity, dispersion, V0, and turnover with center/inclination fixed; its R-hat/ESS apply only conditionally. Its retained posterior `config.yaml`, not current [`KGAS007.json`](../../configs/targets/KGAS007.json), lacks row/channel/bin/weight metadata; its target-specific mock was waived.

Evidence: [`chi2.py`](../../src/kinuv/likelihood/chi2.py), [`rotation.py`](../../src/kinuv/profiles/rotation.py), [`KinMS worker`](../../external/_kinms_best_worker.py), [`template builder`](../../src/kinuv/forward/sb.py), [`mock benchmark`](../../scripts/run_s3_mock_benchmark.py), [controlled mock](../reviews/artifacts/2026-09-05-kgas066-s3-image-benchmark/mock_controlled/summary.json), and [KGAS007 profiles](/arc/projects/KILOGAS/analysis/toby_sandbox/results/production/KGAS007/kinuv-KGAS007-e1ee1a-milestone1/benchmark/profiles.npz).

## Stages

| Stage | Owner after license | Dependency | Evidence |
|---|---|---|---|
| S0 audit | Implementer | two proposal accepts | Correct null, posterior, data, brightness-frequency/noise, omega, and mock-predicate semantics |
| S1 operator/comparator | Implementer | S0 | Intrinsic KinMS adapter, analytic closure, fixed rendering seeds and high-cloud repeat |
| S2 geometry/covariance | Implementer | S1 | Turnover grid, 12-start basin, line-free covariance/whitening report |
| S3 joint ablations | Implementer | S2 | One-factor emissivity, supported rings, then dispersion; joint nuisances and Jacobian evidence |
| S4 paired selection | Implementer | S3 | Preregistered paired mocks and valid held-out folds; bootstrap and image diagnostics |
| S5 verification | Registrar/reviewers | S4 | Dual code reviews, manifests/checksums, gate dossier, Consultant sign-off |

No NUTS or calibrated-interval claim is licensed.

## Frozen prospective gates

S1 float64 analytic complex-visibility relative L2 is at most `1e-6`, noise-normalized per-component RMS at most `0.001`, zero-baseline flux error at most `0.1%`, centroid error at most `0.02` native channel, and doubled spatial/spectral sampling changes chi-square by at most `0.1`. KinMS rendering-noise RMS is at most `0.1` thermal SD and high-cloud convergence changes chi-square by at most `0.1`, with fixed/common RNG and an independent high-cloud repeat. One noisy visibility realization feeds both fits in every pair; KinMS images that same realization through a frozen external recipe. Native models may differ but receive the same data. Whitened mean is within three SE of zero; variance is within `max(0.05,3 sqrt(2/nu_eff))` of one; prespecified lag correlations are at most `max(0.05,3/sqrt(N_eff))`. Derivations and unavailable metadata must be reported. Baseline replay must reproduce chi-square within `0.1`; report deterministic full-pipeline latency and peak memory on matched hardware/backend and optimizer budgets, without a speed-superiority claim.

A nested extended kinematic profile must gain at least 10 training chi-square over matched arctan to remain a candidate; this is not a significance claim. S4 has 16 crossed cells: two actual uv samplings by inclinations 29/44 degrees by smooth/asymmetric brightness truths by `1x/2x` noise, with at least 32 paired independent preregistered realizations per cell (512 pairs). Use independent truth rendering, high-resolution convergence, and identical training masks/brightness inputs. Numerical failure rate is at most 5% per cell. Out-of-family thickness/non-circular regimes require later proposals.

Use a fixed truth grid spanning cumulative 5--95% truth-emissivity flux with fixed weights `w_j`: `E^2=sum_j w_j[u_fit(r_j)-u_truth(r_j)]^2/sum_j w_j`. Aggregate ratio is `sqrt(sum E_kinUV^2/sum E_KinMS^2)` with equal preregistered cell weights, never mean realization ratios. Per target, superiority requires ratio `<=0.90` and one-sided paired stratified-bootstrap upper 95% bound `<1` from 10,000 whole-realization resamples and a fixed seed. Every cell upper bound is at most 1.05 as an intersection-union claim; otherwise report inconclusive, without favorable-cell selection or uncorrected per-cell superiority. KGAS066 new-versus-frozen-baseline upper bound is at most 1.05 in every cell. Intrinsic `v_c` is eligible only with propagated, identifiable external inclination uncertainty. Mock flux absolute fractional error has median at most 5% and 90th percentile at most 10%; signed radial bias is at most 5% of measurable truth-u amplitude. Report geometry separately: absolute median PA bias at most 3 degrees, center error at most `0.1 BMAJ`, and systemic error at most 0.2 fitted-channel width.

PA errors use circular wrapping and the declared receding-side convention; controlled projection/reflection tests prevent 360-degree artifacts.

KGAS007 candidate-minus-current held-out predictive advantage must have group-bootstrap lower 95% bound above zero. KGAS066 upper 95% relative held-out chi-square degradation is at most 2%. A claim that kinUV outperforms KinMS on real held-out visibilities requires a positive lower 95% bound for kinUV's predictive advantage under the fair contract.

Secondary matched image budgets are KGAS007 voxel RMS `<=0.064421 K`, spectrum RMSE `<=3.60628 K`, PVD RMSE `<=0.023872 K`; KGAS066 allows at most 5% worsening versus a regenerated operator-correct baseline. Moment-1 profiles computed with kinUV PA/inclination/systemic velocity are common proxies, not latent circular-speed measurements. Never alter visibility selection, weights, or axes to meet image budgets. Failure is partial diagnostic progress returned to Astra.

Any change to physics, priors, covariance, comparison contract, selection, transforms, or gates requires a revised Consultant specification and two fresh reviews. Production promotion additionally requires S5 and Consultant sign-off.
