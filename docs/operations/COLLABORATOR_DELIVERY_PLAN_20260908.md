# Collaborator delivery plan: 2026-09-08

**Authority:** Project PI's pre-meeting delivery directive; issued by Astra.
**Executor:** Sol, Senior Implementation Engineer.
**Targets:** KGAS066 and KGAS007.
**Starting repository:** `dev`, `b4d9b726698ccb2c37d5e8ce8f6c55410843ef11`.
**State:** Authorized execution contract; no new MAP or NUTS run is claimed here.

## Mandate and automatic progression

Deliver the best converged visibility fits found within this bounded campaign,
matching posterior products, the authenticated synthetic comparison, and clear
figures for the 2026-09-08 collaborator meeting. Execute phases 1--4 without
routine human check-ins. This PI directive supersedes the previous MAP-only
and no-NUTS restrictions and licenses the localized implementation needed for
this campaign despite the historical `code_freeze: true` status. It does not
license CASA reimaging, dark-matter decomposition, new radial-motion models,
or open-ended emissivity tuning.

Each target advances to NUTS as soon as its selected MAP is valid and resources
are available; it need not wait for the other target. Phase 3 rendering can
overlap inference. Final production promotion requires independent dual accept.
No new proposal/ADR round is required before implementation. Sol owns software
design, worker scheduling, and routine numerical repairs inside this contract.

The physical goals are to resolve the approaching-wing deficit, major-axis
envelope offsets, and coherent minor-axis/moment-1 residuals. Optimize only the
visibility objective. Restored cubes and KinMS image residuals are supporting
diagnostics, not ground truth or penalties added to the likelihood. Report any
remaining residuals honestly; their disappearance is not guaranteed by this plan.

## Shared inputs, resources, and storage

- Start from `results/production/<TARGET>/best_model/parameters.json` and
  `checkpoint.json`, bound to
  `results/validation/crossdomain-recovery-s4-remediation-20260907-r1/`.
  Preserve those accepted inputs and all historical S4/S5 evidence.
- Use the metadata-complete `visibilities/KILOGAS066.v2.npz` and
  `visibilities/KILOGAS007.v2.npz`, with the accepted selection, C1 covariance,
  visibility weights, phase center, channel edges, frame conversion, PB model,
  and spectral response. Resolve exact inputs from checkpoint provenance.
- Use up to **16 CPU cores and 32 GB RAM in aggregate**, not per target.
  Start with at most four workers, each with at most four computational threads.
  Schedule all eight MAP tasks through this shared pool. Cap nested BLAS/XLA
  threads and leave memory headroom for the controller and operating system;
  reduce concurrency before approaching the RAM limit. Reuse compiled operators
  where practical. Rendering and sampling share the same resource budget.
- Use node-local `/scratch` for compilation caches and frequent checkpoints.
  Flush recoverable chain state and bounded logs to a unique durable campaign
  directory under `results/incoming/`. Headless workers must survive terminal
  disconnect, retain exit codes/PIDs, and be monitored by a durable controller.
  A background PID is not task completion.
- Use ASCII-only terminal output, file logs, disabled animated progress bars,
  and a compact heartbeat approximately once per minute. Do not send external
  notifications. Record seeds, environment, exact implementation SHA, resolved
  configuration and input hashes as part of the run, without a separate
  pre-flight accounting report.

## Phase 1: parallel bounded joint visibility MAP

For the arctan/two-zone model, use the conditioned chart

\[
\{\log F,\log u_\infty,\log R_{\rm turn},\cos i,\mathrm{PA},
V_{\rm sys},\Delta x,\Delta y,\log\sigma_{\rm inner},
\log\sigma_{\rm outer}\},\qquad V_{\rm flat}=u_\infty/\sin i.
\]

Free inclination and centering; these must not become fixed host constants.
Keep the approved broad physical bounds and isotropic orientation prior.
Starting inclinations are not new priors. Preserve the physical prior measure
when changing computational coordinates; a log coordinate does not itself
authorize a log-uniform physical prior.

KGAS007 retains its supported projected-speed knots and interpolation. Replace
the arctan amplitude/turnover coordinates with the active projected-speed knot
coordinates; do not add an unidentifiable amplitude alongside every free knot.
No scalar `R_turn` or asymptotic `V_flat` exists for that family by default.
Preserve the accepted per-target dispersion prescription: KGAS066 has two free
zones; KGAS007 retains its accepted tied dispersion unless a separate physical
extension is authorized. The shared contract is free geometry and projected
kinematics, not an artificial identical vector containing inactive parameters.

Hold accepted emissivity shape/weights fixed and free total flux normalization.
Posterior results will therefore be conditional on this retained morphology.

Run **four starts per target**, with all active coordinates free after seeding:

| Start | KGAS066 | KGAS007 |
|---|---|---|
| 1 | Accepted checkpoint | Accepted checkpoint |
| 2 | `u_inf=205 km/s`, `i=55 deg` | Accepted projected-speed knots multiplied by 1.11; accepted inclination |
| 3 | `u_inf=210 km/s`, `i=65 deg` | Accepted projected-speed knots multiplied by 1.14; accepted inclination plus 10 deg |
| 4 | Accepted checkpoint with systemic velocity shifted by -10 km/s | Same relative systemic shift |

Apply the systemic shifts in the declared optical-LSRK reporting convention,
then convert through the existing native-frame contract. Other seed values come
from the accepted checkpoint. Keep seeds within approved bounds and record any
necessary clipping. Do not expand to a Cartesian restart grid.

Use the existing gradient-based bounded optimizer, parameter scaling and
convergence controls. Select the physically admissible, converged start with
the lowest visibility negative log-likelihood; likelihood ranking must exclude
prior penalties even when the fitted objective includes the accepted priors.
The ordinary fit result contains both terms, optimizer termination and bound
status; do not require a separate historical likelihood/prior replay report.
Keep the accepted checkpoint as a candidate if no new converged fit improves it.
Do not require all four starts to reach the same basin. Retain competing modes.

**Automatic transition:** save selected parameters, active/free names, likelihood,
prior specification and optimizer result; update STATUS and enqueue phase 2.
Non-finite fits, an unidentifiable primary parameter at a bound, or an invalid
gradient cannot qualify for automatic sampling. Local solver repairs are allowed;
do not hide failures or change the physical model to obtain a passing status.

## Phase 2: automatic production headless NUTS

Immediately launch the selected target's NUTS job when phase 1 completes and a
worker allocation is available. Use the **same active model, C1 likelihood,
data, emissivity conditioning, priors and physical parameter freedoms** as MAP.
Include the required sampling-coordinate Jacobian exactly once. PA wrapping
must be handled consistently in the sampler, summaries and mixing diagnostics.

Implementation prerequisite, not a new approval round: the existing
`src/kinuv/infer/nuts.py` and target-specific headless scripts implement an older
six-parameter chart with fixed centering and legacy input routes. They cannot
be launched unchanged for this campaign. Sol must connect the existing NUTS
machinery to the current joint model and add focused tests of parameter freedom,
finite gradients and MAP/NUTS likelihood identity. Reuse the production forward
operator; do not construct a secondary Fourier engine. Do not silently fall
back to the legacy posterior or a differently named sampler.

Initial allocation: **four independently seeded chains per target, 1,000 warm-up
and 1,000 retained draws per chain**, target acceptance 0.90, maximum tree depth
10. Initialize from small, independent valid perturbations around the selected
MAP, scaled in its conditioned coordinates. Retained competitive modes must
be investigated or explicitly identified as unsampled; four near-identical
chains alone do not demonstrate global exploration.

Sol may schedule chains in waves or use the available parallel-chain backend.
Do not run eight memory-intensive chain processes merely to make both targets
appear simultaneous. Checkpoint chain identity, RNG state, adaptation state and
draw position. Resume only a compatible chain; never merge overlapping draws
or warm-up samples into production draws.

On completion, automatically write chain/draw-labelled posterior samples,
16th/50th/84th percentiles in physical units, covariance and correlation matrices,
and rank-normalized split R-hat, bulk/tail ESS, divergences, BFMI and tree-depth
saturation. Derive `V_flat` or intrinsic knot velocities draw by draw using each
draw's inclination. Retain chain identity for every summary and corner plot.

**Posterior acceptance:** R-hat <= 1.05 and bulk/tail ESS >= 400 for primary
kinematics/geometry: projected and intrinsic velocity summaries, arctan turnover
where applicable, inclination, PA, systemic velocity and active dispersions.
For KGAS007 use supported projected-speed summaries, not invented arctan
parameters. Also inspect centering correlations. Peripheral nuisance ESS alone
does not veto an otherwise valid primary result. Require no retained divergences,
finite valid draws, and no unresolved energy/tree-depth pathology affecting the
primary posterior. Do not discard bad chains or individual divergent draws to
manufacture acceptance.

If ESS is the only failure, extend each compatible chain by up to 1,000 draws.
If adaptation is inadequate, one automatic retry with 2,000 warm-up steps and
target acceptance 0.95 is authorized; Sol may adjust tree depth within the
resource budget. Retain the unsuccessful attempt. Remaining failure is reported
as an unaccepted posterior, not relabeled as convergence.

Mixing is not coverage calibration. Report these intervals as conditional on
the declared model, morphology and priors; do not claim calibrated frequentist
coverage without a corresponding study. Broad inclination uncertainty must
remain visible in intrinsic-velocity intervals.

## Phase 3: authenticated mock ground-truth comparison

Re-render, without refitting or changing scores, the accepted Python-native
synthetic evidence from
`results/validation/crossdomain-recovery-s4-final-20260907/synthetic/` and
`results/validation/crossdomain-recovery-s5-subbeam-20260907/`.
Verify their manifests and retain truth, seed and fitted-model identities.
No CASA dependency or training-fold reimaging is permitted.

For truths with `R_turn < BMAJ`, show the per-realization absolute turnover error
normalized by BMAJ and projected-velocity RMSE restricted to `r <= BMAJ`, alongside
the full radial profiles and full-disk RMSE. Retain the same radial support and
metric definitions for both methods. Show individual realizations as well as
the aggregate; do not select favorable seeds.

| Registered result | KGAS066 | KGAS007 |
|---|---:|---:|
| Mean absolute turnover error, kinUV / KinMS (arcsec) | 0.000996 / 0.039977 | 0.020044 / 0.072452 |
| Turnover-error ratio, kinUV / KinMS | 0.02493 | 0.27665 |
| Inner-beam projected-velocity RMSE, kinUV / KinMS (km/s) | 0.13830 / 9.61877 | 0.63702 / 4.63336 |
| Inner-beam RMSE ratio | 0.01438 | 0.13749 |

The approximately 40-fold turnover-error advantage belongs to KGAS066's
registered synthetic experiment. It is not a claim about both galaxies or
unknown real truth. Existing synthetic tests use an arctan truth/recovery
experiment; they are not new validation of the real KGAS007 knot posterior.
These are strong reproducible results for the tested family, not universal
proof of superiority across all possible galaxies or optimizers.

Explain spectro-astrometric sensitivity using

\[
\Delta\phi(\mathbf q,v)\simeq
-2\pi\mathbf q\cdot[\overline{\mathbf x}(v)-\overline{\mathbf x}(v_{\rm ref})],
\]

with baselines `q` in wavelengths and centroids in radians, in the marginally
resolved/short-baseline limit. Differential phases constrain channel-dependent
centroids while amplitudes constrain flux and spatial structure. Clumpy emission
also moves flux-weighted centroids: this is complementary information, not exact
statistical independence of kinematics and morphology. The relation motivates
the method; recovery measurements establish the demonstrated advantage. See
[Lachaume (2003)](https://arxiv.org/abs/astro-ph/0304259).

Visibility inference avoids requiring CLEAN inversion and restoring-beam pixel
covariance. The actual exported visibilities retain measured spectral
correlation, treated by C1; do not describe them as universally uncorrelated.
Keep the accepted real grouped-prediction evidence separate from newly fitted
full-data MAP scores. Historical lower 95-percent gains of 0.03178592 and
0.00290304 chi-square/component are not freshly measured gains for this campaign.

## Phase 4: figures, independent review, and production seal

Use `kinuv.diagnostics.style` with its internal ApJ font stack, STIX math,
Type 42 fonts and inward ticks. Do not install an external formatting package.
Produce legible figures at intended presentation/publication size: axis labels
at least 12 pt, ticks at least 10 pt, annotations at least 11 pt. Sol owns layout.

Final destinations beneath
`/arc/projects/KILOGAS/analysis/toby_sandbox/results/production/<TARGET>/`:

| Directory | Required contents |
|---|---|
| `best_model/` | Selected MAP, resolved configuration, likelihood summary, checkpoint provenance, matched model cube, posterior samples and diagnostics, percentile/covariance tables, manifest |
| `plots/` | `moments_comparison`, `pv_diagrams`, `spectral_profiles`, `rotation_curve`, `posterior_corner`: PDF and PNG pairs |
| `benchmarks/` | `moments_kinuv_vs_kinms`, `pvd_kinuv_vs_kinms`, `spectra_kinuv_vs_kinms`, `rotation_curve_kinuv_vs_kinms`, `synthetic_benchmark`: PDF and PNG pairs; a JPEG PVD copy for the requested presentation workflow |

Moments must include M0/M1/M2 data, model and residuals, common physically defined
support, explicit units, restoring beam and a crop enclosing the emission.
PVDs include both axes using the repaired astronomical PA convention. Spectra
include residual tracks under the same aperture. Use common data/model scaling
and symmetric residual limits; do not hide mismatches by independent rescaling.
Show beam scale and turnover for arctan profiles; mark turnover not applicable
for the real knot model. Synthetic arctan turnover remains valid for that mock.

Keep one selected MAP cube for model panels; distinguish it from posterior
credible bands derived draw by draw. Never render a component-wise median
parameter vector and imply that it is necessarily a sampled physical solution.
Corners must use this campaign's NUTS samples, show primary parameter correlations,
and identify conditional or unconverged status. Do not reuse historical corners.
Refresh all active links and remove stale figures through verified archiving.

Generate into a new versioned candidate bundle and verify it before installation.
Each manifest binds the code/environment, configuration, inputs, MAP, chains,
synthetic sources, and every delivered product by size and SHA-256. Preserve the
old production bundle in a verified archive before replacing canonical paths.
Keep large cubes/traces in durable `/arc` storage; commit code, configurations,
compact evidence, manifests and documentation to Git, not caches or raw MS data.

**Independent review board:** Sol commissions Reviewer A and Reviewer B after
candidate outputs are sealed. Each reviews the same exact code/evidence hashes
and independently runs checksum and provenance audits before reading the other
verdict. Reviewer A checks physical parameterization, C1/response consistency,
posterior validity, synthetic arithmetic/claim scope and science completeness.
Reviewer B checks reproducibility, sample/chain integrity, input/output identities,
headless completion, archive verification and consistent figure provenance.
Record `accept` or `changes-requested` under `docs/reviews/`, identifying exact
reviewed hashes. Neither reviewer implements the candidate. Repair substantive
findings and re-review affected evidence; no tests of prose or procedural wording.

Dual accept is required before replacing the accepted production record. Sol
then records both verdicts, exact implementation/evidence commits, scalar results,
product paths and limitations in STATUS, PLAN and PRODUCTION_RECORD; commit and
push to `origin/dev`. No further interactive approval is needed. Astra receives
the completed science handoff, rather than a request to restart the pipeline.

## Deadline and failure handling

The meeting date is 2026-09-08; no exact meeting time has been supplied. Start
immediately when Sol takes over and prioritize validated MAP figures and the
already accepted synthetic packet while NUTS runs. Record actual throughput and
ETA after work begins; do not promise a wall time unsupported by measurement.

Checkpoint and report numerical/resource failures promptly. A bounded retry may
reduce parallelism or repair a solver/interface defect without changing physics.
An unresolved physical degeneracy, invalid posterior, failed review or lack of
time must remain visible. Do not relax convergence gates or claim a global best
model or universal superiority to meet the presentation deadline.

If NUTS cannot finish and pass, deliver the independently reviewed MAP-only
meeting packet with that explicit label, preserve the prior accepted production
bundle, and keep incomplete chains outside accepted posterior products. Continue
authorized recoverable work automatically; escalate genuine blockers with
evidence. A posterior may be promoted later only after its own dual acceptance.

The final handoff lists both targets, MAP selection and parameters, posterior
status/R-hat/ESS/divergences, unchanged or new metric provenance, figure paths,
archive/manifest identities, reviewer verdicts, and Git hashes. This document
authorizes the execution; it does not report that execution has already occurred.
