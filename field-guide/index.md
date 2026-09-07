# kinUV Production Field Guide

This guide defines how kinUV science is specified, implemented, verified, run, and promoted. It is deliberately independent of any galaxy, survey, filesystem layout, sampler result, or current campaign. Target values and campaign thresholds belong in versioned configuration and run manifests.

## 1. Scope and source of truth

kinUV is a standalone visibility-domain modeling and inference engine. This guide governs production work in the repository. It does not contain target coordinates, inclinations, position-angle seeds, data paths, fitted values, or target-specific pass thresholds.

When records disagree, apply this order:

1. The Project PI's current written directive.
2. Astra's current written directive within the PI-defined boundary.
3. Accepted architectural and scientific decisions in `docs/decisions/`.
4. This field guide.
5. The active campaign proposal and its frozen configuration.
6. `docs/architecture/STATUS.md` and `docs/reviews/BOARD.md`.
7. `PLAN.md`, diagnostics, and implementation notes.

Historical campaign decisions remain evidence for their original products. They do not silently become defaults for another target or campaign.

## 2. Agent hierarchy and authority boundaries

Agent names describe authority, not a particular vendor or model release. The same agent must not approve its own scientific proposal or review its own implementation.

| Role | Typical capability | Owns | Must not do |
|---|---|---|---|
| **Project PI** | Human authority | Mission, publication authority, gate overrides, final dispute resolution | Be displaced by an agent-authored threshold |
| **Principal Architect: Astra** | Lead scientific authority | Mission architecture, physical strategy, scientific gates within the PI boundary | Re-impose a gate superseded by the PI |
| **Consultant: Lead Architect / frontier model** | Highest available reasoning capability | Scientific strategy, parameterization, physical priors, model class, declared gate criteria, residual-risk acceptance, scientific promotion sign-off | Run an unreviewed production campaign or delegate away scientific accountability |
| **Senior Registrar** | Senior coordinating agent | Canon, proposal registration, configuration freeze, provenance, review tally, state transitions, promotion dossier | Relax a scientific criterion or reinterpret a failed gate as a pass |
| **Implementer** | Execution model such as Composer or GPT-5.6 | Code, tests, diagnostics, profiling, refactoring, batch execution, bounded operational decisions inside the accepted specification | Change priors, likelihoods, sign conventions, model class, or promotion thresholds without change control |
| **Reviewer A: science/numerics** | Independent senior reviewer | Physical validity, identifiability, units, conventions, likelihood, mocks, statistical gates | Read Reviewer B before submitting a verdict or implement the proposal under review |
| **Reviewer B: software/reproducibility** | Independent senior reviewer | Package boundaries, tests, data contracts, performance, storage, manifests, failure recovery | Read Reviewer A before submitting a verdict or waive a scientific failure |

The Consultant writes or signs scientific specifications. The Registrar converts production campaigns into executable records without changing their meaning. The Implementer owns localized corrections that restore an already declared mathematical or physical contract.

Any proposed change to the model, prior, covariance, data selection, or gate threshold returns to the Consultant. A localized mathematical defect fix may proceed immediately with focused tests, an atomic commit, and a STATUS entry. It does not require a new ADR, proposal review, or pre-implementation sign-off when it preserves the declared science and acceptance threshold.

An Implementer may stop a run immediately for corruption, invalid numerics, resource exhaustion, or a failed gate. Stopping protects the specification; it does not constitute authority to weaken it.

### Governance & Engineering Velocity Invariant

* Proportional Verification: Secondary reference engines (e.g., ungridded slow-DFT direct-cloud calculations) are prohibited unless a primary numerical implementation demonstrably fails empirical refinement closure on target data.
* Direct Repair Authority: Localized mathematical, numerical, unit, or interpolation repairs do not require formal ADR drafting, immutable dossier serialization, or pre-implementation approval rounds. Implementers proceed directly with code fixes.
* Bias Toward Velocity: Directives must prioritize empirical numerical closure (refinement gate ≤ 0.1) over asymptotic float64 perfection (1e-6) to maintain engineering momentum and avoid review deadlock.

### Architectural Scope & Delegation Invariants

* Visionary Architecture, Not Micromanagement: The Lead Architect (Astra) formulates physical hypotheses, identifies model degeneracies, defines high-level likelihood formulations, and sets scientific acceptance gates. Astra must not prescribe line-by-line implementation mechanics, low-level data structures, or auxiliary test harness designs.
* Trust the Implementer: Once a physical direction is set, Senior Implementer agents (e.g., GPT-5.6 Sol) hold full authority over software design, numerical quadrature choices, algorithm efficiency, and code organization.
* Prohibition on Overengineering: Astra is strictly barred from requiring secondary/redundant reference engines (e.g., ungridded slow-DFT calculations) or extreme theoretical tolerances (e.g., 1e-6 float64 closures) for localized bug fixes when a standard empirical refinement gate (e.g., <= 0.1) confirms physical convergence on real data.
* Bias Toward Delivery: Theoretical conservatism must not stall development velocity. Architectural specifications must target the minimum viable mathematical formulation required to pass empirical gates on target datasets.

### Empirically Right-Sized Gate Invariant

The binding S2-S5 rules are recorded in
`docs/decisions/DEC-HUMAN-OVERRIDE-RIGHTSIZED-GATES.md`.

* S2 multi-start acceptance requires consensus in the top-ranked,
  lowest-chi-square cluster. It does not require every initialization to reach
  one identical point. All starts and failures remain visible.
* Native-to-reporting spectral-frame variation below 1.0 km/s is sufficient
  when the conversion and its variation are preserved in provenance.
* For posterior campaigns, R-hat <= 1.05 and ESS >= 400 apply to primary
  kinematic and geometric parameters. Peripheral nuisance mixing becomes a
  veto only when it changes a primary result or promoted prediction.
* Cross-domain superiority requires at least 10 percent lower projected-
  velocity RMSE against registered truth and improved overall reduced
  chi-square on both real canonical benchmarks. Per-region residual dominance
  is diagnostic.
* Reviewers must return `changes-requested` on an asymptotic or unphysical gate
  whose strictness is unsupported by resolution, calibration, noise, or the
  promoted scientific claim.

## 3. Configuration boundary

Target and campaign state must be external to the operational manual and modeling modules.

- Target metadata belongs in `configs/targets/<target_id>.yaml` or an equivalent versioned catalogue adapter.
- Campaign choices belong in `configs/campaigns/<campaign_id>.yaml`.
- Machine paths belong in deployment configuration or environment variables.
- Secrets, credentials, and short-lived certificates never belong in configuration committed to Git.

A target configuration records coordinates and reference frame, systemic-velocity convention, inclination information, position-angle convention, phase centre, data-product identifiers, and scientifically justified priors. A campaign configuration records model family, free and fixed parameters, initialization policy, transform and spectral response, covariance treatment, null models, gate criteria, sampler settings, random seeds, storage root, and requested products.

Configuration is immutable after a production run starts. A correction creates a new configuration revision and run ID. Local development and numerical gate runs need only record the exact commit, inputs, settings, and metrics required to reproduce the decision.

## 4. Architectural invariants

These are production invariants. Violating one requires an architectural decision and a new review; a STATUS note cannot waive it.

### Package boundaries

1. `kinuv` must import and run without uvkin, uvfit, or any archived predecessor on the filesystem or Python path.
2. CASA, casatasks, python-casacore, pyuvdata, and Measurement Set calibration code remain outside kinUV.
3. `ms2kinuv` is a separately installed ETL companion. kinUV consumes its versioned data product and never imports it at runtime.
4. Legacy archives are read-only evidence. They are never added to `sys.path`, installed into a production environment, or used as a hidden backend.
5. CI must scan production source, scripts, and package metadata for forbidden imports, sibling-path injection, and legacy symlinks.

### Data and coordinate contracts

1. Visibility inputs declare schema version, units, frame, polarization selection, flag treatment, and provenance.
2. Projected baselines are stored in metres. Per-channel spatial frequencies are derived from the actual channel frequencies and the speed of light.
3. Complex visibility, weight, frequency, time, baseline, and phase-centre arrays pass shape, dtype, finiteness, positivity, and alignment checks before modeling.
4. Fourier sign, sky-axis orientation, position-angle definition, velocity convention, and phase-centre behavior are explicit and covered by analytic tests.
5. Flags map to zero statistical weight. Weight rescaling and covariance correction are recorded transformations, never implicit constants.

### Model and likelihood contracts

1. The forward model, primary beam, astrometric shift, spectral response, channel binning, and visibility sampling execute in a declared order.
2. Data and model receive compatible spectral operators. Previously correlated or smoothed data are not smoothed a second time.
3. Likelihood accounting includes both real and imaginary visibility components and states the assumed covariance.
4. Image-plane cubes, moments, spectra, and position-velocity diagrams are diagnostics unless an accepted specification explicitly defines an image-domain likelihood.
5. Parameter bounds are computational constraints only when declared as such. A posterior or optimizer pressing a bound is a failed identifiability or prior-pressure diagnostic, not a measurement.
6. Sampler labels describe the algorithm actually run. Approximate, Laplace, importance, or Metropolis results are not relabeled as HMC or NUTS.
7. The scientific likelihood is evaluated from visibility residuals. Dark-matter profiles, baryonic mass models, cosmology, physical-radius grids, and mass decomposition are excluded from the forward-model and sampler dependency graph.
8. A fitted rotation profile is a kinematic representation. Gravitational interpretation is a versioned downstream analysis that consumes an immutable posterior product and cannot feed values back into the visibility likelihood.

### Product integrity

1. Promoted products are immutable. A rerun writes a new run directory.
2. Every product identifies code commit, dirty-state hash, resolved configuration, input checksums, environment, backend, random seeds, and gate results.
3. Failed, interrupted, unmixed, or uncalibrated outputs remain evidence with an explicit state; they are not promoted or silently merged.

## 5. Generalized scientific pipeline gates

The Consultant declares quantitative tolerances before seeing production answers. Tolerances may vary with precision, signal-to-noise, uv coverage, and scientific objective, so target values do not appear in this guide. Each gate produces a machine-readable result and a short scientific interpretation.

### Gate 0 — Intake and preflight

- Validate configuration schemas and freeze resolved copies.
- Verify input hashes, units, frames, phase centre, polarization, spectral ordering, flags, and weight support.
- Record array dimensions, channel width, baseline range, time sampling, and missing-data fractions.
- Confirm the environment imports kinUV with forbidden packages absent.
- Estimate memory, runtime, and storage before allocating the production run.

Failure blocks all later gates.

### Gate 1 — Analytic closure

- Start with controlled invariants and factor-two refinement on the production numerical path.
- Test amplitude normalization, Fourier sign, east/north orientation, channel-frequency scaling, phase shifts, primary-beam ordering, spectral response, and guard channels.
- Compare automatic gradients with finite differences when gradients are part of the changed path.
- Do not construct a secondary brute-force or reference engine while the primary method satisfies its empirical closure gate. Authorize a secondary reference only after a primary refinement failure leaves the cause unresolved.

The declared tolerance must reflect numeric precision and the downstream noise floor. A transform that fails closure cannot proceed to empirical tuning.

### Gate 2 — Mock recovery and identifiability

- Generate exact-model mocks with the campaign's actual sampling and covariance.
- Recover all promoted parameters from multiple initializations and relevant symmetry-related modes.
- Include noise realizations spanning the intended operating regime.
- Test known misspecification cases separately from exact closure.
- Report bias, dispersion, interval coverage where applicable, parameter correlations, bound pressure, and failure rate.

Passing one convenient injection is insufficient. Parameters that are not identifiable must be fixed, reparameterized, regularized by a declared physical prior, or removed from the promoted claim.

### Gate 3 — Null and baseline comparisons

- Evaluate the declared zero-signal null and any scientifically required nested baseline model on exactly the same visibility cells and covariance.
- Report raw likelihood or chi-square terms, the direction and magnitude of improvement, parameter counts, and the selected comparison criterion.
- Keep prior contributions separate from likelihood-only null comparisons.
- Verify that the optimizer did not manufacture improvement through a sign, phase-centre, flux-normalization, or data-selection error.

A positive improvement alone does not establish scientific adequacy. The proposal defines the threshold and its interpretation before the production fit.

### Gate 4 — Covariance and residual adequacy

- Estimate the weight scale from declared line-free or noise-only data without contaminating signal channels.
- Measure residual correlation across frequency, baseline, time, polarization, and repeated averaging groups.
- Test whitened residual location, scale, tails, and structured dependence.
- Account for known correlator, smoothing, averaging, and binning correlations with a covariance operator or demonstrate that the diagonal approximation meets the declared tolerance.
- Re-run Gates 2 and 3 if covariance treatment changes.

Posterior sampling is prohibited until this gate passes or the Consultant explicitly narrows the scientific claim to a documented diagnostic result.

### Gate 5 — MAP stability and model adequacy

- Use multiple starts for periodic, reflected, or otherwise multimodal coordinates.
- Record optimizer status, gradients, evaluations, bound pressure, and sensitivity to initialization.
- Compare residual structure against visibility coordinates and frequency, and inspect image-domain diagnostic products.
- Distinguish kinematic mismatch from surface-brightness, calibration, primary-beam, or covariance mismatch.

The MAP supplies initialization and a reproducible likelihood identity. It is not by itself a calibrated uncertainty result.

### Gate 6 — Posterior sampling and convergence

- Use the sampler and parameter chart named in the accepted campaign specification.
- Run independent chains with recorded seeds and initial states.
- Report split rank-normalized convergence statistics, bulk and tail effective sample sizes, divergences, energy/BFMI diagnostics, tree-depth saturation, acceptance behavior, and chain-wise parameter ranges as applicable.
- Test all relevant posterior modes or state which mode the product conditions on.
- Run simulation-based calibration or an accepted coverage study before describing intervals as calibrated.

The Consultant sets numeric convergence and coverage thresholds. An Implementer may extend a run under an approved contingency but may not lower the thresholds, drop a bad chain, or merge incompatible configurations to obtain a pass.

### Gate 7 — Promotion

- Recompute the promoted likelihood identity from the saved parameter record.
- Generate declared visibility residuals and Data/Model/Residual diagnostic figures.
- Verify manifest completeness, checksums, sampler label, units, and configuration identity.
- Record every gate as pass, fail, waived-by-Astra, or not applicable, with evidence paths.
- Obtain Registrar verification and Consultant scientific sign-off.

Only then may a run become a production product or a source for scientific tables.

## 6. Production run protocol and state machine

Every run uses a unique, target-neutral identifier such as `<campaign>-<UTC timestamp>-<git short sha>-<kind>`. The resolved target ID is metadata, not executable naming logic.

Allowed states are:

`PLANNED → REVIEWED → PREFLIGHTED → RUNNING → CHECKPOINTED → SUCCEEDED → VERIFIED → PROMOTED`

Terminal non-promotion states are `FAILED`, `INTERRUPTED`, `UNMIXED`, `UNCALIBRATED`, and `REJECTED`. State transitions are append-only in the run manifest. File presence alone is not completion evidence.

Before a production inference campaign, the Registrar records its accepted specification, code commit, configuration and input checksums, expected resources, and output contract. This production ceremony does not apply to localized bug fixes or bounded numerical refinement runs.

Long jobs run asynchronously under the configured batch platform. Interactive agents monitor bounded status records and do not block on the sampling loop. A retry receives a new attempt identifier and links to its predecessor.

## 7. Storage tiering

Paths are resolved from deployment configuration. Source code must not embed a user's home directory, target directory, or site-specific project root.

| Tier | Purpose | Contents | Retention |
|---|---|---|---|
| Git repository | Reviewable source and compact evidence | Code, schemas, configuration, decisions, small summaries, selected figures | Permanent, versioned |
| Node-local `/scratch/kinuv-$USER/<run_id>` | High-frequency disposable I/O | JIT cache, temporary arrays, verbose sampler stream, staging files | Ephemeral |
| Durable `${KINUV_RUN_ROOT}/<run_id>` on `/arc` | Reproducible run record | Manifest, bounded logs, checkpoints, draws, posterior summaries, required figures | Durable, immutable after promotion |
| Data store | Calibrated scientific inputs | Visibility tables, FITS products, external catalogues | Managed independently; referenced by checksum |
| Archive | Closed legacy or superseded evidence | Verified compressed bundles with checksums and manifests | Durable, read-only |

Write large intermediate arrays to scratch first. Promote a checkpoint by closing it, validating it, copying to a temporary durable path, fsyncing file and directory, verifying size/checksum, and atomically renaming it. Never stream high-volume progress output, JIT caches, or repeatedly rewritten arrays directly to `/arc`.

On success or failure, preserve the bounded log, status, last valid checkpoint, environment record, and failure reason. Delete scratch only after durable verification. Never copy raw inputs into every run directory.

## 8. Review and promotion workflow

Use proportional verification. The evidence burden follows the scientific and operational risk of the change.

For a localized mathematical bug fix that preserves the model, priors, data selection, and frozen gate:

1. The Implementer writes the focused correction and regression tests.
2. Run the primary refinement gate on the affected target set.
3. If every declared target and axis passes, commit atomically, update STATUS, close the stage, and advance immediately.
4. If the primary gate fails after two bounded debugging iterations or exposes a scientific choice, escalate with numerical evidence.

An immutable dossier, new ADR, proposal tally, or pre-implementation sign-off is not required for this fast path. Preserve enough evidence to reproduce the reported metric; avoid serializing large duplicate arrays merely to document routine development.

New scientific models, priors, likelihoods, covariance families, data selection, thresholds, production campaigns, and publication promotions use the full workflow:

1. **Consultant specification.** Define the scientific question, model, parameterization, priors, covariance, gates, products, and residual risks.
2. **Registrar registration.** Create the proposal, assign a campaign ID, validate configurations, and freeze acceptance criteria.
3. **Independent review.** Reviewer A assesses science and numerics; Reviewer B assesses implementation and reproducibility. They work independently and issue `accept`, `accept-with-required-changes`, or `reject` verdicts with evidence.
4. **Tally.** The Registrar may license implementation only when both reviews accept and all required changes are incorporated into the frozen proposal. A rejection returns the proposal to the Consultant.
5. **Implementation.** The Implementer writes code and tests, runs the accepted gates, and records results. Dual code review is reserved for scientific-contract changes and production promotion, rather than routine localized repairs.
6. **Verification.** The Registrar checks that the implementation matches the frozen specification and that artifacts are complete and reproducible.
7. **Scientific sign-off.** The Consultant reviews gate evidence and signs or rejects promotion. Astra resolves exceptions and authorizes publication policy.
8. **Closeout.** Summarize durable findings, link the immutable product, clear the active review card, and archive superseded discussion.

Review comments are classified as required or advisory. Required comments block tally until resolved. No role may convert a failed scientific gate into an advisory comment after seeing the result.

## 9. Change control and emergency rules

- A scientific-specification change creates a revised proposal and repeats both reviews.
- A localized defect fix uses the proportional fast path. A production bug fix also records affected runs and whether products require invalidation or regeneration.
- A security or data-corruption issue may freeze execution immediately. The Registrar records the freeze and affected scope.
- Only Astra may waive an invariant or promotion gate. The waiver must identify the evidence, scope, expiration, and prohibited claims.
- Secrets are never committed. Promoted data are never overwritten in place.

## 10. Required campaign dossier

Each promoted production campaign links:

- accepted proposal and two independent reviews;
- resolved target and campaign configurations with checksums;
- input manifest and data-contract validation;
- environment and code provenance;
- Gate 1 analytic closure report;
- Gate 2 mock recovery report;
- Gate 3 null comparison;
- Gate 4 covariance report;
- Gate 5 MAP and residual diagnostics;
- Gate 6 convergence and calibration report;
- Gate 7 promotion receipt and sign-offs.

This dossier applies to production promotion, not localized repairs or bounded numerical stage checks. Missing production evidence means the campaign is incomplete, regardless of whether a plausible figure exists.
