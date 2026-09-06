---
role: reviewer
seat: science-numerics
phase: implementation
date: 2026-09-06
reviewer: /root/s0_review_a
canon_generation: 9
campaign_id: crossdomain-recovery-s0
proposal: docs/decisions/DEC-KINUV-CROSSDOMAIN-RECOVERY.md
reviewed_commit: fb4a14543d579168c9224c8ebca6a7591147f4db
verdict: accept
---
# Independent review

I reviewed commit `7471cd7154b9a9ec5edb238724bd769b239f49bf`
without reading or contacting Reviewer B. I did not modify the implementation.

## Attempted falsification

I tried to turn an emission-versus-blank score into a rotation claim, reverse
the likelihood-improvement sign, pass the null gate with insufficient trials,
reuse the KGAS066 residual-omega threshold outside its declared scope, and
trace the corrected mock and Ico-noise semantics through their downstream
records. I also inspected the reported KGAS066/KGAS007 calculations for prior
contamination and nuisance mismatch.

The focused S0 tests passed (`43 passed, 3 skipped`) and the complete suite
passed (`221 passed, 8 skipped`). The likelihood signs are correct:
`delta_chi2_blank = chi2_blank - chi2_rot` and
`delta_chi2_nonrot = chi2_nonrot - chi2_rot`. The non-rotating implementation
uses the same forward, visibility-sampling, Hann/bin, flux, systemic-velocity,
dispersion, center, center-prior, and final prior-free likelihood treatment as
Stage A; PA, inclination, and turnover are immaterial when circular speed is
fixed exactly to zero. The bootstrap plus-one arithmetic is also correct:
`k=1, M=199` gives `p=0.01`.

## Findings

1. **Required — the historical omega scope is descriptive, not enforced.**
   [`OmegaCriterion`](../../src/kinuv/profiles/rotation.py) stores a free-form
   `calibration_scope` string and `mock_count`, but
   [`stage_b_model_adequate`](../../src/kinuv/infer/stage_b.py) checks neither
   the target, ring count, residual-versus-absolute mode, reference curve, nor
   calibration identity of the result. The unit test even applies the declared
   fixed seven-ring KGAS066 criterion to a three-knot anonymous record. Thus a
   caller can pass a KGAS007 or otherwise unmatched result with
   `HISTORICAL_KGAS066_OMEGA_CRITERION` and receive `True`, contrary to the
   decision that `0.3` may survive only in its exact historical scope. Encode
   the scope as machine-checkable fields and reject mismatches, with a negative
   cross-target/cross-ring/cross-mode test. Production correctly passes `None`
   and is currently blocked; preserve that behavior.

2. **Required — the null-bootstrap result can claim scientific `pass` from
   unverified integers.**
   [`rotation_test_metrics`](../../src/kinuv/validation/rotation.py) returns
   `status="pass"` after receiving only `bootstrap_exceedances` and
   `bootstrap_trials`. It cannot establish the frozen requirements of
   preregistration, actual covariance, complete null and alternative refits,
   or repetition of data-dependent template/mask/regularization/initialization
   selection. Its test demonstrates a pass from three scalar chi-squares and
   two counts. Until the S4 evidence object exists, this function may report
   the arithmetic threshold result but must not emit an unqualified scientific
   `pass`. Alternatively, require and validate a structured bootstrap evidence
   record containing those invariants before `pass` is possible.

3. **Required — the mock-predicate rename is incomplete downstream.**
   [`run_s3_mock_benchmark.py`](../../scripts/run_s3_mock_benchmark.py) correctly
   replaces the inverted `mock_inner_slope_recovered` field with
   `mock_turnover_radius_recovered` and applies the same absolute turnover
   tolerance to both fitters. However,
   [`external/run_image_benchmarks.py`](../../external/run_image_benchmarks.py)
   still reads and publishes `mock_inner_slope_recovered` in two active summary
   paths. A newly generated mock will therefore silently propagate `null` under
   the old scientific label. Update all active consumers and add an end-to-end
   schema assertion for both kinUV and KinMS records.

4. **Required — the new scientific template input is absent from run
   provenance.**
   [`load_sb_template`](../../src/kinuv/forward/sb.py) now uses the companion
   `*_err.fits` median to set the Wiener scale. That is a material improvement
   over the hard-coded frequency and `0.02*peak` proxy. On the official maps I
   independently obtain median-error/peak ratios `0.0248877` (KGAS066) and
   `0.0364823` (KGAS007), so this input changes the morphology and reported
   chi-square. Yet `_require_files` in
   [`run_production_milestone.py`](../../scripts/run_production_milestone.py)
   hashes only the Ico image, not the derived error-map path, and the loader
   returns no noise-source or `k_wiener` metadata. Add the error map to the
   immutable input manifest and record its path/hash plus the reduction rule,
   scalar sigma, image peak, and resulting dimensionless Wiener factor.

5. **Advisory — duplicate thresholds can drift.** The target JSON files now
   contain `25`, `199`, and `0.01`, while the executable gate uses independent
   constants in `validation/rotation.py`. They agree in this commit and no
   target physical parameter was changed, but the runner does not verify their
   equality. Use one authoritative source or fail when configuration and the
   frozen decision differ.

6. **Advisory — preserve the limitations attached to the initial metrics.**
   The scratch record supports the STATUS values: KGAS066 has
   `(chi2_blank, chi2_nonrot, chi2_rot) = (204228.248, 200023.444,
   168526.073)` and KGAS007 has `(128282.392, 126567.540, 122144.496)`, giving
   prior-free non-rotating improvements `31497.371` and `4423.045`. Both null
   fits reach the 50 km/s dispersion ceiling and KGAS066 reaches the +2 arcsec
   declination-offset bound. They are therefore useful prospective accounting
   checks, not validated null distributions or robust rotation detections. The
   current STATUS states this limitation correctly.

7. **Advisory — canonical historical prose still carries the wrong unit.**
   `docs/MILESTONE-001.md` and `docs/PRODUCTION_RECORD.md` continue to label
   historical `max_omega` values in km/s. Preserve the sealed numeric products,
   but annotate these prose records so readers cannot mistake the known
   dimensionless quantity for a velocity.

## Gate assessment

The blank/non-rotating likelihood distinction, prior-free subtraction, minimum
delta chi-square, plus-one p-value arithmetic, and dimensionless omega formula
are measurable and correctly precommitted. The initial target metrics are
reported with appropriate boundary and bootstrap caveats. Stage B promotion is
correctly blocked in both target configurations and in the production runner.

S0 cannot close at this commit because the software does not enforce the
historical omega criterion's declared domain, can issue a null-bootstrap
`pass` without evidence for the frozen resampling contract, loses the corrected
mock predicate in downstream aggregation, and omits the error map controlling
the new Wiener template from input provenance.

## Residual risks

The non-rotating target checks use a single optimization start and both
dispersion estimates are boundary-limited. A complete bootstrap must repeat
every data-dependent operation and refit both hypotheses. The propagated Ico
error map is heteroscedastic; reducing it to one median Wiener scale is a
declared scalar approximation that later brightness ablations should test.
Neither the large delta chi-square values nor passing unit tests establish
rotation significance, posterior calibration, fair KinMS comparison, or
cross-domain superiority.

## Verdict rationale

**Accept with required changes.** The core numerical corrections are in the
right direction and the reported target arithmetic is internally consistent.
The four required issues are gate-integrity and provenance failures: each can
permit a scientifically invalid conclusion or make the corrected calculation
irreproducible without necessarily failing a unit test. S0 should remain open
until they are corrected and independently rechecked on a new exact commit.

## Re-review of revised commit

I independently re-reviewed exact commit
`776256f6b45923e1659a049c220db8e62ed1eee9` against the four required findings
above and the durable artifact at
`results/validation/crossdomain-recovery-s0-20260906/`. I did not read or
contact Reviewer B.

### Required-change checks

1. **Omega registration and scope — satisfied.** `OmegaCriterion` now carries
   machine-readable criterion, target, ring-count, metric-definition,
   reference-curve, spectral-response, mock-count, and calibration-artifact
   fields. `StageBResult` records the corresponding target and operator IDs,
   and `stage_b_model_adequate` requires exact scope equality, presence in the
   reviewed registry, and a calibration-artifact checksum. The production
   registry is empty, the historical criterion has no artifact checksum, and
   therefore the historical `0.3` cannot promote a run. Negative tests reject
   KGAS007, a different ring count, a different reference, an unlabeled scalar,
   and the unregistered historical criterion.

2. **Bootstrap scientific status — satisfied.** Bare counts now yield only
   `bootstrap_arithmetic_pass` and the explicit status
   `arithmetic_pass_evidence_unverified`; `scientific_gate_pass` remains false.
   With no counts, both durable target records remain `pending_bootstrap`.
   The v2 target configurations name one authoritative gate ID instead of
   duplicating its numerical constants, and the runner rejects a mismatched ID.

3. **Mock schema propagation — satisfied.** All active source consumers now use
   `mock_turnover_radius_recovered`. The aggregator validates that both a ran
   kinUV record and a ran KinMS record contain the corrected field before
   rebuilding summaries. The schema test exercises omission from each fitter
   independently. A repository search found the old key only in the decision's
   description of the historical defect.

4. **Ico/Wiener provenance — satisfied.** The production runner hashes the
   companion error map and records the path, reduction rule, scalar
   uncertainty, Ico peak, dimensionless Wiener factor, and relative
   unit-integral convention. The durable metrics contain those fields and input
   hashes for both targets. I recomputed every durable input hash and both
   manifest file hashes successfully. The manifest names the revised exact
   commit, and each non-rotating likelihood identity error is exactly zero.

### Durable accounting and tests

The durable artifact reproduces the prior-free improvements without changing
their interpretation: `31497.370777` for KGAS066 and `4423.044575` for
KGAS007. It records both rotating PA starts, resolved bounds, environment, data
loading metadata, and the audited template factors. KGAS066's non-rotating fit
records pressure at the lower systemic-velocity bound, upper dispersion bound,
and upper declination-offset bound; KGAS007 records the upper dispersion bound.
STATUS presents these as MAP-only diagnostics with bootstrap pending.

The revised focused suite passed (`58 passed, 3 skipped`) and the complete suite
passed (`225 passed, 8 skipped`). Historical prose now identifies the blank
baseline and the dimensionless KGAS007 omega value.

### Residual risks after re-review

The two non-rotating optimizations still use one start and remain
boundary-limited, while both rotating accounting fits place turnover radius at
its lower bound. These facts prohibit physical promotion but do not invalidate
the S0 accounting split. The registered-criterion mapping is intentionally
empty; any later population of it, calibration checksum, or scientific
bootstrap evidence contract needs the fresh decision and review required by
the field guide. The propagated Ico error field remains compressed to one
median Wiener scale and should remain an explicit ablation in later stages.
`PRODUCTION_RECORD.md` still describes KGAS066's old Stage B gate as passed;
that sentence should be annotated as historical under the invalidated gate
before the next documentation promotion.

### Re-review verdict

**Accept.** All four required S0 changes are satisfied at the revised exact
commit, the durable artifact is internally consistent and checksum-verifiable,
and every scientifically incomplete result fails closed. This accepts S0
scientific accounting only. It does not accept a rotation detection, Stage B
smoothness calibration, posterior claim, S1 comparator closure, or production
promotion.

## Final exact-commit recheck

I independently checked the delta from accepted commit
`776256f6b45923e1659a049c220db8e62ed1eee9` to final commit
`fb4a14543d579168c9224c8ebca6a7591147f4db`, without reading or contacting
Reviewer B. The code delta contains only the required-gate name set in
`runner/state.py` and its tests.

`terminal_validation_state` now requires the exact ten-gate production schema
before `verified` is possible. Missing, additional, pending, blocked, unknown,
or incomplete gates return `validation_pending`; an explicit failure in the
complete schema returns `failed`. This closes the omission path without
changing any scientific threshold or prior accepted S0 calculation.

The durable artifact manifest and metrics both name exact commit
`fb4a14543d579168c9224c8ebca6a7591147f4db`. I recomputed the hashes and sizes
of `run_accounting.py` and `metrics.json`, then recomputed every recorded input
hash for both targets; all match. The target results remain
`pending_bootstrap` with `scientific_gate_pass=false`, zero non-rotating
likelihood-identity error, and the same explicit boundary limitations.

Focused gate tests passed (`12 passed`) and the full suite passed
(`225 passed, 8 skipped`).

**Final verdict: accept.** The final gate-name change is fail closed and
preserves all prior S0 science/numerics findings. Acceptance remains limited to
S0 scientific accounting and does not close any later scientific gate.
