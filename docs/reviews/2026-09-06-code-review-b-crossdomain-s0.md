---
role: reviewer
seat: software-reproducibility
phase: implementation
date: 2026-09-06
reviewer: s0-review-b
canon_generation: 9
campaign_id: crossdomain-recovery-s0
proposal: docs/decisions/DEC-KINUV-CROSSDOMAIN-RECOVERY.md
reviewed_commit: fb4a14543d579168c9224c8ebca6a7591147f4db
verdict: accept
---
# Independent code review B: cross-domain recovery S0

I reviewed commit `7471cd7154b9a9ec5edb238724bd769b239f49bf`
without reading or contacting Reviewer A. This review covers Stage S0 only; it
does not assess or close Stage S1.

## Attempted falsification

I traced new likelihood, null-fit, omega, template, runner, configuration, and
serialization paths; checked legacy replay and standalone imports; and ran the
full CPU suite in `/scratch/kinuv-thbrown/s0-validation`:

- focused S0/architecture suite: `62 passed, 3 skipped`;
- full suite: `221 passed, 8 skipped` in 116.28 seconds;
- three official KGAS066 likelihood-identity replay tests: `3 passed`;
- direct import with `uvkin`, `uvfit`, `ms2kinuv`, `casatasks`, `casacore`, and
  `pyuvdata` blocked at import resolution: passed;
- `python3 -m compileall -q src scripts tests`: passed.

The strongest falsification succeeded against omega provenance. A synthetic
Stage B record with `max_omega_dimensionless=10.18` was accepted after passing
an arbitrary `OmegaCriterion(20.0, "unregistered arbitrary scope", 2)`. The
class therefore labels provenance but does not enforce it.

I also followed the production state machine with both newly blocked gates.
The runner records them as blocked or pending, then writes `SUCCEEDED` and
`VERIFIED`, exits zero, and prints `status: verified` because its blocking list
contains only gates whose status is exactly `fail`.

## Findings

1. **Required — Omega provenance is descriptive rather than enforced.**
   [`stage_b_model_adequate`](../../src/kinuv/infer/stage_b.py) checks only that
   its second argument is an `OmegaCriterion`; it does not bind the criterion
   to the evaluated target, model family, residual-versus-absolute formula,
   channel operator, ring count, calibration artifact, or result record. Any
   caller can construct a permissive criterion and pass the gate. The existing
   unit test confirms only the field name and Python type. Add machine-readable
   calibration identity to both criterion and evaluated result, validate exact
   scope equality, and reject criteria that are not backed by a registered
   calibration artifact/checksum. The historical criterion must be applicable
   only to its verified calibration regime. Also reconcile its stated
   `fixed 7-ring setup` scope with the historical campaign code, which
   recursively evaluates 6--8 rings.

2. **Required — The production runner reports verification despite blocking
   scientific gates.** In
   [`run_production_milestone.py`](../../scripts/run_production_milestone.py),
   the rotation test is `pending_bootstrap` and Stage B is
   `blocked_pending_mock_calibration`, but only literal `fail` states enter
   `blocking`. The runner then records `SUCCEEDED` and `VERIFIED`, returns zero,
   and prints `verified`. Make gate-state handling explicit and closed: blocked,
   pending, uncalibrated, or unknown states must never produce `VERIFIED` or a
   verification success message. Preserve a non-promotion terminal state and
   make the command outcome machine-detectable.

3. **Required — The runner reuses a posterior from a different forward-model
   likelihood.** S0 changed `load_sb_template` from the legacy
   `0.02 * peak` Wiener choice to a propagated uncertainty map. The S0 evidence
   itself shows that frozen-parameter chi-square changes under this template.
   Nevertheless, `_consolidate_posterior` copies historical draws and records
   that the mathematical likelihood is unchanged. Those draws are not samples
   from the refitted S0 model. Do not consolidate them as a current posterior
   or report their R-hat/ESS beside the new fit. Keep them linked as an
   incompatible historical artifact with their original configuration and
   likelihood identity until a separately licensed sampling campaign exists.

4. **Required — S0 run evidence is ephemeral and lacks a reproducibility
   manifest.** [`STATUS.md`](../architecture/STATUS.md) points to
   `/scratch/kinuv-thbrown/s0-validation/s0-initial-metrics.json`. Scratch is an
   explicitly disposable tier, and that JSON does not record the code commit,
   dirty state, commands, environment, input/config checksums, optimizer
   bounds, both start records, or backend. Preserve a compact durable S0
   artifact or immutable `/arc` manifest with those fields and its checksum,
   then link it from STATUS. Boundary pressure already reported for both null
   fits makes this provenance necessary.

5. **Required — Configuration and output schemas changed without a versioned
   contract, and gate constants have two unsynchronized sources.** Both target
   files retain `kinuv-production-target-v1` while replacing acceptance keys;
   `kinuv-milestone-summary-v1` likewise gains different gate semantics and
   Stage B field names. The runner ignores the configured non-rotation delta,
   trial-count, and p-value limits because `rotation_test_metrics` uses module
   constants. Choose one frozen campaign-level source, validate it against the
   accepted decision, and version the changed config/summary/checkpoint
   schemas. Add migration or explicit rejection tests for old records. This
   also removes the inaccurate STATUS statement that the run used “unchanged
   target configurations.”

6. **Advisory — Finish the terminology migration in active code and tests.**
   The new likelihood functions are clear, and compatibility aliases are
   reasonable for sealed products. Active test names and several docstrings
   still call `chi2_zero` a `V=0` model, including the Stage B module header.
   Mark historical fields as blank-baseline compatibility fields everywhere a
   new run can surface them, so future automation cannot recreate the original
   ambiguity.

7. **Advisory — Record optimizer diagnostics for the fitted null.**
   `NonRotatingResult` records status and evaluations but not initial values,
   resolved bounds, final projected gradient, bound-pressure flags, backend,
   or an independently recomputed likelihood identity. Add these to the run
   artifact even if they remain outside the compact public API.

## Gate assessment

The blank/non-rotating likelihood separation is implemented with explicit
names, and the non-rotating fitter reuses `infer.map.predict_binned`, which
preserves the visibility sampling and Hann/bin response path. The plus-one
bootstrap accounting correctly remains pending when no bootstrap evidence is
supplied. The omega formula is now documented and tested as dimensionless, and
legacy KGAS066 likelihood replay passes.

S0 cannot close at this commit. Omega provenance can be bypassed, the runner's
state machine promotes blocked evidence to `VERIFIED`, and the durable evidence
needed to reproduce the reported metrics is absent. The null-bootstrap gate is
properly unexecuted rather than failed; no rotation promotion is supported.
S1 closure thresholds are outside this review and remain unevaluated.

## Residual risks

- The fitted non-rotating solutions press dispersion bounds for both targets
  and a center bound for KGAS066, so optimizer success alone does not establish
  an adequate null.
- The current null fit still uses the diagonal weight model. The required
  covariance-aware complete-refit bootstrap remains future work.
- Renaming `StageBResult.max_omega` and serialized keys breaks old constructor
  and JSON consumers despite an in-memory read property; schema migration must
  be explicit.
- The production runner has no focused state-machine test that enumerates all
  allowed gate states.

## Verdict rationale

**Accept with required changes.** The core S0 direction is sound, standalone,
and well covered by unit and identity tests. The required defects are routine
implementation and provenance repairs within the accepted specification; they
do not require a new physical model or relaxed gate. They do prevent Reviewer B
from accepting this exact commit or closing S0. A revised exact commit must be
reviewed after the five required findings are resolved.

## Re-review of revised commit 776256f6b45923e1659a049c220db8e62ed1eee9

This section preserves the initial review above and records an independent
re-review of the revised exact commit. I again did not read or contact Reviewer
A.

### Verification performed

- Focused S0, architecture, runner-state, and schema suite: `66 passed, 3
  skipped`.
- Full CPU suite: `225 passed, 8 skipped` in 113.71 seconds.
- Official KGAS066 legacy likelihood replay: `3 passed`.
- Artifact manifest hashes for `run_accounting.py` and `metrics.json`: both
  match.
- Every recorded target config and scientific-input hash: present on disk and
  matches.
- Artifact commit identity: exact revised commit, branch `dev`, tracked tree
  clean; the two independent review files are explicitly recorded as untracked.
- Both targets record two rotating starts, resolved bounds, environment/backend,
  template/error-map provenance, null-fit bound pressure, and an exact
  non-rotating likelihood identity.

### Resolution of initial required findings

1. **Omega provenance — resolved for S0.** An unregistered criterion, the
   historical criterion without an artifact checksum, and scope mismatches now
   return false. Stage B results carry target, ring-count, omega-definition,
   reference, and spectral-response identities. The historical replay is
   restricted to seven rings, and the production registry remains empty until
   a later reviewed calibration is registered. The registry is a mutable module
   dictionary, so an in-process monkeypatch can alter it; this is an advisory
   hardening opportunity rather than an S0 promotion path because trusted code
   and commit identity define the executable boundary.

2. **Closed runner terminal behavior — partially resolved, one required defect
   remains.** The actual production runner now reduces its complete gate record
   to `validation_pending`, writes no `VERIFIED` transition, prints the pending
   state, and exits 2. Failed gates remain machine-detectable. However,
   `terminal_validation_state({})` returns `verified`, as does a record
   containing only `preflight: pass`. The helper therefore remains open to
   omission even though its docstring says omitted states close toward pending.
   Add a required-gate identity set or at minimum reject an empty/incomplete
   gate mapping, and add tests for both cases. The state-transition vocabulary
   should also be reconciled with the field guide's declared state machine or
   explicitly versioned as an accepted extension.

3. **Historical posterior incompatibility — resolved.** The runner no longer
   copies or summarizes historical draws as current samples. It writes a
   checksum-bearing provenance reference, marks the likelihood incompatible,
   reports no current R-hat/ESS, and blocks the posterior gate.

4. **Durable S0 evidence — resolved.** The record at
   `/arc/projects/KILOGAS/analysis/toby_sandbox/results/validation/crossdomain-recovery-s0-20260906/`
   contains a reproducible script, detailed metrics, and a verified manifest.
   It records the exact commit, config/input checksums, environment, backend,
   template construction, optimizer settings and bounds, both starts, null
   identity, and bound pressure. STATUS now links this durable record rather
   than scratch.

5. **Schema and gate authority — resolved.** Target config, run manifest,
   summary, and omega-campaign schemas are versioned at v2 where changed. Target
   configs name one registered rotation-gate identity instead of duplicating
   thresholds. `_validate_target_config` rejects v1 and unknown gate IDs, and
   the validation module is the single numeric authority for the accepted
   rotation arithmetic.

### Re-review gate assessment and residual risks

The revised implementation satisfies the S0 scientific-accounting objectives,
preserves exact legacy likelihood replay, keeps historical posteriors separate,
and supplies durable reproducibility evidence. The pending bootstrap and
smoothness calibration remain correctly blocked and are not defects in this
stage.

One software gate remains open: the supposedly closed gate reducer verifies
empty or incomplete gate maps. This is directly testable and can be repaired
without changing physics, priors, covariance, or quantitative gates. After
that repair, the residual advisory risks are the mutable in-process omega
registry, null-fit boundary pressure, absent covariance-aware bootstrap, and
the need to document any v2 state-machine vocabulary added beyond the field
guide.

### Re-review verdict

**Accept with required changes.** Four of the five original required findings
are fully resolved, and the fifth is resolved for the runner's current complete
gate dictionary. I cannot issue `accept` while the shared terminal-state helper
returns `verified` for empty or incomplete evidence. A focused fix and test for
required gate presence is sufficient for another exact-commit re-review.

## Final recheck of commit fb4a14543d579168c9224c8ebca6a7591147f4db

I verified the focused terminal-state repair without reading or contacting
Reviewer A. `REQUIRED_GATE_NAMES` now defines the complete production gate set,
and `terminal_validation_state` returns `validation_pending` unless the supplied
mapping has exactly that set. Direct checks produced:

- empty gate map: `validation_pending`;
- `preflight: pass` only: `validation_pending`;
- complete passing map: `verified`;
- complete map plus an unknown extra gate: `validation_pending`.

The focused runner/schema/scientific tests pass (`11 passed`), and the full CPU
suite passes (`225 passed, 8 skipped` in 116.70 seconds). The durable artifact
at
`/arc/projects/KILOGAS/analysis/toby_sandbox/results/validation/crossdomain-recovery-s0-20260906/`
names exact commit `fb4a14543d579168c9224c8ebca6a7591147f4db` in both its manifest and metrics.
Its `run_accounting.py` and `metrics.json` sizes and SHA-256 hashes match the
manifest; both target configuration hashes and every recorded scientific input
hash also match. Each target retains two starts and zero non-rotating
likelihood-identity error.

The last required finding is resolved. Remaining risks are advisory and already
recorded above: bootstrap and a newly registered smoothness calibration remain
future blocked work, the current null optima press declared bounds, and the
in-process omega registry could be made immutable as defense in depth.

**Final verdict: accept.** Commit
`fb4a14543d579168c9224c8ebca6a7591147f4db` satisfies Reviewer B's Stage S0
software and reproducibility requirements. This verdict closes Reviewer B's S0
seat only; it does not assess S1 or constitute scientific promotion.
