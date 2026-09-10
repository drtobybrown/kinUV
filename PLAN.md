# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. MILESTONE-001 provides one immutable accepted bundle for each target. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

The visibility likelihood and samplers contain kinematics only. Mass decomposition, cosmology, and physical-radius inference are downstream analyses and must consume immutable fit products without entering the hot path.

## Priority work

The PI licensed the unified architecture and execution sequence in
[`DEC-KINUV-UNIFIED-FOUNDATION`](docs/decisions/DEC-KINUV-UNIFIED-FOUNDATION.md).
The carrier-free common model, targeted four-start MAP campaign, corrected PVD
reporting transform, and production packaging are complete for both targets.
Commit `c91396b16e70c407c5bfa57d72e619bda8efdd8e` reproduces the packaging path;
the active products live directly under `results/production/<TARGET>/` and
their manifests are recorded in `STATUS.md`.

The fair same-prior sampler comparison is active on the identifiable unified
chart. A bounded NUTS probe reached the depth-8 cap and completed only 10
warm-up transitions in 20 minutes, so the declared Dynesty fallback now runs
as two independent flexible CANFAR sessions per target from commit `cca9b7f`.
Do not inherit old arctan/ring posterior draws. Promote only weighted posterior
products whose independent replicates pass effective-sample, evidence,
parameter, and radial-profile agreement gates.

The new Phase-4 mock and real-data evidence is complete. Across smooth,
non-monotonic, and varying-dispersion truths, unified kinUV's inner-beam
projected-speed RMSE is 23.8--25.5 percent of KinMS and smooth-control absolute
errors remain below one percent of projected amplitude. Genuine grouped
training-refit prediction has positive one-sided 95-percent lower bounds on
both real targets. Phase 5 will stage the accepted weighted posterior beside
the MAP, retain MAP as the selected point model unless posterior-median native
visibility scoring justifies replacement, render matching credible bands, and
rehash the complete production trees.

The active, PI-authorized pre-meeting campaign is defined by
[`COLLABORATOR_DELIVERY_PLAN_20260908`](docs/operations/COLLABORATOR_DELIVERY_PLAN_20260908.md).
The former target-specific collaborator MAP and NUTS products are archived.
Both current `best_model/selection.json` records select the unified MAP and both
`nuts/status.json` records explicitly say that no posterior has been run for
this chart. Each target exposes the complete four-start `map/` record,
`best_model/`, `nuts/`, `plots/`, `benchmarks/`, and provenance. Historical
posteriors must not be mixed with these fitted profiles.

The cross-domain recovery program is complete through S5 under
[`DEC-PI-S4-STANDARD-USE-BENCHMARK`](docs/decisions/DEC-PI-S4-STANDARD-USE-BENCHMARK.md).
The historical S4 failure remains preserved. Reviewer A and Reviewer B accept
the corrected S4 evidence, and the S5 verification seal is immutable.
The additive `crossdomain-recovery-s5-subbeam-20260907` record now supplies the
publication diagnostic for `R_turn < BMAJ`; it reuses the accepted S4 fits and
has no gating role.

The canonical production hierarchy is `results/production/<target>/` with
`best_model/`, `map/`, `nuts/`, `plots/`, `benchmarks/`, and `provenance/`.
Diagnostic repair `fdcbf165cdd8b1f9b6d89d71cfd5353ce4ac9e7c`
corrects celestial slit orientation and selects the exact S4-bound smooth-S3
checkpoints, instead of mixing historical cubes and posteriors. Both target
suites now include stage-labelled matched moments, kinematic-overlay PVDs,
full-disk and central-beam spectra, two-panel rotation/dispersion profiles, and
truth-annotated synthetic recovery. KGAS007 adds its accepted posterior corner
and 16th--84th percentile radial bands. No fit or synthetic score was rerun.

1. **Calibrate rotation significance.** Complete the fitted non-rotating emitting-disk bootstrap before promoting a formal rotation-detection probability.
2. **Calibrate intervals separately.** A later authorized campaign applies R-hat and ESS to primary parameters and establishes exact-workflow coverage before calibrated interval claims.
3. **Broaden the recovery frontier.** Test warped, lopsided, thick, noncircular, and radially varying dispersion truths under fixed evaluation samples. Preserve the accepted thin axisymmetric arctan result as the baseline.
4. **Prepare survey expansion.** Define a target-selection and resource contract before adding more galaxies or a population runner.
5. **Unify without forcing adequacy.** Preserve one schema and explicit unresolved/noncircular flags across the survey; do not introduce target-specific model fallbacks or force an accepted measurement when the common thin circular model is inadequate.

## Exit criteria for the next production increment

- S0--S5 are closed with dual review, manifest verification, and the PI-authorized standard-use comparison.
- Exact-workflow SBC meets a separately declared coverage criterion only for a calibrated-interval claim.
- MILESTONE-001 remains reproducible from the frozen target configurations and run manifests.
- A clean checkout reproduces MAP identity chi2 and validates retained NUTS summaries from manifests.
- The review board contains only an active card; completed discussion is folded into the production record.
- No runtime code or data extraction depends on legacy repositories; valid archive provenance links remain preserved.
