# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. MILESTONE-001 provides one immutable accepted bundle for each target. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

The visibility likelihood and samplers contain kinematics only. Mass decomposition, cosmology, and physical-radius inference are downstream analyses and must consume immutable fit products without entering the hot path.

## Priority work

The active, PI-authorized pre-meeting campaign is defined by
[`COLLABORATOR_DELIVERY_PLAN_20260908`](docs/operations/COLLABORATOR_DELIVERY_PLAN_20260908.md).
Both four-start joint visibility MAP sweeps are complete. Reviewer A and
Reviewer B independently accepted the MAP-only candidate; its best products are
installed directly under each `results/production/<TARGET>/` root. KGAS007's
four-chain conditional posterior is accepted and installed in its canonical
`best_model/posterior/` directory.
The pathological KGAS066 start was diagnosed and preserved; its replacements
run in four independent flexible headless sessions from commit `8f11648`, one
per chain. The first oversubscribed launch and the superseded unbounded-jitter
launch remain explicit evidence. Archived candidate history remains available
while KGAS066 posterior sampling proceeds.

The cross-domain recovery program is complete through S5 under
[`DEC-PI-S4-STANDARD-USE-BENCHMARK`](docs/decisions/DEC-PI-S4-STANDARD-USE-BENCHMARK.md).
The historical S4 failure remains preserved. Reviewer A and Reviewer B accept
the corrected S4 evidence, and the S5 verification seal is immutable.
The additive `crossdomain-recovery-s5-subbeam-20260907` record now supplies the
publication diagnostic for `R_turn < BMAJ`; it reuses the accepted S4 fits and
has no gating role.

The canonical production hierarchy is `results/production/<target>/` with
`best_model/`, `plots/`, and `benchmarks/`. Diagnostic repair `fdcbf165cdd8b1f9b6d89d71cfd5353ce4ac9e7c`
corrects celestial slit orientation and selects the exact S4-bound smooth-S3
checkpoints, instead of mixing historical cubes and posteriors. Both target
suites have four direct and five comparator PDF/PNG pairs; no corresponding
posterior exists. S0--S5 gates and accepted evidence are unchanged. No new fit
or score was run. See STATUS and PRODUCTION_RECORD for verified archive and
product identities and Astra's science-completeness acceptance.

1. **Calibrate rotation significance.** Complete the fitted non-rotating emitting-disk bootstrap before promoting a formal rotation-detection probability.
2. **Calibrate intervals separately.** A later authorized campaign applies R-hat and ESS to primary parameters and establishes exact-workflow coverage before calibrated interval claims.
3. **Broaden the recovery frontier.** Test warped, lopsided, thick, noncircular, and radially varying dispersion truths under fixed evaluation samples. Preserve the accepted thin axisymmetric arctan result as the baseline.
4. **Prepare survey expansion.** Define a target-selection and resource contract before adding more galaxies or a population runner.

## Exit criteria for the next production increment

- S0--S5 are closed with dual review, manifest verification, and the PI-authorized standard-use comparison.
- Exact-workflow SBC meets a separately declared coverage criterion only for a calibrated-interval claim.
- MILESTONE-001 remains reproducible from the frozen target configurations and run manifests.
- A clean checkout reproduces MAP identity chi2 and validates retained NUTS summaries from manifests.
- The review board contains only an active card; completed discussion is folded into the production record.
- No runtime code or data extraction depends on legacy repositories; valid archive provenance links remain preserved.
