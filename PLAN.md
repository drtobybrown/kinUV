# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. MILESTONE-001 provides one immutable accepted bundle for each target. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

The visibility likelihood and samplers contain kinematics only. Mass decomposition, cosmology, and physical-radius inference are downstream analyses and must consume immutable fit products without entering the hot path.

## Priority work

The cross-domain recovery program is complete through S5 under
[`DEC-PI-S4-STANDARD-USE-BENCHMARK`](docs/decisions/DEC-PI-S4-STANDARD-USE-BENCHMARK.md).
The historical S4 failure remains preserved. Reviewer A and Reviewer B accept
the corrected S4 evidence, and the S5 verification seal is immutable.
The additive `crossdomain-recovery-s5-subbeam-20260907` record now supplies the
publication diagnostic for `R_turn < BMAJ`; it reuses the accepted S4 fits and
has no gating role.

The final ApJ production figure suite is complete at code commit `c6822b1`.
Each accepted target bundle has an additive `publication/` directory with the
four required PDF/PNG figure pairs, current-WCS cube centroids, explicit claim
limits, and a checksum-bound manifest. Astra's science-completeness review
accepts both suites. No production fit was rerun.

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
