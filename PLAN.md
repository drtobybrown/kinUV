# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. MILESTONE-001 provides one immutable accepted bundle for each target. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

The visibility likelihood and samplers contain kinematics only. Mass decomposition, cosmology, and physical-radius inference are downstream analyses and must consume immutable fit products without entering the hot path.

## Priority work

The implementation and empirical S4 gates are complete under
[`DEC-PI-S4-STANDARD-USE-BENCHMARK`](docs/decisions/DEC-PI-S4-STANDARD-USE-BENCHMARK.md).
The historical failure remains preserved. The final S4 dossier is frozen for
dual independent review; S5 has not started.

1. **Complete dual S4 review.** Reviewer A assesses the physical comparison, matched-family truth recovery, and interpretation. Reviewer B verifies exact commits, manifests, runner behavior, and the CASA-free boundary.
2. **Close S4 only on dual acceptance.** Record both verdicts without rewriting the historical r3 failure. Any requested correction returns to a new immutable evidence directory.
3. **Seal S5 after S4 closure.** Verify final artifact integrity and production references, then update the production record and milestone registry. No new scientific fit or posterior campaign is part of S5 sealing.
4. **Calibrate intervals separately.** A later authorized campaign applies R-hat and ESS to primary parameters and establishes coverage before calibrated interval claims.

## Exit criteria for the next production increment

- S0--S3 retain historical closure; S4 empirical gates pass and await dual review; S5 has not started.
- Exact-workflow SBC meets a separately declared coverage criterion only for a calibrated-interval claim.
- MILESTONE-001 remains reproducible from the frozen target configurations and run manifests.
- A clean checkout reproduces MAP identity chi2 and validates retained NUTS summaries from manifests.
- The review board contains only an active card; completed discussion is folded into the production record.
- No runtime code or data extraction depends on legacy repositories; valid archive provenance links remain preserved.
