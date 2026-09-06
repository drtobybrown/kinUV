# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. MILESTONE-001 provides one immutable accepted bundle for each target. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

The visibility likelihood and samplers contain kinematics only. Mass decomposition, cosmology, and physical-radius inference are downstream analyses and must consume immutable fit products without entering the hot path.

## Priority work

1. **Close S1 through proportional verification.** Run the two-target refinement matrix for the cubic-B-spline/analytic-LOSVD adapter and close immediately if every `|Delta chi2| <= 0.1` and relative-L2 gate passes.
2. **Begin S2 geometry work.** Execute the turnover and basin audits that do not depend on missing Measurement Set grouping metadata.
3. **Re-export both targets through `ms2kinuv`.** Neither retained file meets the real-S2 grouping and provenance contract; covariance selection and held-out scoring remain blocked until compliant exports exist.
4. **Complete S2 covariance validation.** Run grouped covariance selection and held-out scoring when compliant exports exist.
5. **Test one-factor candidates.** Execute S3--S4 joint brightness/ring/dispersion ablations, paired mocks, and valid held-out folds.
6. **Calibrate intervals separately.** No NUTS is licensed here; exact-workflow SBC remains required before calibrated interval claims.

## Exit criteria for the next production increment

- S0--S5 MAP/recovery gates pass with dual review and Consultant sign-off; S1 is currently blocked by the recorded renderer-convergence escalation.
- Exact-workflow SBC meets a separately declared coverage criterion only for a calibrated-interval claim.
- MILESTONE-001 remains reproducible from the frozen target configurations and run manifests.
- A clean checkout reproduces MAP identity chi2 and validates retained NUTS summaries from manifests.
- The review board contains only an active card; completed discussion is folded into the production record.
- No runtime code or data extraction depends on legacy repositories; valid archive provenance links remain preserved.
