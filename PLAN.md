# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. MILESTONE-001 provides one immutable accepted bundle for each target. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

The visibility likelihood and samplers contain kinematics only. Mass decomposition, cosmology, and physical-radius inference are downstream analyses and must consume immutable fit products without entering the hot path.

## Priority work

1. **Complete the executable S1/S2 amendment.** Astra freezes the metric algebra, refinement schedule, unit/flux ledger, covariance estimators, finite parameter chart, exact starts, optimizer budgets, and checksum-bound campaign configuration required by both proposal reviewers.
2. **Implement and close S1 after dual acceptance.** Replace nearest-cell and finite Gauss-Hermite rendering with the external cubic-B-spline/analytic-LOSVD continuum adapter, correct the flux-density contract, and pass the unchanged closure gates.
3. **Re-export both targets through `ms2kinuv`.** Neither retained file meets the new real-S2 grouping and provenance contract; covariance selection and held-out scoring remain blocked until compliant exports exist.
4. **Execute and close S2.** Run the frozen geometry/basin/covariance protocol with valid metadata and dual review before any S3 work.
5. **Test one-factor candidates.** Execute S3--S4 joint brightness/ring/dispersion ablations, paired mocks, and valid held-out folds.
6. **Calibrate intervals separately.** No NUTS is licensed here; exact-workflow SBC remains required before calibrated interval claims.

## Exit criteria for the next production increment

- S0--S5 MAP/recovery gates pass with dual review and Consultant sign-off; S1 is currently blocked by the recorded renderer-convergence escalation.
- Exact-workflow SBC meets a separately declared coverage criterion only for a calibrated-interval claim.
- MILESTONE-001 remains reproducible from the frozen target configurations and run manifests.
- A clean checkout reproduces MAP identity chi2 and validates retained NUTS summaries from manifests.
- The review board contains only an active card; completed discussion is folded into the production record.
- No runtime code or data extraction depends on legacy repositories; valid archive provenance links remain preserved.
