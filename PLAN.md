# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. MILESTONE-001 provides one immutable accepted bundle for each target. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

The visibility likelihood and samplers contain kinematics only. Mass decomposition, cosmology, and physical-radius inference are downstream analyses and must consume immutable fit products without entering the hot path.

## Priority work

1. **Close S1 review.** Obtain independent science/numerics and software/reproducibility code verdicts on the exact intrinsic-comparator implementation and durable closure dossier.
2. **Execute S2 geometry/covariance.** Run the frozen turnover grid, twelve-start basin audit, and line-free covariance/whitening report without changing target priors or visibility selection.
3. **Test one-factor candidates.** Execute S3--S4 joint brightness/ring/dispersion ablations, paired mocks, and valid held-out folds.
4. **Calibrate intervals separately.** No NUTS is licensed here; exact-workflow SBC remains required before calibrated interval claims.
5. **Re-export KGAS007 through `ms2kinuv`.** Replace the historical export when its source Measurement Set becomes available; until then real grouped holdout remains blocked rather than approximated.

## Exit criteria for the next production increment

- S0--S5 MAP/recovery gates pass with dual review and Consultant sign-off.
- Exact-workflow SBC meets a separately declared coverage criterion only for a calibrated-interval claim.
- MILESTONE-001 remains reproducible from the frozen target configurations and run manifests.
- A clean checkout reproduces MAP identity chi2 and validates retained NUTS summaries from manifests.
- The review board contains only an active card; completed discussion is folded into the production record.
- No runtime code or data extraction depends on legacy repositories; valid archive provenance links remain preserved.
