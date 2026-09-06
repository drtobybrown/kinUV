# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. MILESTONE-001 provides one immutable accepted bundle for each target. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

The visibility likelihood and samplers contain kinematics only. Mass decomposition, cosmology, and physical-radius inference are downstream analyses and must consume immutable fit products without entering the hot path.

## Priority work

1. **Calibrate posterior intervals.** Run exact-workflow simulation-based calibration before treating retained NUTS quantiles as credible intervals.
2. **Re-export KGAS007 through `ms2kinuv`.** Replace its retained historical wavelength-coordinate NPZ when the source Measurement Set becomes available.
3. **Benchmark the hot path.** Track compiled visibility-likelihood evaluations per second, memory, and scaling with rows, channels, and image grid. Keep GPU disabled until an official-kernel benchmark beats the CPU path.
4. **Resolve model adequacy.** Investigate KGAS066 velocity-structured residuals and define an outer-ring support criterion before revisiting KGAS007 Stage B.

## Exit criteria for the next production increment

- Exact-workflow SBC meets the declared coverage criterion.
- MILESTONE-001 remains reproducible from the frozen target configurations and run manifests.
- A clean checkout reproduces MAP identity chi2 and validates retained NUTS summaries from manifests.
- The review board contains only an active card; completed discussion is folded into the production record.
- No production result depends on an archived artifact path.
