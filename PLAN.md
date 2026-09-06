# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. The KGAS066 official MAP is read-only. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

The visibility likelihood and samplers contain kinematics only. Mass decomposition, cosmology, and physical-radius inference are downstream analyses and must consume immutable fit products without entering the hot path.

## Priority work

1. **Validate the canonical KinMS benchmark.** Run the downstream comparator for KGAS066 and KGAS007 in a KinMS environment and retain matched cube, moment, spectrum, PVD, channel-map, and rotation-profile receipts.
2. **Complete the data migration.** Target metadata now lives in `kinuv.targets`; re-export KGAS007 through `ms2kinuv` when its source Measurement Set becomes available so both targets use the same metres-based NPZ schema.
3. **Make artifacts reproducible.** Write compact manifests with input hashes, environment versions, run IDs, parameterization, and output locations. Keep large model cubes and stochastic cloud arrays in scratch or durable run storage rather than Git.
4. **Benchmark the hot path.** Track compiled visibility-likelihood evaluations per second, memory, and scaling with rows, channels, and image grid. Keep GPU disabled until an official-kernel benchmark beats the CPU path.
5. **Calibrate posterior intervals after this refactor closes.** Do not launch NUTS during the refactor. Resume exact-workflow SBC only under a separately licensed campaign.

## Exit criteria for the next production increment

- Exact-workflow SBC meets the declared coverage criterion.
- KGAS066 and KGAS007 run from the same target schema without legacy catalogue imports.
- A clean checkout can reproduce MAP identity chi2 and validate retained NUTS summaries from manifests.
- The review board contains only an active card; completed discussion is folded into the production record.
- No production result depends on an archived artifact path.
