# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. The KGAS066 official MAP is read-only. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

## Priority work

1. **Calibrate posterior intervals.** Run simulation-based calibration through the exact JAX/FINUFFT/NumPyro production path. Publish interval summaries only after the calibration gate passes.
2. **Complete the data migration.** Target metadata now lives in `kinuv.targets`; re-export KGAS007 through `ms2kinuv` when its source Measurement Set becomes available so both targets use the same metres-based NPZ schema.
3. **Make artifacts reproducible.** Write compact manifests with input hashes, environment versions, run IDs, parameterization, and output locations. Keep large model cubes and stochastic cloud arrays in scratch or durable run storage rather than Git.
4. **Harden the runner.** Preserve the four-shard CPU workflow, validate merge provenance, and make interrupted-chain replacement explicit. Keep GPU disabled until an official-kernel benchmark beats the CPU path.
5. **Improve the surface-brightness model carefully.** Use residual diagnostics to propose richer templates with an explicit visibility-chi2 comparison. Preserve the 30 km/s Wiener Ico baseline until a replacement passes that comparison.

## Exit criteria for the next production increment

- Exact-workflow SBC meets the declared coverage criterion.
- KGAS066 and KGAS007 run from the same target schema without legacy catalogue imports.
- A clean checkout can reproduce MAP identity chi2 and validate retained NUTS summaries from manifests.
- The review board contains only an active card; completed discussion is folded into the production record.
- No production result depends on an archived artifact path.
