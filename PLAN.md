# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. MILESTONE-001 provides one immutable accepted bundle for each target. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

The visibility likelihood and samplers contain kinematics only. Mass decomposition, cosmology, and physical-radius inference are downstream analyses and must consume immutable fit products without entering the hot path.

## Priority work

1. **Resolve the S4 architectural escalation.** The checksum-bound `r3` benchmark fails both PI gates for KGAS066 and leaves the favorable KGAS007 velocity diagnostic ineligible with only two common radial bins. Astra must diagnose the cross-domain mismatch and set the next physical/model-selection direction from `docs/decisions/ESC-KINUV-S4-CROSSDOMAIN-SUPERIORITY.md`.
2. **Keep S5 closed.** Do not promote, retune target parameters, relax the registered S4 gates, or begin S5 until a new Astra directive resolves the explicit stop condition.
3. **Preserve and tighten the S4 evidence.** Retain the exact implementation commit `5d08507`, the 23-file manifest, the common-domain products, and both independent failed-gate reviews as the baseline for the next recovery design. Bind the covariance input path and hash directly before structured residual diagnostics are reused in a later promotion record.
4. **Calibrate intervals separately.** Any later posterior campaign applies R-hat and ESS promotion gates only to primary kinematic and geometric parameters; exact-workflow SBC remains required before calibrated interval claims.

## Exit criteria for the next production increment

- S0--S5 MAP/recovery gates pass with dual review and Consultant sign-off; S1--S3 are closed, S4 is failed and escalated, and S5 has not started.
- Exact-workflow SBC meets a separately declared coverage criterion only for a calibrated-interval claim.
- MILESTONE-001 remains reproducible from the frozen target configurations and run manifests.
- A clean checkout reproduces MAP identity chi2 and validates retained NUTS summaries from manifests.
- The review board contains only an active card; completed discussion is folded into the production record.
- No runtime code or data extraction depends on legacy repositories; valid archive provenance links remain preserved.
