# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. MILESTONE-001 provides one immutable accepted bundle for each target. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

The visibility likelihood and samplers contain kinematics only. Mass decomposition, cosmology, and physical-radius inference are downstream analyses and must consume immutable fit products without entering the hot path.

## Priority work

1. **Execute S3 one-factor candidates.** S2 closed at `8248a05` with dual acceptance. Test joint emissivity, supported projected-velocity knots, and only then a two-zone dispersion model while retaining joint nuisance optimization.
2. **Close S3 through independent review.** Freeze the exact implementation commit and checksum-bound ablation dossier, obtain Reviewer A and Reviewer B verdicts, and record the tally.
3. **Execute S4 paired selection.** Apply the PI-authorized aggregate gates: at least 10 percent lower projected-velocity RMSE than frozen stock KinMS and improved overall reduced chi-square on both real targets. Include baseline, antenna, time, and real/imaginary residual diagnostics.
4. **Seal S5 evidence.** Verify exact manifests, clean-checkout reproducibility, review tallies, and the accepted production record without launching an unlicensed posterior campaign.
5. **Calibrate intervals separately.** Any later posterior campaign applies R-hat and ESS promotion gates only to primary kinematic and geometric parameters; exact-workflow SBC remains required before calibrated interval claims.

## Exit criteria for the next production increment

- S0--S5 MAP/recovery gates pass with dual review and Consultant sign-off; S1 and S2 are closed and S3 is active.
- Exact-workflow SBC meets a separately declared coverage criterion only for a calibrated-interval claim.
- MILESTONE-001 remains reproducible from the frozen target configurations and run manifests.
- A clean checkout reproduces MAP identity chi2 and validates retained NUTS summaries from manifests.
- The review board contains only an active card; completed discussion is folded into the production record.
- No runtime code or data extraction depends on legacy repositories; valid archive provenance links remain preserved.
