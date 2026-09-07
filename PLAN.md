# kinUV active plan

Current state and measurements are in [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Scientific interpretation and closed experiments are in [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md). Binding implementation choices remain in [`docs/decisions/`](docs/decisions/).

## Scope

The production target set is KGAS066 plus KGAS007. MILESTONE-001 provides one immutable accepted bundle for each target. G4, a survey-scale dispatcher, and hierarchical population inference are outside the current licensed scope.

The visibility likelihood and samplers contain kinematics only. Mass decomposition, cosmology, and physical-radius inference are downstream analyses and must consume immutable fit products without entering the hot path.

## Priority work

The implementation handoff is
[`DEC-KINUV-S4-SCIENTIFIC-RECOVERY`](docs/decisions/DEC-KINUV-S4-SCIENTIFIC-RECOVERY.md).
Astra has resolved the architectural stop and licensed Sol's bounded recovery;
the historical S4 failure remains preserved and S5 is not yet eligible.

1. **Repair diagnostic accounting first.** Resolve sky handedness, WCS registration, PB frequency inversion, absolute velocity conventions, and measured spectral response. Re-score all retained S3 ancestors for both targets without changing their fitted parameters. This separates an inherited mismatch from an ablation-induced regression.
2. **Fit uncertain morphology jointly if needed.** Keep the intrinsic brightness transform fixed during diagnostic repair. If corrected replay or predictive checks still implicate it, replace rigid radial reweighting with a smooth positive representation and consistent brightness/velocity geometry. Retain projected velocity u(r), assess the unresolved inner rise and existing dispersion gradient, and constrain unsupported outskirts through declared smooth regularization rather than a target-specific cutoff or fixed inclination. Skip unnecessary expansion if the repaired model passes.
3. **Select on independent prediction.** Build training-only morphology and KinMS imaging products inside intact MS-group partitions. Choose complexity within training data and evaluate both methods on untouched visibilities with the same covariance. Bind the covariance record and report support/geometry sensitivity.
4. **Verify the scientific S4 gates.** Require at least 10 percent lower known-truth projected-velocity RMSE for each target sampling and improved real held-out prediction, including the uncertainty and KGAS066 non-regression requirements in the handoff. Retain cube, PVD, spectra, and flux diagnostics; masked moment-1 agreement alone cannot certify intrinsic recovery. Sol owns execution budgets and dual independent implementation/evidence reviews.
5. **Proceed to S5 only after fresh S4 acceptance.** Verify artifacts and reproducibility, then present the production record for Astra's scientific sign-off. Preserve `5d08507`, its 23-file manifest, and both historical reviews; do not relabel the old failed attempt.
6. **Calibrate intervals separately.** No new posterior campaign is licensed by this handoff. A later authorized campaign applies R-hat and ESS to primary parameters and establishes coverage before calibrated interval claims.

## Exit criteria for the next production increment

- S0--S5 MAP/recovery gates pass with dual review and Consultant sign-off; S1--S3 retain historical closure, S4 recovery is licensed under the revised scientific contract, and S5 has not started.
- Exact-workflow SBC meets a separately declared coverage criterion only for a calibrated-interval claim.
- MILESTONE-001 remains reproducible from the frozen target configurations and run manifests.
- A clean checkout reproduces MAP identity chi2 and validates retained NUTS summaries from manifests.
- The review board contains only an active card; completed discussion is folded into the production record.
- No runtime code or data extraction depends on legacy repositories; valid archive provenance links remain preserved.
