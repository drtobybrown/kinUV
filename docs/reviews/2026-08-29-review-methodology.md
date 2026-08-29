---
role: reviewer
date: 2026-08-29
agent: canfar-066-methodology
canon_generation: 4
ids:
  - DEC-066-TARGET
  - DEC-066-VC
  - DEC-066-OSCMETRIC
  - DEC-066-INFER
  - DEC-066-POL
  - DEC-066-ZEROMODEL
  - DEC-066-SB
  - DEC-066-GRID
  - DEC-066-SPECRESP
  - DEC-066-WEIGHT
  - DEC-066-INC
  - DEC-HIER-SELFUNC
verdict: ack
---

# ACK S1 with four tighter bounds (degeneracy, Hann+bin, Stage A, XX-only \(s\))

**Read after** `AGENTS.md` → field-guide → `STATUS.md` → [`2026-08-29-propose-methodology.md`](2026-08-29-propose-methodology.md). Canon generation 4. No new `DEC-*` id. User is the tie-breaker; this ACK licenses **S1 + leftover \(\chi^2\)** only. Official MAP remains `kinuv-KGAS066-uvsign-map`. No NUTS in this turn.

## Scope

Existing ids only. The propose correctly states that UV fitting is unproven as a scientific advance over image-plane fitting until an injection-and-recovery test shows CLEAN-cube kinematics biased where the \(uv\)-plane model succeeds. That S1 experiment is licensed **with the four modifications below**, which are tighter quantitative bounds on the propose’s executable list, not new ADRs.

## What changed / what was checked

Propose [`2026-08-29-propose-methodology.md`](2026-08-29-propose-methodology.md): mission 10× is vis posteriors + sub-beam \(V_c\) vs CLEAN, not NUFFT vs KinMS clouds. Scorecard: 066 vis MAP is the right likelihood domain; S1/S2 not shown (`r_t` floor, no NUTS). Residual risks 1–4 of that propose are the ones this ACK tightens.

Checked against code, not the propose’s wording:

- Gate 2 [`tests/test_mock_recovery.py`](../../tests/test_mock_recovery.py) still **allows** `native_diagonal`. [`src/kinuv/forward/mocks.py`](../../src/kinuv/forward/mocks.py) imports `kinuv.likelihood.hann_then_bin`, which does not exist, so the mock never applies DEC-066-SPECRESP. Production MAP uses [`kinuv.response.spectral.hann_then_bin`](../../src/kinuv/response/spectral.py) via `predict_binned`.
- Stage A L-BFGS box [`RT_BOUNDS_ARCSEC = (0.5, 15)`](../../src/kinuv/infer/seeds.py) would pin a sub-beam inject to the 0.5″ floor. S1 recovery must use a **script-local** \(r_t\) box; production bounds stay.
- [`DEC-066-INC`](../decisions/DEC-066-INC.md) freezes \(i\). The thin-disk forward model has **no \(h_z\)**. S1 cannot co-fit the \(i\)–\(h_z\)–\(\sigma_0\) triad; it must record the freeze and still report PA–\(\sigma\) / PA–inner-slope covariance, plus a diagnostic \((\sigma_0, i)\) \(\chi^2\) slice.
- Empirical \(s\) is XX-only ([`DEC-066-POL`](../decisions/DEC-066-POL.md) unused). Any later NUTS interval that assumed Stokes \(I\) or XX+YY would be too narrow by \(\sim\sqrt{2}\).

## Attacks (tighter bounds)

1. **Degeneracy, not only 1-D curves.** Injecting \(r_t \ll \mathrm{BMAJ}\) with low \(\sigma\) tests beam smearing. It does not test the ungridded failure mode \(i\)–\(h_z\)–\(\sigma_0\). **Bound:** S1 must record `i_held_fixed` and `h_z_in_model: false`. Official recovery does not co-fit \(i\). Report 2-D covariance of \(\sigma_0\) with PA (fitted) and a diagnostic \((\sigma_0, i)\) slice with \(i\) unfrozen for the scan only, so sub-beam \(\mathrm{d}V/\mathrm{d}r\) is not a silent trade against unconstrained inclination.
2. **Hann+bin or the likelihood is invalid.** \(\chi^2\) on `native_diagonal` overstates precision. **Bound:** `assert` the pipeline operator is `kinuv.response.spectral.hann_then_bin` (`"Hann+bin"`) in [`tests/test_mock_recovery.py`](../../tests/test_mock_recovery.py) **before** generating synthetic visibilities. S1 inject uses `load_kgas066` + `predict_binned` (881×95, \(N=4\)). Do not Hann strided mock channels and call that SPECRESP.
3. **Stage A first; no \(\lambda=0\) rings.** Stage B at \(\lambda=0\) (`max_omega≈10.2`) is the OSCMETRIC conflict. Unregularised rings can oscillate into noise or SB misspecification and contaminate the vis-vs-CLEAN table. **Bound:** S1 benchmarks Stage A (parametric arctan) only. Residual-\(\Omega\) vs Stage A is a **user DEC stub**, not this wave.
4. **XX-only \(s\).** **Bound:** S1 (and any later Stage A NUTS) must use the empirical XX `VisData.s` already in \(\chi^2 = s \sum w |d-m|^2\). Artifact must state `pol: "XX"` and that Stokes \(I\) / XX+YY is not assumed. Do **not** run NUTS until S1 recovers ([DEC-066-INFER](../decisions/DEC-066-INFER.md)).

## Residual risks

1. S1 may **fail**: visibilities may not constrain inner \(\mathrm{d}V/\mathrm{d}r\) inside the Ico beam even with the correct kernel. Then UV fitting is not yet better science than CLEAN.
2. \(h_z\) is not a parameter, so the full \(i\)–\(h_z\)–\(\sigma_0\) triad cannot be co-fitted. A \((\sigma_0, i)\) slice is a diagnostic, not a 3-D posterior.
3. Residual-\(\Omega\) (excess vs Stage A, not absolute \(\Omega\)) remains a user stub. Do not reopen the OSCMETRIC \(\lambda\)-grid.
4. XX-only leaves \(\sqrt{2}\) on the table and will bias any “exact” posterior width until POL is used.
5. Frozen Wiener Ico cannot represent spirals; leftover vis \(\chi^2\) can look kinematic when it is SB.
6. Opening \(N\approx 400\), baryons, warps, GPU, or NUTS without S1 recovery forks the architecture.
7. Carry-forward: [`DEC-HIER-SELFUNC`](../decisions/DEC-HIER-SELFUNC.md) deferred; do not quote \(\gamma\).

## STATUS updates required

- `next_role: proposer` (S1 + leftover \(\chi^2\) licensed with the four bounds)
- `pending: []`
- `last_review:` this file
- Official product unchanged: `kinuv-KGAS066-uvsign-map`
- No NUTS this turn
- `deadlocks:` none
