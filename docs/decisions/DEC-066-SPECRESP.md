---
id: DEC-066-SPECRESP
status: accepted
generation: 3
date: 2026-08-18
owner: 066-6-likelihood
supersedes: generation 2 (no spectral guard channels; edge Hann truncated)
---
# Spectral response function handling

**Question:** How should the ALMA Hanning spectral response be handled in the forward model and likelihood?

## Answer

Two separate treatments. Do not conflate them.

1. **Signal:** the model is Hanning-smoothed on the **native MS channel grid**, then binned with the same operator as the data.
2. **Noise:** Phase 1 uses a **single empirical** per-SPW scale `s` measured on the npz that will be fit. The theoretical factor `12/29` is a synthetic-test result, not a 066 weight multiplier.

## Signal path (mandatory)

ALMA's default correlator window is Hann. On native channels that is convolution with `[0.25, 0.5, 0.25]`. Software spectral binning (`uvkin.bin_channels`) then takes a weighted mean of N adjacent channels and sums the weights.

**066 implementation:**

1. Evaluate the line profile on the native channel grid **plus at least one guard channel** on each spectral end (kernel `[0.25, 0.5, 0.25]` needs one neighbour). If the parent MS still has those channels, use them. If the npz is already trimmed, evaluate the model at `ν_edge ± Δν_native` — do **not** zero-pad visibilities (that biases the outer bin if line wings remain). The YAML `vel_buffer_kms ≈ 100` is an upper bound, not a requirement, once one native guard exists.
2. Convolve along that extended native axis with `[0.25, 0.5, 0.25]`.
3. Bin with the same `N` and the same weighted-mean / weight-sum as the data npz.
4. Trim guard bins so the model spectral axis matches the data npz.

**Forbidden:** convolving `[0.25, 0.5, 0.25]` along an already-binned axis (e.g. 125 channels). That applies Hann at ~5–10 km/s instead of ~1.27 km/s and biases `gas_sigma` and the inner curve. **Forbidden:** Hann on a finite native axis with implicit zeros at the ends.

`N` and `Δv` are properties of the npz being fit, not constants. Record `(n_row, n_chan, Δv_kms, N)` in 066-0. Replica forensics used 881×125 at bin-4; YAML default is bin-8. Prefer the replica window so Δχ² is comparable.

## Noise path (Phase 1)

Cycle 3+ CASA `WEIGHT` already uses `EFFECTIVE_BW` (Hanning equivalent noise bandwidth). The forensic line-free statistic on KGAS066 was `⟨w|V|²⟩ = 2.42` (expect 2.0): **21% optimistic**, not the `1/0.414 ≈ 2.4×` overcount that uncorrected Hanning-then-bin-8 would produce.

Therefore:

- `s = 2.0 / ⟨w|V|²⟩_line-free` on the fit npz. Sanity: `0.3 < s < 1.5`. Forensic preview: `s ≈ 0.826`.
- **Do not also multiply by `12/29`.** Applying both (`≈ 0.342`) would shrink χ² by ~3×.
- Apply `12/29` (or `s_theory(N)` below) only if the **same npz** measures `⟨w|V|²⟩ ~ 2 / s_theory(N)` (≈4.8 for N=8). That would mean WEIGHT does not already include equivalent bandwidth.

## Theoretical covariance (unit-test algebra, not 066 weights)

Hanning of independent channels with variance σ²:

- `Var(d') = (3/8) σ²`, `Cov(d'_k, d'_{k+1}) = 0.25 σ²`, `ρ = 2/3`

Equal-weight average of N consecutive Hanning channels, `Z = S/N`:

- Native coefficients: `0.25, 0.75, 1 × (N−2), 0.75, 0.25`
- `Var(S) = (N − 0.75) σ²`
- `s_theory(N) = 3N / (8(N − 0.75))` relative to “N independent Hanning channels”
  - N=8: `12/29 ≈ 0.414`
  - N=4: `6/13 ≈ 0.462`
- Adjacent-bin correlation at N=8: `ρ_bin = 3/58 ≈ 0.052` (use this number; not 1.6%). Small enough for a **diagonal MAP**. Not small enough to ignore if NUTS error bars are the product.

## Phase 2 (after 066 MAP exists)

Banded GLS with the measured inter-bin correlation. Not in the 066 MAP critical path.

## Unit tests

1. Synthetic white noise → Hann → bin N: recover `s_theory(N)` and `ρ_bin`. This tests the formula, not 066 WEIGHT.
2. On the real 066 npz: line-free `⟨w|V|²⟩` in range, `s` applied, line-free reduced statistic ≈ 2 after `s`.
3. Forward-model test: Hann on native then bin matches a reference; Hann on the binned axis does **not** (negative test).
4. **Edge-padding gate:** a narrow line sitting in the first native channel of the unpadded window must not change the outer binned amplitude by more than the thermal noise after guards are added; unpadded Hann must fail this test.
