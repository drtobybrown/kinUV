# Survey-readiness (checklist only)

Not a DEC. Not a 400-galaxy runner. Sequence for the **066 kernel** (flags, then CPU JAX, then NUTS, then SBC): [`gold-standard-roadmap.md`](gold-standard-roadmap.md). CANFAR I/O: [`scratch.md`](scratch.md). [`DEC-066-TARGET`](../decisions/DEC-066-TARGET.md) still locks **code** to KGAS066 until MAP beats \(V=0\) (passed) **and** injected vsys/PA/flux recover on 066 uv (Hann+bin mock; see methodology). [`DEC-HIER-SELFUNC`](../decisions/DEC-HIER-SELFUNC.md) stays Phase 5. G0 flags on one Stage A JSON are not a subset dispatcher.

Recommended user stubs (you add them): TARGET subset; optional `h_z`; unfreeze `i` with an optical prior; warp/strip/KDC classes. Do not treat Stage B rings as a warp.

KILOGAS DR1 is ~452 galaxies (283 MaNGA + 169 SAMI, \(z=0.016\)–\(0.060\)), compact ALMA CO(2–1). A “decent subset” without a new TARGET stub: regular disks that look like 066 (Ico+npz present, \(i\) not face-on, no strong bar). Bars, stripping tails, and mosaics wait on an S4 user DEC.

## Must stay in any future ingest

- CASA vis Fourier sign: `NPZ_UV_SIGN = -1` (textbook \(-2\pi i\) with \((u,v)\to(-u,-v)\)).
- Ico `CDELT1<0`: `fits_image_east_north` before Wiener.
- Cube-window trim (not YAML `obs_freq_range`); empirical \(s=2/\langle w|V|^2\rangle\) on line-free fit channels; Hann on **native** then bin.
- Two-start receding PA (seed and seed−180°). Do not use absolute OSCMETRIC Ω as a vis-fit veto until a user DEC replaces it.
- Primary beam in the image plane after shift; no vis phase ramp after PB.

## Hardcoded 066 that would break a subset

- Paths: laptop `/Users/thbrown/kilogas/...` and CANFAR `KGAS66/30kms/` / `KILOGAS066.npz`.
- Geometry: `ba = 0.721`, PA seed 205.2°.
- Spectral window from the 066 Ico cube VOPT span.
- Frozen circular \(i\); single isotropic \(\sigma\); Wiener Ico required (exponential fallback is a science change).
- XX-only npz (√2 still on the table).

## Costing sketch (Systems Y3)

At ~0.5 s/eval CPU: Stage A MAP (~10³ evals) is minutes per galaxy → 400 MAP ~ hours–a day on a small CPU pool. NUTS at \(10^5\) evals is ~12–17 h **per galaxy** on CPU → not a survey product until eval/s rises. GPU only after a 066 CPU NUTS smoke (`sampler: nuts`, `R_hat` / `ESS`). G1 is CPU JAX `predict_binned`, not a Skaha GPU image.

Do not write a multi-galaxy runner until a user TARGET stub exists.
