# S1 mock: vis Stage A vs CLEAN-beam moments (KGAS066)

This is the inject-and-recovery test licensed by the 2026-08-29 reviewer ACK ([`docs/reviews/2026-08-29-review-methodology.md`](../reviews/2026-08-29-review-methodology.md)). Visibility χ² is the fit. The cube numbers are a beam-smearing comparator, not a second likelihood. No NUTS. No Stage B. No new DEC.

**Runners:** [`scripts/run_s1_mock.py`](../../scripts/run_s1_mock.py), [`scripts/plot_leftover_chi2.py`](../../scripts/plot_leftover_chi2.py).

**Artifacts:** [`docs/reviews/artifacts/2026-08-29-s1-mock/`](../reviews/artifacts/2026-08-29-s1-mock/).

## Operator (mod 2)

`assert` `kinuv.response.spectral.hann_then_bin` **before** generating visibilities. S1 uses `load_kgas066` + `predict_binned` on the production fit array (881×95, \(N=4\), \(s=0.514\)). Gate 2 (`tests/test_mock_recovery.py`) no longer allows a diagonal-native fallback.

## Inject (Stage A only, mod 3)

Real 066 \((u,v,\nu)\). Thin disk, \(i\) frozen at catalogue \(43.86°\) ([DEC-066-INC](../decisions/DEC-066-INC.md)). No \(h_z\). Truth:

| | value |
|---|---|
| \(r_t\) | \(0.25″\) (\(\ll\) Ico BMAJ \(1.30″\)) |
| \(V_0\) | 250 km/s |
| \(\sigma\) | 8 km/s |
| PA | \(199.73°\) |
| flux | 70 Jy |
| noise | complex Gaussian with XX empirical \(s\) and data weights |

S1 recovery uses a **script-local** \(r_t\) box \((0.05″, 15″)\). Production [`RT_BOUNDS_ARCSEC = (0.5, 15)`](../../src/kinuv/infer/seeds.py) is unchanged.

Pre-registered vis window: \(|\Delta V_0|<10\) km/s, \(|\Delta\sigma|<2\) km/s. Inner slope is \(\mathrm{d}V_c/\mathrm{d}r\) of the arctan at \(r=0.25\,\mathrm{BMAJ}=0.325″\).

## Vis recovery vs CLEAN-beam cube

| | truth | vis Stage A | cube (restoring beam) |
|---|---|---|---|
| \(V_0\) (km/s) | 250 | 250.19 | — |
| \(r_t\) (arcsec) | 0.25 | 0.254 | — |
| \(\sigma\) (km/s) | 8.0 | 7.89 | M2 56.1 |
| inner \(\mathrm{d}V_c/\mathrm{d}r\) (km/s / ″) | 236.7 | 237.8 | M1 94.7 |
| PA (deg) | 199.73 | 199.75 | — |

Vis: \(\Delta V_0=+0.19\) km/s, \(\Delta\sigma=-0.11\) km/s — **inside the window**. Cube inner slope is \(\sim 2.5\times\) too shallow; apparent M2 is \(\sim 7\times\) the injected \(\sigma\). 3DBarolo was **not on PATH**; the cube estimator is `sky_cube` → restoring beam of the 10 km/s cube → major-axis M1/M2.

This is the UV science claim the propose asked for: visibilities recover sub-beam \(\mathrm{d}V/\mathrm{d}r\) and \(\sigma\) where the CLEANed cube does not.

## Degeneracy (mod 1)

Official S1 MAP: `i_held_fixed: true`, `h_z_in_model: false`. Laplace covariances from 5×5 \(\chi^2\) slices around the MAP ([`s1_chi2_slices.png`](../reviews/artifacts/2026-08-29-s1-mock/s1_chi2_slices.png)):

- PA–\(\sigma\) correlation \(0.014\) (not a PA–smear trade).
- \(\sigma\)–\(i\) (i unfrozen **scan only**) correlation \(-0.17\).
- PA–\(r_t\) correlation \(0.067\).

Sub-beam recovery is not buying inner slope by sliding inclination.

## XX-only \(s\) (mod 4)

`pol: "XX"`, `s=0.5136` from line-free fit channels of the real npz. Likelihood is \(\chi^2 = s \sum w |d-m|^2\). Stokes \(I\) / XX+YY is **not** assumed. No NUTS this wave ([DEC-066-INFER](../decisions/DEC-066-INFER.md) still waits on this mock, which now recovers).

## Leftover \(\chi^2\) of the real MAP

Official Stage A (`kinuv-KGAS066-uvsign-map`): leftover sum \(168675.6\), matching `stage_a_map.json`. Mean \(\chi^2\) per row \(\approx 191\), per channel \(\approx 1776\).

- **vs baseline:** binned mean is nearly flat from \(\sim 50\)–\(350\) m (slightly higher on the shortest spacings). Not a classic missing-flux bowl at \(u=v=0\).
- **vs velocity:** leftover is structured across the line (\(\sim 7900\)–\(8300\) km/s), not white. That is **SB misspecification** (frozen Wiener Ico cannot represent spirals), not CLEAN covariance.

Figures: [`leftover_chi2.png`](../reviews/artifacts/2026-08-29-s1-mock/leftover_chi2.png).

## Caveats

- Cube M1 slope used 12 major-axis inner pixels; the qualitative bias (shallow + high M2) is robust, the exact 94.7 number is not a 3DBarolo posterior.
- Injected noise is diagonal XX Gaussian, not a CASA simulator.
- Production Stage A on *real* 066 still sits on the \(0.5″\) \(r_t\) floor — S1 shows the *engine* can recover \(0.25″\) when that is the truth; the real galaxy may be consistent with the floor or with Ico-scale SB.
- NUTS / S2 coverage is still not shown.
