---
role: literature
field: astrophysics
date: 2026-09-03
canon_generation: 4
ids:
  - DEC-066-PA
  - DEC-066-INFER
  - DEC-066-VC
  - DEC-066-SB
  - DEC-066-INC
  - DEC-066-ZEROMODEL
---

# Rank

Not an ADR. If this file disagrees with a `DEC-*`, the DEC wins. Rank below `docs/architecture/STATUS.md` and the board log. INDEX is unchanged. Do not paste this essay into the Field Guide. Official MAP `kinuv-KGAS066-uvsign-map` is read-only. Do not start G4. Do not call Laplace-MH "NUTS". Harmonic terms `s1`/`c3` are not in `DEC-066-VC`; leftover structure is not a detection of them.

# Question

For a rotating disk fit to interferometric CO visibilities with a frozen (non-axisymmetric) SB template, is PA and PA+180 a true degeneracy? What does the literature say about approaching vs receding kinematic PA, vsys–PA coupling, and when 180° flips fail?

# Answer

No. With `V_c > 0` the two angles are **named sides of one dipole**, not interchangeable MAP modes. Radio/ALMA convention fixes PA on the **receding** major axis (east of north). PA+180 puts the receding model on the approaching sky side. That is a discrete two-start, scored by `Delta_chi2` vs `V=0`. It becomes a true invariance only if one also flips the sign of `V_c` (usually forbidden) or drops the sign of the velocity field (photometric ellipse; unsigned `|M1-vsys|`). A frozen spiral `I_CO` breaks even the weak version: brightness is not 180°-symmetric, so the wrong-side start cannot hide in the visibilities.

# Convention: receding major-axis PA

Schoenmakers, Franx & de Zeeuw (1997, MNRAS 292, 349) define `Gamma` as the anticlockwise angle from north to the major axis of the **receding** half. 3DBarolo uses the same sentence (Di Teodoro & Fraternali 2015, MNRAS 451, 3021, Fig. 2): PA is the receding-side major axis, anticlockwise from north. The code can fit `SIDE = A | R | B` because approaching and receding **curves** can differ (warp, lopsidedness), not because the angles are equivalent. KinMS (Davis et al. 2013, MNRAS 429, 534) sets `posAng = 0` when the **redshifted** half lies along +y; `vPosAng` is the usual astronomical kinematic PA when morphological and kinematic axes split. ALMA disk tools follow the redshifted axis: `eddy` measures polar angle east of north relative to the **redshifted** major axis (Teague 2019, JOSS 4, 1220); Wölfer et al. (2023, A&A 670, A154) state PA as north to the redshifted semi-major axis, easterly.

`DEC-066-PA` already chose this convention: fit receding-side PA (E of N); seed 205.2° from the CO YAML, not optical 108.9°. Photometric PA is 180°-ambiguous and is not receding-side.

# When PA+180 is, and is not, a degeneracy

Projected circular rotation is

`V_los = vsys + V_c(R) * sin(i) * cos(theta)`,

with `theta` measured from the receding major axis (van der Kruit & Allen 1978; Begeman 1989, A&A 223, 47). Replacing PA by PA+180 sends `cos(theta) -> -cos(theta)`. The cube (and therefore the visibilities) match the original only if `V_c` changes sign. Stage A keeps `V_0 > 0` (arctan; `DEC-066-VC`), so the two starts are different models.

If the SB is axisymmetric, PA and PA+180 with `V_c > 0` are related by a velocity inversion about `vsys`. The data pick one side of the Doppler dipole. Two-start L-BFGS is the ordinary way to name that side; it is a **discrete search**, not a continuous degeneracy and not a license to stack draws.

Photometry is the actual 180° invariance: an ellipse has no approaching/receding label. That is why `DEC-066-PA` refuses the optical PA as a kinematic seed.

# vsys–PA coupling, solid body, and when the flip fails

Schoenmakers et al. (1997, App. A2–A3): a PA error `delta Gamma` mixes into `s1` and, through the tilted-ring fit, into `c1`, `c3`, `s3`. A centre error mixes into `c0` (the local `vsys`), `s2`, and `c2`. An `m=2` potential or spiral couples to the fitted `Gamma` and `vsys`; their NGC 2403 rings show `Gamma` and `vsys` wiggling where the HI has spiral structure. That is **coupling**, not PA = PA+180.

Solid-body rotation (`alpha = 1`) is the documented place where uniqueness fails (Schoenmakers App. A.3.3; Krajnović et al. 2006, MNRAS 366, 787, §4.3). Iso-velocity contours are parallel; the opening angle is 180°. PA of the gradient can still be measured; flattening `q`, centre, and `vsys` cannot. Kinemetry therefore treats solid body as a special case, not as a free 180° flip of a flat-curve disk. Slow rotation (no odd parity) loses PA as well.

Begeman (1989) is the `i`–`V_c` caution (`i < 40°` ill-defined on moment-1 rings). 066 freezes `i` at 43.9° (`DEC-066-INC`), so that continuous degeneracy is not the 066 PA problem. Józsa et al. (2007, A&A 468, 731, TiRiFiC) note a different PA zero (90° from GALMOD/ROTCUR); that is a software offset, not a physical flip.

**When the 180° start fails as a peer basin:**

1. **Frozen non-axisymmetric SB.** A spiral or lopsided `I_CO` is not invariant under 180° sky rotation. Putting the receding model on the approaching side misaligns bright arms with the wrong channels. On visibilities the Fourier phase of that template does not commute with a velocity-axis flip. 3D cube fitters that *renormalize* SB per ring (3DBarolo `AZIMUTH` / KinMS free `sbProf`) can hide some of this; a frozen Wiener template cannot.
2. **Signed `V_c`.** With `V_0 > 0` the approaching start is a different Doppler dipole. `Delta_chi2` vs `V=0` is the score (`DEC-066-ZEROMODEL`).
3. **Init that is not an approaching MAP.** Overwriting only PA on a receding MAP leaves the chain in the receding geometry with the dipole reversed. The sampler then walks a ridge: `vsys` absorbs the sign error, `r_t` collapses toward the solid-body limit where Schoenmakers say `vsys` and centre are formally free, and PA does not lock.
4. **Structured leftover.** Franx, van Gorkom & de Zeeuw (1994, ApJ 436, 642) and Schoenmakers et al. (1997) name harmonic residuals after a circular fit. Those terms are **not** in `DEC-066-VC`. Leftover `chi2` that is more structured vs velocity than vs `uv` is an SB-dominated circular-model failure, not evidence that approaching is a second kinematic mode.

# Map onto 066 (numbers as given; do not invent others)

Official two-start L-BFGS already scored the discrete pair: receding start `Delta_chi2 = 35553`, approaching start `Delta_chi2 = 4260`. That is an ~8× gap on the `V=0` gate, not a coin flip. Frozen Wiener `I_CO` has spiral structure; leftover `chi2` is more structured vs velocity than vs `uv`. Inclination is frozen; Stage A is arctan `V_c`. Approaching NUTS started from the receding MAP with only `PA = 25.2` overwritten: chains sat near 15° and 64°, `r_t` collapsed, and they did not mix. That is the solid-body / wrong-basin failure mode above, not a second official product.

`DEC-066-INFER`: MAP first; do not sample a likelihood that cannot beat the noise pedestal; do not switch tool because the loser start failed.

# What 066 should do next

- MAP-first on any approaching claim: L-BFGS from PA=25.2 (and optional finite chain medians) to a **new tree**. Do not overwrite `kinuv-KGAS066-uvsign-map`. Report `chi2`, `Delta_chi2` vs `V=0`, and `Delta_chi2` vs the official receding MAP.
- Do not relaunch NUTS from receding MAP + PA-only overwrite. That experiment is the unmixed 15°/64° / collapsed-`r_t` run.
- Do not stack receding and approaching draws, and do not average the 15° pair. Occupancy of unmixed chains is not posterior mass.
- Do not quote approaching as a peer 066 product unless its MAP `Delta_chi2` vs `V=0` competes with 35553 (same order, not 4260). If it stays ~4e3, write the failure and stop.
- Leave `i` frozen and Stage A arctan. Do not add Franx/Schoenmakers harmonics without a user `DEC-*` stub. Do not start G4.

# References

- Begeman, K. 1989, A&A, 223, 47
- Davis, T. A., et al. 2013, MNRAS, 429, 534 (KinMS)
- Di Teodoro, E. M., & Fraternali, F. 2015, MNRAS, 451, 3021 (3DBarolo)
- Franx, M., van Gorkom, J. H., & de Zeeuw, P. T. 1994, ApJ, 436, 642
- Józsa, G. I. G., et al. 2007, A&A, 468, 731 (TiRiFiC)
- Krajnović, D., Cappellari, M., de Zeeuw, P. T., & Copin, Y. 2006, MNRAS, 366, 787
- Schoenmakers, R. H. M., Franx, M., & de Zeeuw, P. T. 1997, MNRAS, 292, 349
- Teague, R. 2019, JOSS, 4, 1220 (`eddy`)
- van der Kruit, P. C., & Allen, R. J. 1978, ARA&A, 16, 103
- Wölfer, L., et al. 2023, A&A, 670, A154
