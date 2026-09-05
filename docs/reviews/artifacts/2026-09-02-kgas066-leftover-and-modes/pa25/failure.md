# Approaching PA 25.2 failure (066)

Official MAP `kinuv-KGAS066-uvsign-map` was not written. Receding G3 remains
the only `sampler: nuts` product. Do not start G4. Do not quote inner dV/dr
or S2 16/50/84.

## Four-chain merge already on disk (`pa25/`)

Written 2026-09-03T13:10Z as `COMPLETED_UNMIXED`. **Do not overwrite.**

| Lie on disk | Fact |
|---|---|
| `sampler: laplace_mh` | Autodiff NUTS merge that failed mix. Label leak in `product_record`. |
| `leftover_chi2_structured: false` | Leftover was not measured. Official leftover-vs-velocity is True. |

c1/c3 PA ~15°, c2 ~64°, c4 exploded (flux ~1e262). R_hat(PA) ~22. Not a mode.

## Official two-start (already discarded approaching)

`stage_a_map.json` message: PA=205.2 Δχ²=35552.7, PA=25.2 Δχ²=4260.2.
Approaching start χ² ≈ 199968 (gap vs official MAP **+31292.5**).
DEC-066-PA is receding-side PA, seed 205.2°.

## c1–c3 diagnostic

See `c1c3-diagnostic/`. Expect `sampler: nuts_unmixed`, leftover key omitted,
`mixing_pass: false`. Three chains cannot mint `sampler: nuts`.

Landed: sampler=nuts_unmixed, n_kept=1,
mixing_pass=False.

## Approaching L-BFGS (new tree only)

See `approaching-map/`. Winner χ²=168675.5875362135,
Δχ² vs V=0=35552.66026627409,
Δ vs official MAP=-0.008015875908313319,
PA=199.72810599233483.
Leftover bit from arrays: True.

Quoted NUTS-median χ² uses `r_t=0.5` arcsec only (`chain_medians_rt_clamped`).
Raw `r_t~6e-4` medians are under `approaching-map/raw/` and are unphysical.

## Gate

No approaching NUTS this card (dual accept major). If a later MAP ever
competes (Δ vs official MAP ≥ −200 and Δχ² vs V=0 ≥ 35000), new propose.
Do not stack modes. `KGAS066-latest` stays receding.
