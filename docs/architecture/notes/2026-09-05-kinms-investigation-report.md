# KinMS geometry investigation (2026-09-05)

Inspected local KinMSpy: `external_fitters/venv/lib/python3.12/site-packages/kinms/KinMS.py`.

## What KinMS actually does

1. **`inClouds` are face-on disk coordinates**, not sky offsets. `r_flat = sqrt(x^2+y^2)` sets `V_c`. Line-of-sight velocity is
   `v_los = -V_c * cos(θ) * sin(i)` with `θ = atan2(y, x)` (`KinMS.py` ~466–470). Disk `+x` is the approaching side.
2. KinMS then **projects** those clouds: `inclination_projection` compresses `y` by `cos(i)`, then `position_angle_rotation` with angle `90° − PA`. Docstring: PA = 0 puts the **redshifted** side on **+y** (north).
3. Native cube shape is **`(nx, ny, nv)`** (`histogramdd` / `bincount` bins `(x_size, y_size, v_size)`). FITS `getdata` is **`(nv, ny, nx)`**.
4. **`vSys` is a FITS-header keyword only.** The array is always centred on channel `nv/2`. Spectral shifts must use **`vOffset`**.

## Why our earlier models had a 90° kinematic swap

Two independent bugs, both in our wrapper (not a KinMS likelihood limitation):

| Bug | Effect |
|---|---|
| `np.transpose(cube, (2, 0, 1))` on KinMS `(nx, ny, nv)` | Writes `(nv, nx, ny)` into a `(nv, ny, nx)` FITS header → **RA/Dec swap** → major PV flat, minor PV holds the gradient |
| Sky-sampled M0 positions passed as `inClouds` after `sky_to_galaxy` | KinMS applied inc/PA **again** → double projection, `cos²(i)` slit, ~5″ vs 7.9″ M0 |

kinUV `sky_to_galaxy` (PA=0 → north on +x) also does not match KinMS (PA=0 → receding on +y). Inverting KinMS with the kinUV rotation is not a drop-in.

## Fix

- Production path is **`sbProf` / `sbRad`**: I(R) from elliptical annuli on CLEAN M0; KinMS samples and projects internally.
- Cube conversion is **`transpose (2, 1, 0)`**.
- Systemic velocity uses **`vOffset = vsys − v_cube[nv/2]`**.

Pre-flight `external/test_kinms_geometry.py` (PA=199.7°, i=43.9°, V0=250 km/s): major PV half-amplitude ≈ 160 km/s (expected 173), minor max deviation ≈ 23 km/s, model M0 90% radius 7.5″ vs data 7.85″.

## Limitations (not bugs)

- Axisymmetric `sbProf` cannot represent non-axisymmetric CO (arms, leftover SB). That is the intended image-plane comparator, not a kinUV likelihood.
- Cube χ² with a free flux scale is not the visibility χ². `quote_inner_slope: false` on real data.
- Morphological PA of a b/a ≈ 0.72 disk is weakly constrained; kinematic major/minor PV is the geometry gate.
