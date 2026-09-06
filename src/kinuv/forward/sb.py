"""SB template on the vis ImageGrid (DEC-066-SB, DEC-066-GRID, DEC-066-SHIFT)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from kinuv.decisions import requires
from kinuv.template.fftpad import default_pad_n, embed_centered
from kinuv.template.fourier_shift import fourier_shift
from kinuv.template.resample import resample_flux_conserving, sky_axes
from kinuv.template.wiener import ico_to_template
from kinuv.transforms.grid import ImageGrid

R_SCALE_066_ARCSEC = 3.0
ICO_FITS = Path(
    "/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits"
)


def image_grid_xy_arcsec(grid: ImageGrid):
    """East / North pixel centres [arcsec] in FINUFFT mode order (DEC-066-GRID)."""
    x = (np.arange(grid.nx, dtype=np.float64) - grid.nx // 2) * grid.cell_arcsec
    y = (np.arange(grid.ny, dtype=np.float64) - grid.ny // 2) * grid.cell_arcsec
    return x, y


def fits_image_east_north(data, header) -> np.ndarray:
    """CASA FITS array → ImageGrid axes: ``+x`` east, ``+y`` north.

    ``CDELT1 < 0`` means NAXIS1 increases west. Wiener BPA is east of north
    on the sky, so the flip must happen before :func:`ico_to_template`.
    """
    img = np.squeeze(np.asarray(data, dtype=np.float64))
    if img.ndim != 2:
        raise ValueError(f"Ico must be 2-D after squeeze, got {img.shape}")
    if float(header["CDELT1"]) < 0.0:
        img = np.flip(img, axis=1)
    return np.ascontiguousarray(img)


def _overlap_axes(x_in, cell_in, x_out, cell_out) -> np.ndarray:
    """Same area-overlap kernel as ``kinuv.template.resample._overlap_1d``."""
    left_in = x_in - 0.5 * cell_in
    right_in = x_in + 0.5 * cell_in
    left_out = x_out[:, None] - 0.5 * cell_out
    right_out = x_out[:, None] + 0.5 * cell_out
    return np.maximum(
        0.0,
        np.minimum(right_in[None, :], right_out) - np.maximum(left_in[None, :], left_out),
    )


@requires("DEC-066-SB", "DEC-066-GRID")
def place_template_on_grid(sb, cell_arcsec, grid: ImageGrid) -> np.ndarray:
    """Flux-conserving remap of a centred SB stamp onto ``ImageGrid``.

    Uses :func:`resample_flux_conserving` when the cell changes, then the same
    area-overlap kernel onto FINUFFT pixel centres (not Ico 0.4″ CDELT).
    Output is unit ``∫ I dΩ`` on the vis grid.
    """
    img = np.asarray(sb, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError("sb must be 2-D")
    cell_in = float(cell_arcsec)
    cell_out = float(grid.cell_arcsec)
    if not np.isclose(cell_in, cell_out, rtol=0.0, atol=1e-15):
        img = resample_flux_conserving(img, cell_in, cell_out)
        cell_in = cell_out
    ny, nx = img.shape
    x_in = sky_axes(nx, cell_in)
    y_in = sky_axes(ny, cell_in)
    x_out, y_out = image_grid_xy_arcsec(grid)
    flux = _overlap_axes(y_in, cell_in, y_out, cell_out) @ img @ _overlap_axes(
        x_in, cell_in, x_out, cell_out
    ).T
    out = flux / (cell_out * cell_out)
    tot = float(out.sum() * cell_out * cell_out)
    if abs(tot) < 1e-30:
        raise ValueError("template integral vanishes on ImageGrid")
    return out / tot


@requires("DEC-066-GRID")
def exponential_template(grid: ImageGrid, r_scale_arcsec: float = R_SCALE_066_ARCSEC):
    """Unit-integral exponential disk on ImageGrid axes. Not the 0.4″ Ico cell."""
    x, y = image_grid_xy_arcsec(grid)
    xg, yg = np.meshgrid(x, y, indexing="xy")
    sb = np.exp(-np.hypot(xg, yg) / float(r_scale_arcsec))
    d_omega = grid.cell_arcsec**2
    return sb / (float(sb.sum()) * d_omega)


def _ico_wiener_metadata(path: Path, data: np.ndarray) -> dict:
    from astropy.io import fits

    error_path = path.with_name(f"{path.stem}_err{path.suffix}")
    if not error_path.is_file():
        raise FileNotFoundError(f"Ico uncertainty map required: {error_path}")
    with fits.open(error_path) as hdul:
        error = fits_image_east_north(hdul[0].data, hdul[0].header)
    valid_error = np.isfinite(error) & (error > 0.0) & np.isfinite(data)
    if not np.any(valid_error):
        raise ValueError(f"no positive finite Ico uncertainties in {error_path}")
    sigma_ico = float(np.median(error[valid_error]))
    peak = float(np.nanmax(np.abs(data)))
    if not np.isfinite(peak) or peak <= 0.0:
        raise ValueError(f"Ico peak is not positive finite in {path}")
    return {
        "ico_path": str(path),
        "error_path": str(error_path),
        "noise_reduction": "median_positive_error_on_finite_ico_support",
        "sigma_ico": sigma_ico,
        "ico_peak": peak,
        "k_wiener_dimensionless": (sigma_ico / peak) ** 2,
        "template_units": "relative_unit_integral",
    }


def ico_template_metadata(ico_path: Path) -> dict:
    """Return the audited Wiener inputs used by :func:`load_sb_template`."""
    from astropy.io import fits

    path = Path(ico_path)
    with fits.open(path) as hdul:
        data = fits_image_east_north(hdul[0].data, hdul[0].header)
    return _ico_wiener_metadata(path, data)


def _load_sb_template(
    grid: ImageGrid,
    ico_path: Path | None = None,
    *,
    legacy_relative_noise_fraction: float | None = None,
    legacy_observed_frequency_hz: float | None = None,
) -> np.ndarray:
    """Load a normalized Ico morphology using its propagated error map.

    The companion ``*_err.fits`` supplies the Wiener noise scale. A guessed
    fraction of the image peak is not a scientifically valid substitute. The
    paired ``legacy_*`` arguments exist only to replay sealed artifacts,
    including their historical K-versus-Jy noise-unit inconsistency.
    """
    path = ICO_FITS if ico_path is None else Path(ico_path)
    if path.is_file():
        from astropy.io import fits

        with fits.open(path) as hdul:
            h = hdul[0].header
            data = fits_image_east_north(hdul[0].data, h)
        if legacy_relative_noise_fraction is None:
            if legacy_observed_frequency_hz is not None:
                raise ValueError(
                    "legacy_observed_frequency_hz requires "
                    "legacy_relative_noise_fraction"
                )
            metadata = _ico_wiener_metadata(path, data)
            sigma_ico = float(metadata["sigma_ico"])
            units = "relative"
            observed_frequency_hz = None
        else:
            if legacy_observed_frequency_hz is None:
                raise ValueError(
                    "legacy_relative_noise_fraction requires an explicit "
                    "legacy_observed_frequency_hz"
                )
            fraction = float(legacy_relative_noise_fraction)
            if not np.isfinite(fraction) or fraction <= 0.0:
                raise ValueError("legacy_relative_noise_fraction must be positive")
            sigma_ico = fraction * float(np.nanmax(np.abs(data)))
            units = "K"
            observed_frequency_hz = float(legacy_observed_frequency_hz)
        bmaj = float(h["BMAJ"]) * 3600.0
        bmin = float(h["BMIN"]) * 3600.0
        bpa = float(h["BPA"])
        cell = abs(float(h["CDELT2"])) * 3600.0
        tmpl = ico_to_template(
            data,
            cell,
            observed_frequency_hz,
            bmaj,
            bmin,
            bpa,
            units=units,
            sigma_empty=sigma_ico,
        )
        return place_template_on_grid(tmpl.sb, tmpl.cell_arcsec, grid)
    return exponential_template(grid)


@requires("DEC-066-SB", "DEC-066-GRID")
def load_sb_template(grid: ImageGrid, ico_path: Path | None = None) -> np.ndarray:
    """Load normalized Ico morphology with its propagated uncertainty map."""
    return _load_sb_template(grid, ico_path)


@requires("DEC-066-SB", "DEC-066-GRID")
def load_legacy_sb_template(
    grid: ImageGrid,
    ico_path: Path,
    *,
    relative_noise_fraction: float,
    observed_frequency_hz: float,
) -> np.ndarray:
    """Replay a sealed template with its explicitly supplied legacy choices."""
    return _load_sb_template(
        grid,
        ico_path,
        legacy_relative_noise_fraction=relative_noise_fraction,
        legacy_observed_frequency_hz=observed_frequency_hz,
    )


M2_R2_ARCSEC = 2.5
M2_SIGMA_ARCSEC = 1.5
M2_MIN_POSITIVE = 8
CENTROID_TOL_ARCSEC = 0.01


def galaxy_r_phi(grid: ImageGrid, pa_rad, i_rad):
    """Galaxy-plane R, phi with phi = atan2(yg, xg). Not sky atan2."""
    from kinuv.geometry import sky_to_galaxy

    x, y = image_grid_xy_arcsec(grid)
    xe, yn = np.meshgrid(x, y, indexing="xy")
    xg, yg = sky_to_galaxy(xe, yn, pa_rad, i_rad)
    radius = np.hypot(xg, yg)
    phi = np.arctan2(yg, xg)
    return radius, phi


def _template_centroid(image, grid: ImageGrid):
    x, y = image_grid_xy_arcsec(grid)
    xe, yn = np.meshgrid(x, y, indexing="xy")
    w = np.asarray(image, dtype=np.float64)
    tot = float(np.sum(w))
    if not np.isfinite(tot) or abs(tot) < 1e-30:
        return float("nan"), float("nan")
    return float(np.sum(xe * w) / tot), float(np.sum(yn * w) / tot)


@requires("DEC-066-SB", "DEC-066-GRID", "DEC-066-PA", "DEC-066-INC")
def axisymmetrise_template(
    sb,
    grid: ImageGrid,
    pa_rad,
    i_rad,
    *,
    min_positive: int = M2_MIN_POSITIVE,
    dr_arcsec=None,
):
    """Azimuthal mean I0(R) on deprojected elliptical annuli.

    Membership is ``(I > 0) & isfinite``. Do not ``nan_to_num`` first.
    Inner dropped bins share one nuclear aperture mean; last valid mean
    is held outward. ``dr`` defaults to one vis cell.
    """
    img = np.asarray(sb, dtype=np.float64)
    if img.shape != (grid.ny, grid.nx):
        raise ValueError(f"template {img.shape} != grid {(grid.ny, grid.nx)}")
    radius, _ = galaxy_r_phi(grid, pa_rad, i_rad)
    pos = (img > 0.0) & np.isfinite(img)
    dr = float(grid.cell_arcsec) if dr_arcsec is None else float(dr_arcsec)
    if dr <= 0.0:
        raise ValueError("dr_arcsec must be positive")
    r_max = float(np.max(radius))
    n_bin = max(int(np.ceil(r_max / dr)), 1)
    edges = np.arange(n_bin + 1, dtype=np.float64) * dr
    means = np.full(n_bin, np.nan)
    counts = np.zeros(n_bin, dtype=int)
    for i in range(n_bin):
        sel = pos & (radius >= edges[i]) & (radius < edges[i + 1])
        counts[i] = int(np.sum(sel))
        if counts[i] >= int(min_positive):
            means[i] = float(np.mean(img[sel]))
    valid = np.where(np.isfinite(means))[0]
    if valid.size == 0:
        raise ValueError("no annulus has enough positive pixels")
    first, last = int(valid[0]), int(valid[-1])
    if first > 0:
        nuclear = pos & (radius < edges[first])
        if int(np.sum(nuclear)) >= 1:
            nuc = float(np.mean(img[nuclear]))
        else:
            nuc = float(means[first])
        means[:first] = nuc
    if last < n_bin - 1:
        means[last + 1 :] = means[last]
    centres = 0.5 * (edges[:-1] + edges[1:])
    i0 = np.interp(radius, centres, means, left=means[0], right=means[-1])
    i0 = np.where(np.isfinite(img), i0, 0.0)
    tot = float(np.sum(i0) * grid.cell_arcsec**2)
    if abs(tot) < 1e-30:
        raise ValueError("axisymmetrised integral vanishes")
    i0 = i0 / tot
    return i0, {
        "dr_arcsec": dr,
        "r_centre_arcsec": centres,
        "I0_R": means,
        "n_positive": counts,
        "first_valid_bin": first,
        "last_valid_bin": last,
    }


@requires("DEC-066-SB", "DEC-066-GRID", "DEC-066-PA", "DEC-066-INC")
def apply_m2(
    i0,
    grid: ImageGrid,
    pa_rad,
    i_rad,
    amplitude,
    phi2,
    *,
    r2_arcsec: float = M2_R2_ARCSEC,
    sigma_arcsec: float = M2_SIGMA_ARCSEC,
):
    """I = I0 [1 + a2(R) cos 2(phi-phi2)], then unit integral."""
    base = np.asarray(i0, dtype=np.float64)
    if base.shape != (grid.ny, grid.nx):
        raise ValueError(f"I0 {base.shape} != grid {(grid.ny, grid.nx)}")
    amp = float(amplitude)
    if amp < 0.0 or amp >= 1.0:
        raise ValueError(f"A must be in [0, 1), got {amp}")
    phi2 = float(phi2) % np.pi
    radius, phi = galaxy_r_phi(grid, pa_rad, i_rad)
    a2 = amp * np.exp(-0.5 * ((radius - float(r2_arcsec)) / float(sigma_arcsec)) ** 2)
    raw = base * (1.0 + a2 * np.cos(2.0 * (phi - phi2)))
    d_omega = grid.cell_arcsec**2
    integ0 = float(np.sum(base) * d_omega)
    monopole = float(np.sum(base * a2 * np.cos(2.0 * (phi - phi2))) * d_omega)
    monopole_frac = monopole / integ0 if abs(integ0) > 1e-30 else float("nan")
    if np.any(raw < 0.0):
        raw = np.maximum(raw, 0.0)
    tot = float(np.sum(raw) * d_omega)
    if abs(tot) < 1e-30:
        raise ValueError("m=2 integral vanishes")
    out = raw / tot
    cx0, cy0 = _template_centroid(base, grid)
    cx, cy = _template_centroid(out, grid)
    shift = float(np.hypot(cx - cx0, cy - cy0))
    if shift >= CENTROID_TOL_ARCSEC:
        raise ValueError(f"m=2 centroid shift {shift:.4f}\" >= 0.01\"")
    imag_sum = float(np.sum(np.imag(np.asarray(out, dtype=np.complex128))))
    return out, {
        "A": amp,
        "phi2": phi2,
        "monopole_frac": monopole_frac,
        "centroid_shift_arcsec": shift,
        "im_sum": imag_sum,
    }


@requires("DEC-066-SHIFT")
def fourier_shift_padded(image, dx_arcsec, dy_arcsec, cell_arcsec, pad_n=None):
    """Fourier shift on the Wiener pad; **no crop**. For the SHIFT broadening bound."""
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    pad = int(pad_n) if pad_n is not None else default_pad_n(max(ny, nx))
    padded, _, _ = embed_centered(img, pad, pad)
    # Reuse the production interpolator on the padded canvas (crop is identity).
    return fourier_shift(padded, dx_arcsec, dy_arcsec, cell_arcsec, pad_n=pad)


@requires("DEC-066-SHIFT")
def exponential_r_scale(
    image,
    cell_arcsec,
    x0_arcsec: float = 0.0,
    y0_arcsec: float = 0.0,
    r_min_arcsec: float = 1.5,
    r_max_arcsec: float = 8.0,
) -> float:
    """Azimuthally averaged exponential scale length about ``(x0, y0)`` [arcsec]."""
    img = np.asarray(image, dtype=np.float64)
    ny, nx = img.shape
    x = (np.arange(nx, dtype=np.float64) - nx // 2) * float(cell_arcsec) - float(
        x0_arcsec
    )
    y = (np.arange(ny, dtype=np.float64) - ny // 2) * float(cell_arcsec) - float(
        y0_arcsec
    )
    X, Y = np.meshgrid(x, y, indexing="xy")
    r = np.hypot(X, Y)
    sel = (img > 0.0) & np.isfinite(img) & (r >= r_min_arcsec) & (r <= r_max_arcsec)
    if int(sel.sum()) < 8:
        raise ValueError("too few pixels to fit r_scale")
    design = np.vstack([np.ones(int(sel.sum())), -r[sel]]).T
    coeff, *_ = np.linalg.lstsq(design, np.log(img[sel]), rcond=None)
    slope = float(coeff[1])
    if slope <= 0.0:
        raise ValueError(f"non-positive inverse scale {slope}")
    return 1.0 / slope
