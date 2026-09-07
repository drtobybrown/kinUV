"""Pure-array S4 comparison metrics for common restored-cube products."""

from __future__ import annotations

import numpy as np

from kinuv.constants import C_LIGHT_KM_S


def topo_radio_to_lsrk_radio(velocity_kms, frequency_correction_kms):
    """Apply a frequency-equivalent frame correction to radio velocity."""

    velocity = np.asarray(velocity_kms, dtype=np.float64)
    correction = float(frequency_correction_kms)
    return velocity - correction * (1.0 - velocity / C_LIGHT_KM_S)


def channel_noise_from_integrated_error(error_moment0, mask, dv_kms):
    """Recover a common channel RMS from an integrated-intensity error map."""

    error = np.asarray(error_moment0, dtype=np.float64)
    support = np.asarray(mask, dtype=bool)
    if support.ndim != 3 or error.shape != support.shape[1:]:
        raise ValueError("moment-0 error and cube mask spatial shapes must match")
    count = np.sum(support, axis=0)
    valid = np.isfinite(error) & (error > 0.0) & (count > 0)
    if not np.any(valid):
        raise ValueError("no valid integrated-error pixels overlap the cube mask")
    sigma = error[valid] / (abs(float(dv_kms)) * np.sqrt(count[valid]))
    return float(np.median(sigma)), int(sigma.size)


def common_reduced_chi2(data, model, mask, sigma_k, n_parameter):
    """Empirical reduced chi-square on one common mask and scalar channel RMS."""

    observed = np.asarray(data, dtype=np.float64)
    predicted = np.asarray(model, dtype=np.float64)
    use = np.asarray(mask, dtype=bool) & np.isfinite(observed) & np.isfinite(predicted)
    if observed.shape != predicted.shape or use.shape != observed.shape:
        raise ValueError("data, model, and mask must have identical shapes")
    n = int(np.sum(use))
    dof = n - int(n_parameter)
    if dof <= 0 or sigma_k <= 0.0:
        raise ValueError("positive degrees of freedom and noise are required")
    chi2 = float(np.sum(((observed[use] - predicted[use]) / float(sigma_k)) ** 2))
    return {"chi2": chi2, "dof": dof, "reduced_chi2": chi2 / dof, "n_voxel": n}


def projected_velocity_rmse(profiles, inclination_deg):
    """Compare model moment-1 curves with the common data-curve proxy."""

    radius_data = np.asarray(profiles["rotation_radius_data"], dtype=np.float64)
    speed_data = np.asarray(profiles["rotation_speed_data"], dtype=np.float64)
    if radius_data.size < 2:
        raise ValueError("at least two common data-profile points are required")
    factor = float(np.sin(np.radians(float(inclination_deg))))
    output = {}
    for name in ("kinuv", "kinms"):
        radius = np.asarray(profiles[f"rotation_radius_{name}"], dtype=np.float64)
        speed = np.asarray(profiles[f"rotation_speed_{name}"], dtype=np.float64)
        lo = max(float(radius_data.min()), float(radius.min()))
        hi = min(float(radius_data.max()), float(radius.max()))
        use = (radius_data >= lo) & (radius_data <= hi) & np.isfinite(speed_data)
        if np.sum(use) < 2:
            raise ValueError(f"{name} has fewer than two overlapping profile points")
        estimate = np.interp(radius_data[use], radius, speed)
        residual = factor * (estimate - speed_data[use])
        output[name] = {
            "rmse_kms": float(np.sqrt(np.mean(residual**2))),
            "n_radius": int(np.sum(use)),
            "radius_min_arcsec": float(radius_data[use].min()),
            "radius_max_arcsec": float(radius_data[use].max()),
        }
    output["ratio_kinuv_over_kinms"] = (
        output["kinuv"]["rmse_kms"] / output["kinms"]["rmse_kms"]
    )
    output["gate_eligible"] = min(
        output["kinuv"]["n_radius"], output["kinms"]["n_radius"]
    ) >= 3
    output["eligibility_rule"] = "at least three common moment-1 radial bins"
    return output


def projected_arctan_speed(parameters, radius_arcsec):
    """Projected arctan speed ``u(r)=V0 sin(i) 2/pi atan(r/Rturn)``."""

    radius = np.asarray(radius_arcsec, dtype=np.float64)
    inclination = float(
        parameters.get("inclination_deg", parameters.get("i_deg"))
    )
    v0 = float(parameters.get("v0_kms", parameters.get("v0_kms_diagnostic")))
    turnover = float(parameters["r_t_arcsec"])
    if turnover <= 0.0:
        raise ValueError("r_t_arcsec must be positive")
    return (
        v0
        * np.sin(np.radians(inclination))
        * (2.0 / np.pi)
        * np.arctan(radius / turnover)
    )


def subbeam_turnover_recovery(
    truth,
    fitted,
    radius_arcsec,
    radial_weight,
    bmaj_arcsec,
):
    """Supporting turnover and inner-rise errors for a sub-beam truth."""

    radius = np.asarray(radius_arcsec, dtype=np.float64)
    weight = np.asarray(radial_weight, dtype=np.float64)
    if radius.shape != weight.shape or radius.ndim != 1:
        raise ValueError("radius_arcsec and radial_weight must be matching 1-D arrays")
    if np.any(~np.isfinite(radius)) or np.any(~np.isfinite(weight)):
        raise ValueError("radius and weight must be finite")
    beam = float(bmaj_arcsec)
    if beam <= 0.0:
        raise ValueError("bmaj_arcsec must be positive")
    truth_turnover = float(truth["r_t_arcsec"])
    eligible = truth_turnover < beam
    inner = (radius <= beam) & (radius >= 0.0) & (weight > 0.0)
    if not eligible:
        return {
            "eligible": False,
            "eligibility_rule": "truth r_t_arcsec < 1.0 * BMAJ",
            "truth_turnover_over_bmaj": truth_turnover / beam,
        }
    if not np.any(inner):
        raise ValueError("the profile grid has no positive-weight radius <= BMAJ")
    residual = projected_arctan_speed(fitted, radius[inner]) - projected_arctan_speed(
        truth, radius[inner]
    )
    inner_rmse = float(
        np.sqrt(np.sum(weight[inner] * residual**2) / np.sum(weight[inner]))
    )
    turnover_error = abs(float(fitted["r_t_arcsec"]) - truth_turnover)
    return {
        "eligible": True,
        "eligibility_rule": "truth r_t_arcsec < 1.0 * BMAJ",
        "bmaj_arcsec": beam,
        "truth_turnover_arcsec": truth_turnover,
        "truth_turnover_over_bmaj": truth_turnover / beam,
        "fitted_turnover_arcsec": float(fitted["r_t_arcsec"]),
        "absolute_turnover_error_arcsec": turnover_error,
        "absolute_turnover_error_over_bmaj": turnover_error / beam,
        "inner_projected_velocity_rmse_kms": inner_rmse,
        "inner_radius_max_arcsec": beam,
        "inner_radius_count": int(np.sum(inner)),
        "gate_role": "supporting publication diagnostic; does not gate S4",
    }


def _component_summary(values):
    z = np.asarray(values, dtype=np.complex128).ravel()
    use = np.isfinite(z.real) & np.isfinite(z.imag)
    z = z[use]
    if z.size < 2:
        return None
    real = z.real
    imag = z.imag
    real_sd = float(np.std(real, ddof=1))
    imag_sd = float(np.std(imag, ddof=1))
    return {
        "n_complex": int(z.size),
        "real_mean": float(np.mean(real)),
        "imag_mean": float(np.mean(imag)),
        "real_variance": float(np.var(real, ddof=1)),
        "imag_variance": float(np.var(imag, ddof=1)),
        "real_mean_z": float(np.mean(real) / (real_sd / np.sqrt(real.size)))
        if real_sd > 0.0
        else 0.0,
        "imag_mean_z": float(np.mean(imag) / (imag_sd / np.sqrt(imag.size)))
        if imag_sd > 0.0
        else 0.0,
        "real_imag_correlation": float(np.corrcoef(real, imag)[0, 1])
        if real_sd > 0.0 and imag_sd > 0.0
        else 0.0,
    }


def structured_line_free_diagnostics(table, line_free_mask, scale, rho, row_fold_id):
    """Whiten native line-free cells and summarize MS grouping residuals."""

    line_free = np.asarray(line_free_mask, dtype=bool)
    if line_free.shape != (table.vis.shape[1],):
        raise ValueError("line_free_mask must match native visibility channels")
    channel_index = np.flatnonzero(line_free)
    contiguous = np.zeros(channel_index.size, dtype=bool)
    contiguous[1:] = np.diff(channel_index) == 1
    row_count = np.zeros(table.vis.shape[0], dtype=np.int64)
    row_mean = np.zeros(table.vis.shape[0], dtype=np.complex128)
    moments = np.zeros(6, dtype=np.float64)  # n, sum_r, sum_i, r2, i2, ri
    block_rows = 256
    for start in range(0, table.vis.shape[0], block_rows):
        stop = min(start + block_rows, table.vis.shape[0])
        vis = table.vis[start:stop, channel_index]
        weight = table.weights[start:stop, channel_index]
        good = weight > 0.0
        residual = np.sqrt(np.where(good, weight / float(scale), 0.0)) * vis
        whitened = np.zeros_like(residual)
        starts = good.copy()
        starts[:, 1:] &= ~(good[:, :-1] & contiguous[None, 1:])
        whitened[starts] = residual[starts]
        adjacent = good[:, 1:] & good[:, :-1] & contiguous[None, 1:]
        innovations = (residual[:, 1:] - float(rho) * residual[:, :-1]) / np.sqrt(
            1.0 - float(rho) ** 2
        )
        target = whitened[:, 1:]
        target[adjacent] = innovations[adjacent]
        count = np.sum(good, axis=1)
        row_count[start:stop] = count
        row_mean[start:stop] = np.divide(
            np.sum(whitened, axis=1),
            count,
            out=np.zeros(stop - start, dtype=np.complex128),
            where=count > 0,
        )
        values = whitened[good]
        real = values.real
        imag = values.imag
        moments += np.array(
            [
                values.size,
                np.sum(real),
                np.sum(imag),
                np.sum(real * real),
                np.sum(imag * imag),
                np.sum(real * imag),
            ],
            dtype=np.float64,
        )

    def grouped(labels):
        output = []
        for label in np.unique(labels):
            use = (labels == label) & (row_count > 0)
            summary = _component_summary(row_mean[use])
            if summary is not None:
                summary["group_id"] = int(label)
                output.append(summary)
        return output

    baseline = grouped(table.baseline)
    fold = grouped(np.asarray(row_fold_id, dtype=np.int64))
    antenna = []
    for antenna_id in np.unique(np.concatenate([table.antenna1, table.antenna2])):
        use = (
            ((table.antenna1 == antenna_id) | (table.antenna2 == antenna_id))
            & (row_count > 0)
        )
        summary = _component_summary(row_mean[use])
        if summary is not None:
            summary["group_id"] = int(antenna_id)
            antenna.append(summary)
    n, sum_real, sum_imag, sum_real2, sum_imag2, sum_cross = moments
    mean_real = sum_real / n
    mean_imag = sum_imag / n
    var_real = (sum_real2 - n * mean_real**2) / (n - 1.0)
    var_imag = (sum_imag2 - n * mean_imag**2) / (n - 1.0)
    covariance = (sum_cross - n * mean_real * mean_imag) / (n - 1.0)
    global_summary = {
        "n_complex": int(n),
        "real_mean": float(mean_real),
        "imag_mean": float(mean_imag),
        "real_variance": float(var_real),
        "imag_variance": float(var_imag),
        "real_mean_z": float(mean_real / np.sqrt(var_real / n)),
        "imag_mean_z": float(mean_imag / np.sqrt(var_imag / n)),
        "real_imag_correlation": float(covariance / np.sqrt(var_real * var_imag)),
    }

    def extrema(groups):
        return {
            "group_count": len(groups),
            "max_absolute_component_mean_z": max(
                (
                    max(abs(row["real_mean_z"]), abs(row["imag_mean_z"]))
                    for row in groups
                ),
                default=0.0,
            ),
            "max_absolute_real_imag_correlation": max(
                (abs(row["real_imag_correlation"]) for row in groups), default=0.0
            ),
        }

    return {
        "global": global_summary,
        "folds": {"summary": extrema(fold), "groups": fold},
        "baselines": {"summary": extrema(baseline), "groups": baseline},
        "antennas": {"summary": extrema(antenna), "groups": antenna},
        "whitening": "native C1 innovations on contiguous positive-weight line-free cells",
    }


__all__ = [
    "topo_radio_to_lsrk_radio",
    "channel_noise_from_integrated_error",
    "common_reduced_chi2",
    "projected_velocity_rmse",
    "projected_arctan_speed",
    "subbeam_turnover_recovery",
    "structured_line_free_diagnostics",
]
