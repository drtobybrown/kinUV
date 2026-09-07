"""Pure-array tests for the downstream KinMS comparison."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kinuv.diagnostics.kinms_benchmark import (
    aperture_spectrum,
    cube_metrics,
    fits_sky_offsets_arcsec,
    major_axis_rotation_profile,
)

ROOT = Path(__file__).resolve().parents[1]


def test_cube_metrics_do_not_refit_flux():
    data = np.ones((3, 4, 5))
    model = 0.75 * data
    mask = np.ones_like(data, dtype=bool)
    got = cube_metrics(data, model, mask)
    assert got["n_voxel"] == data.size
    assert got["residual_rms_k"] == pytest.approx(0.25)
    assert got["normalized_rmse"] == pytest.approx(0.25)
    assert got["model_to_data_flux"] == pytest.approx(0.75)


def test_cube_metrics_respect_mask_and_shapes():
    data = np.arange(8.0).reshape(2, 2, 2)
    model = data.copy()
    mask = np.zeros_like(data, dtype=bool)
    mask[:, 0, 0] = True
    assert cube_metrics(data, model, mask)["residual_rms_k"] == 0.0
    with pytest.raises(ValueError, match="share a cube shape"):
        cube_metrics(data, model[:, :, :1], mask)
    with pytest.raises(ValueError, match="selects no finite voxels"):
        cube_metrics(data, model, np.zeros_like(mask))


def test_aperture_spectrum_uses_projected_source_mask():
    cube = np.arange(12.0).reshape(3, 2, 2)
    mask = np.zeros_like(cube, dtype=bool)
    mask[0, 0, 1] = True
    np.testing.assert_allclose(aperture_spectrum(cube, mask), cube[:, 0, 1])


def test_fits_sky_offsets_preserve_signed_east_west_handedness():
    from astropy.io import fits

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 3
    header["NAXIS2"] = 3
    header["CRPIX1"] = 2.0
    header["CRPIX2"] = 2.0
    header["CRVAL1"] = 345.0
    header["CRVAL2"] = 13.0
    header["CDELT1"] = -1.0 / 3600.0
    header["CDELT2"] = 1.0 / 3600.0
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    east, north = fits_sky_offsets_arcsec(header, (3, 3))
    assert east[1, 0] == pytest.approx(1.0, abs=1.0e-7)
    assert east[1, 2] == pytest.approx(-1.0, abs=1.0e-7)
    assert north[0, 1] == pytest.approx(-1.0, abs=1.0e-7)
    assert north[2, 1] == pytest.approx(1.0, abs=1.0e-7)


def test_rotation_profiles_share_data_defined_support():
    from astropy.io import fits

    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 31
    header["NAXIS2"] = 31
    header["CRPIX1"] = 16.0
    header["CRPIX2"] = 16.0
    header["CRVAL1"] = 345.0
    header["CRVAL2"] = 13.0
    header["CDELT1"] = -0.2 / 3600.0
    header["CDELT2"] = 0.2 / 3600.0
    header["CTYPE1"] = "RA---SIN"
    header["CTYPE2"] = "DEC--SIN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    support = np.ones((31, 31))
    moment0 = np.zeros_like(support)
    moment0[:, 8:23] = 1.0
    east, _ = fits_sky_offsets_arcsec(header, support.shape)
    moment1 = 100.0 + 20.0 * np.sign(east)
    radius, _ = major_axis_rotation_profile(
        moment0,
        moment1,
        header,
        pa_deg=90.0,
        inclination_deg=45.0,
        vsys_kms=100.0,
        n_bin=12,
        support_moment0=support,
    )
    assert radius.size > 0
    assert radius.max() > 2.0


def test_canonical_runner_defaults_to_both_targets_without_inference_calls():
    config = json.loads(
        (ROOT / "configs/benchmarks/canonical-kinms.json").read_text()
    )
    assert [item["target_id"] for item in config["targets"]] == [
        "KGAS066",
        "KGAS007",
    ]
    source = (ROOT / "scripts/run_canonical_kinms_benchmark.py").read_text()
    assert 'DEFAULT_TARGETS = ("KGAS066", "KGAS007")' in source
    for forbidden in ("kinuv.infer", "run_stage_a_map", "run_nuts", "launch_headless"):
        assert forbidden not in source
