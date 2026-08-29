"""Frozen i and PA seed (DEC-066-INC, DEC-066-PA). No moment-1 loader."""

from pathlib import Path

import numpy as np
import pytest

from kinuv.geometry import (
    catalogue_ba,
    galaxy_to_sky,
    incline,
    inclination_deg,
    inclination_prior_half_width_deg,
    inclination_rad,
    pa_seed_deg,
    pa_seed_rad,
    rotate_by_pa,
    sky_to_galaxy,
)

YAML_INC_INIT_DISCARDED = 51.690
OPTICAL_PA_NOT_SEED = 108.9
GEOM_SRC = Path(__file__).resolve().parents[1] / "src" / "kinuv" / "geometry.py"


def test_inclination_from_catalogue_ba_not_hardcoded_439():
    assert catalogue_ba() == pytest.approx(0.721)
    i_deg = inclination_deg()
    i_rad = inclination_rad()
    assert i_rad == pytest.approx(np.arccos(0.721))
    assert i_deg == pytest.approx(np.degrees(np.arccos(0.721)))
    assert abs(i_deg - 43.9) < 0.05
    src = GEOM_SRC.read_text(encoding="utf-8")
    assert "arccos" in src
    assert "0.721" in src


def test_inclination_is_not_yaml_51690():
    i = inclination_deg()
    assert i != pytest.approx(YAML_INC_INIT_DISCARDED)
    assert abs(i - YAML_INC_INIT_DISCARDED) > 1.0
    assert inclination_rad() != pytest.approx(np.radians(YAML_INC_INIT_DISCARDED))


def test_inclination_prior_half_width():
    assert inclination_prior_half_width_deg() == pytest.approx(5.0)


def test_pa_seed_is_co_yaml_receding_not_optical():
    pa = pa_seed_deg()
    assert pa == pytest.approx(205.2)
    assert pa_seed_rad() == pytest.approx(np.radians(205.2))
    assert abs(pa - OPTICAL_PA_NOT_SEED) > 1.0
    assert pa != pytest.approx(OPTICAL_PA_NOT_SEED)


def test_geometry_has_no_moment1_loader():
    src = GEOM_SRC.read_text(encoding="utf-8").lower()
    for token in ("mom1", "moment1", "moment_1", "fits.open", "astropy.io"):
        assert token not in src
    import kinuv.geometry as geom

    assert not hasattr(geom, "load")
    assert not hasattr(geom, "from_moment")


def test_rotate_by_pa_matches_standard_disk_frame():
    """x' = E sin PA + N cos PA; y' = E cos PA − N sin PA (receding +x)."""
    rng = np.random.default_rng(66)
    e = rng.normal(size=16)
    n = rng.normal(size=16)
    pa = np.radians(201.9)
    s, c = np.sin(pa), np.cos(pa)
    x_maj, y_min = rotate_by_pa(e, n, pa)
    assert x_maj == pytest.approx(e * s + n * c)
    assert y_min == pytest.approx(e * c - n * s)
    x_maj, y_min = rotate_by_pa(0.0, 1.0, pa_rad=0.0)
    assert x_maj == pytest.approx(1.0)
    assert y_min == pytest.approx(0.0)
    x_maj, y_min = rotate_by_pa(1.0, 0.0, pa_rad=np.pi / 2.0)
    assert x_maj == pytest.approx(1.0)
    assert y_min == pytest.approx(0.0)


def test_incline_face_on_identity_and_stretch():
    x, y = incline(2.0, 3.0, i_rad=0.0)
    assert x == pytest.approx(2.0)
    assert y == pytest.approx(3.0)
    x, y = incline(1.0, 1.0, i_rad=np.pi / 3.0)
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(2.0)


def test_sky_galaxy_roundtrip_at_frozen_geometry():
    rng = np.random.default_rng(66)
    x_e = rng.normal(size=32)
    y_n = rng.normal(size=32)
    pa = pa_seed_rad()
    i = inclination_rad()
    xg, yg = sky_to_galaxy(x_e, y_n, pa, i)
    xs, ys = galaxy_to_sky(xg, yg, pa, i)
    assert xs == pytest.approx(x_e)
    assert ys == pytest.approx(y_n)


def test_requires_bindings():
    assert inclination_deg._kinuv_requires == ("DEC-066-INC",)
    assert pa_seed_deg._kinuv_requires == ("DEC-066-PA",)
    assert sky_to_galaxy._kinuv_requires == ("DEC-066-INC", "DEC-066-PA")
