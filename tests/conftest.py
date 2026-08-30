"""Shared fixtures. Float64 is mandatory.

G1: set CPU / x64 / compile cache before any test imports jax.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp/kinuv-xdg")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", "/tmp/kinuv-jax-cache")

import numpy as np
import pytest


@pytest.fixture(scope="session")
def uv_sampling():
    """Compact-ALMA uv sampling, KILOGAS-like (15–414 m → ~305 kλ at 230 GHz)."""
    rng = np.random.default_rng(20260817)
    n = 400
    b = 10 ** rng.uniform(np.log10(15.0), np.log10(414.0), n)
    th = rng.uniform(0, 2 * np.pi, n)
    return b * np.cos(th), b * np.sin(th)


@pytest.fixture(scope="session")
def freqs():
    """Nine channels around CO(2-1) at z ~ 0.03."""
    return 224.3e9 + np.arange(9) * 7.8e6
