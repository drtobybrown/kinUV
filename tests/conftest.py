"""Shared fixtures. Float64 is mandatory.

Set CPU / x64 / scratch cache before any test imports jax.
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

_USER = os.environ.get("USER") or os.environ.get("LOGNAME") or "kinuv"
_SESS = os.environ.get("SKAHA_SESSION_ID") or os.environ.get("HOSTNAME") or "local"
_ROOT = None
for _base in ("/scratch", "/tmp"):
    _cand = Path(_base) / f"kinuv-{_USER}" / str(_SESS)
    try:
        _cand.mkdir(parents=True, exist_ok=True)
        _probe = _cand / ".w"
        _probe.write_text("ok")
        _probe.unlink()
        _ROOT = _cand
        break
    except OSError:
        continue
if _ROOT is None:
    _ROOT = Path("/tmp") / f"kinuv-{_USER}" / str(_SESS)
    _ROOT.mkdir(parents=True, exist_ok=True)

_tmp = _ROOT / "tmp"
_cache = _ROOT / "jax-cache"
_xdg = _ROOT / "xdg"
_tmp.mkdir(exist_ok=True)
_cache.mkdir(exist_ok=True)
_xdg.mkdir(exist_ok=True)
os.environ.setdefault("TMPDIR", str(_tmp))
os.environ.setdefault("TEMP", str(_tmp))
os.environ.setdefault("TMP", str(_tmp))
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(_cache))
os.environ.setdefault("XDG_CACHE_HOME", str(_xdg))

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
