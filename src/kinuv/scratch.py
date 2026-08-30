"""Node-local scratch for JIT, TMP, and small checkpoints. Not vis cubes."""

from __future__ import annotations

import os
from pathlib import Path


def kinuv_scratch_root() -> Path:
    """``/scratch/kinuv-$USER/$session`` if writable, else ``/tmp/...``.

    Never ``/arc``. JAX compile cache lives here and is not synced to NFS.
    """
    user = os.environ.get("USER") or os.environ.get("LOGNAME") or "kinuv"
    session = (
        os.environ.get("SKAHA_SESSION_ID")
        or os.environ.get("HOSTNAME")
        or "local"
    )
    for base in ("/scratch", "/tmp"):
        root = Path(base) / f"kinuv-{user}" / str(session)
        try:
            root.mkdir(parents=True, exist_ok=True)
            try:
                os.chmod(root, 0o700)
            except OSError:
                pass
            probe = root / ".w"
            probe.write_text("ok")
            probe.unlink()
            return root
        except OSError:
            continue
    return Path("/tmp") / f"kinuv-{user}" / str(session)


def apply_scratch_env() -> Path:
    """Set TMPDIR and JAX cache under the scratch root. Call before jax import."""
    root = kinuv_scratch_root()
    tmp = root / "tmp"
    cache = root / "jax-cache"
    xdg = root / "xdg"
    tmp.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    xdg.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TMPDIR", str(tmp))
    os.environ.setdefault("TEMP", str(tmp))
    os.environ.setdefault("TMP", str(tmp))
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(cache))
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg))
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    return root
