"""Chain-draw checkpoints: scratch first, then /arc. Never ``savez`` a non-.npz path."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from kinuv.runner.canfar import fsync_path


def save_npz_atomic(path: Path, **arrays) -> Path:
    """Write ``.npz`` via a file handle so numpy does not append a second ``.npz``.

    ``np.savez('foo.npz.tmp')`` creates ``foo.npz.tmp.npz`` and leaves the
    replace source missing (the 066 chain-1 crash).
    """
    import numpy as np

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".writing")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)
    fsync_path(path)
    return path


def copy_fsync(src: Path, dest: Path) -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / (dest.name + ".copying")
    shutil.copy2(src, tmp)
    fsync_path(tmp)
    os.replace(tmp, dest)
    fsync_path(dest)
    return dest


def dual_checkpoint(scratch_dir: Path, arc_dir: Path, name: str, **arrays) -> tuple[Path, Path | None]:
    """Persist draws on node-local scratch, then copy to /arc. Arc copy is best-effort."""
    scratch = save_npz_atomic(Path(scratch_dir) / name, **arrays)
    try:
        arc = copy_fsync(scratch, Path(arc_dir) / name)
    except OSError:
        return scratch, None
    return scratch, arc


def flush_scratch_to_arc(scratch_dir: Path, arc_dir: Path) -> list[Path]:
    """Copy any ``*.npz`` from scratch onto /arc (crash / SIGTERM)."""
    scratch_dir = Path(scratch_dir)
    copied: list[Path] = []
    if not scratch_dir.is_dir():
        return copied
    for src in sorted(scratch_dir.glob("*.npz")):
        if src.name.endswith(".writing") or src.name.endswith(".copying"):
            continue
        copy_fsync(src, Path(arc_dir) / src.name)
        copied.append(src)
    return copied
