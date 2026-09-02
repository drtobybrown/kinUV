"""Headless run kind: receding NUTS vs approaching PA 25.2 vs GPU smoke.

Approaching and GPU smoke must not steal KGAS066-latest or write the G3 receding
artifact dir. Do not import canfar here (canfar re-exports these helpers).
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OFFICIAL_PA = 199.72980072503037
APPROACH_PA = 25.2
KIND_PA25 = "nuts-pa25"
KIND_GPU = "nuts-gpu"
ARTIFACT_G3_REL = "docs/reviews/artifacts/2026-08-30-g3-nuts"
ARTIFACT_LEFTOVER_REL = "docs/reviews/artifacts/2026-09-02-kgas066-leftover-and-modes"
ARTIFACT_PA25_REL = ARTIFACT_LEFTOVER_REL + "/pa25"
ARTIFACT_GPU_REL = "docs/reviews/artifacts/2026-09-02-gpu-nuts-smoke"
ARTIFACT_G3 = REPO / ARTIFACT_G3_REL
ARTIFACT_LEFTOVER = REPO / ARTIFACT_LEFTOVER_REL
ARTIFACT_PA25 = REPO / ARTIFACT_PA25_REL
ARTIFACT_GPU = REPO / ARTIFACT_GPU_REL
CUDA_VENV = "/arc/projects/KILOGAS/analysis/toby_sandbox/venvs/kinuv-cuda"
RECOVERY_VENV = "/arc/home/thbrown/kinuv-venv-recovery"
DEFAULT_GPU_IMAGE = "skaha/astroml-cuda:latest"
FALLBACK_GPU_IMAGE = "skaha/astroflow-cuda:latest"
SERIAL_CPU_WALL_S = 17440.032
TENX_WALL_S = SERIAL_CPU_WALL_S / 10.0


def is_approaching_kind(kind: str) -> bool:
    return "pa25" in str(kind).lower()


def is_gpu_kind(kind: str) -> bool:
    return "gpu" in str(kind).lower()


def pa_init_deg(kind: str, override: float | None = None) -> float:
    if override is not None:
        return float(override)
    return APPROACH_PA if is_approaching_kind(kind) else OFFICIAL_PA


def steal_latest(kind: str) -> bool:
    """False for approaching and GPU smoke: do not retarget KGAS066-latest."""
    k = str(kind).lower()
    return not (is_approaching_kind(k) or is_gpu_kind(k))


def artifact_dir_for_kind(kind: str, repo: Path | None = None) -> Path:
    root = Path(repo) if repo is not None else REPO
    if is_approaching_kind(kind):
        return root / ARTIFACT_PA25_REL
    if is_gpu_kind(kind):
        return root / ARTIFACT_GPU_REL
    return root / ARTIFACT_G3_REL


def corner_title(pa_init: float) -> str:
    return (
        f"066 NUTS PA {float(pa_init):.2f}; 6 sampled; not calibrated; "
        "not S2 Laplace; do not quote inner dV/dr"
    )
