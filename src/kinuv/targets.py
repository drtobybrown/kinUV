"""kinUV-owned target metadata used by production entry points."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TargetConfig:
    galaxy: str
    archive_id: str
    ra_deg: float
    dec_deg: float
    vsys_optical_kms: float
    pa_deg: float
    i_deg: float

    @property
    def source(self) -> str:
        return f"kinuv.targets:{self.galaxy}"

    def inference_overrides(self) -> dict[str, float]:
        values = asdict(self)
        return {
            key: float(values[key])
            for key in (
                "ra_deg",
                "dec_deg",
                "vsys_optical_kms",
                "pa_deg",
                "i_deg",
            )
        }


TARGETS = {
    "KGAS007": TargetConfig(
        galaxy="KGAS007",
        archive_id="KILOGAS007",
        ra_deg=146.576065,
        dec_deg=2.88434,
        vsys_optical_kms=14229.0,
        pa_deg=212.6,
        i_deg=28.9,
    ),
}


def get_target(name: str) -> TargetConfig:
    """Return a configured target or fail rather than borrowing another galaxy."""
    key = str(name).upper()
    try:
        return TARGETS[key]
    except KeyError as exc:
        raise KeyError(f"unknown kinUV target {name!r}; configured={sorted(TARGETS)}") from exc
