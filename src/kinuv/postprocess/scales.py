"""Angular-to-physical scale conversion for downstream analysis only.

This module is intentionally outside ``kinuv.forward``, ``kinuv.likelihood``,
and ``kinuv.infer``. A visibility fit does not require a cosmology or a
physical-radius grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinuv.constants import ARCSEC_TO_RAD, C_LIGHT_KM_S


@dataclass(frozen=True)
class Distance:
    """Angular-diameter distance and explicit angular/physical conversions."""

    angular_diameter_mpc: float

    @classmethod
    def from_redshift(
        cls,
        z: float,
        *,
        H0: float = 70.0,
        Om0: float = 0.3,
    ) -> "Distance":
        """Return a flat-LCDM angular-diameter distance.

        Astropy is used when installed. The NumPy integration fallback keeps
        this optional postprocessing helper usable with the core dependency
        set.
        """
        try:
            from astropy.cosmology import FlatLambdaCDM

            cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
            return cls(float(cosmo.angular_diameter_distance(z).to_value("Mpc")))
        except ImportError:
            zz = np.linspace(0.0, z, 4096)
            ez = np.sqrt(Om0 * (1.0 + zz) ** 3 + (1.0 - Om0))
            d_c = (C_LIGHT_KM_S / H0) * np.trapz(1.0 / ez, zz)
            return cls(float(d_c / (1.0 + z)))

    @property
    def kpc_per_arcsec(self) -> float:
        return self.angular_diameter_mpc * 1.0e3 * ARCSEC_TO_RAD

    def arcsec_to_kpc(self, arcsec):
        return np.asarray(arcsec) * self.kpc_per_arcsec

    def kpc_to_arcsec(self, kpc):
        return np.asarray(kpc) / self.kpc_per_arcsec
