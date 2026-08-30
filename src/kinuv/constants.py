"""Physical constants and the single unit-conversion boundary.

Angular-to-physical conversion lives **here and nowhere else**.

The predecessor code mixed arcsec radii with a mass model that is defined in
kpc, which makes ``r_scale`` and ``vmax`` mean different things in different
call sites.  In kinUV the rule is:

* Everything on the **sky** is in **arcsec** (and radians only inside a
  transform kernel).
* Everything in the **mass model** is in **kpc** and **km/s**.
* The conversion happens exactly once, at :class:`Distance`, which is
  constructed from a redshift or an explicit angular diameter distance and
  passed explicitly to any code that needs to cross the boundary.

Never write ``r_kpc = r_arcsec * something`` anywhere else in the package.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Speed of light, exact SI definition [m/s].
C_LIGHT_M_S = 299_792_458.0

#: Speed of light [km/s].
C_LIGHT_KM_S = C_LIGHT_M_S / 1.0e3

#: Radians per arcsecond.
ARCSEC_TO_RAD = np.pi / (180.0 * 3600.0)

#: CO(2-1) rest frequency [Hz].
F_REST_CO21_HZ = 230.538e9

#: Conversion from a Gaussian FWHM to its standard deviation.
FWHM_TO_SIGMA = 1.0 / np.sqrt(8.0 * np.log(2.0))

#: Boltzmann constant, SI exact [J/K].
K_BOLTZMANN_J_K = 1.380649e-23

#: Jansky in SI [W m⁻² Hz⁻¹].
JY_W_M2_HZ = 1.0e-26


@dataclass(frozen=True)
class Distance:
    """The one place where angular and physical scales meet.

    Parameters
    ----------
    angular_diameter_mpc
        Angular diameter distance :math:`D_A` in Mpc.  One arcsecond subtends
        :math:`D_A \\times 1''` in proper length.

    Notes
    -----
    Construct with :meth:`from_redshift` for the standard flat-LCDM case, or
    directly when you already have :math:`D_A` from another cosmology.

    At z = 0.03 in flat LCDM (H0 = 70, Om = 0.3) one arcsecond corresponds to
    roughly 606 pc, which is the scale KILOGAS operates at.
    """

    angular_diameter_mpc: float

    @classmethod
    def from_redshift(
        cls,
        z: float,
        *,
        H0: float = 70.0,
        Om0: float = 0.3,
    ) -> "Distance":
        """Flat-LCDM angular diameter distance.

        Uses astropy when available so the cosmology is the standard one, and
        falls back to a direct numerical integration of the comoving distance
        otherwise (kept so the package has no hard astropy dependency in its
        numerical core).
        """
        try:
            from astropy.cosmology import FlatLambdaCDM

            cosmo = FlatLambdaCDM(H0=H0, Om0=Om0)
            return cls(float(cosmo.angular_diameter_distance(z).to_value("Mpc")))
        except ImportError:
            # comoving distance D_C = (c/H0) * int_0^z dz'/E(z')
            zz = np.linspace(0.0, z, 4096)
            ez = np.sqrt(Om0 * (1.0 + zz) ** 3 + (1.0 - Om0))
            d_c = (C_LIGHT_KM_S / H0) * np.trapezoid(1.0 / ez, zz)
            return cls(float(d_c / (1.0 + z)))

    @property
    def kpc_per_arcsec(self) -> float:
        """Proper kpc subtended by one arcsecond at this distance."""
        return self.angular_diameter_mpc * 1.0e3 * ARCSEC_TO_RAD

    def arcsec_to_kpc(self, arcsec):
        """Convert an angular radius [arcsec] to a physical radius [kpc]."""
        return arcsec * self.kpc_per_arcsec

    def kpc_to_arcsec(self, kpc):
        """Convert a physical radius [kpc] to an angular radius [arcsec]."""
        return kpc / self.kpc_per_arcsec


def freq_to_velocity_kms(freq_hz, f_rest_hz: float = F_REST_CO21_HZ):
    """Radio-convention velocity [km/s] for observed frequencies [Hz].

    .. math:: v = c\\,(\\nu_{\\rm rest} - \\nu) / \\nu_{\\rm rest}
    """
    from kinuv.xp import numpy_or_jax

    xp = numpy_or_jax(freq_hz)
    freq_hz = xp.asarray(freq_hz)
    return C_LIGHT_KM_S * (f_rest_hz - freq_hz) / f_rest_hz


def velocity_to_freq_hz(v_kms, f_rest_hz: float = F_REST_CO21_HZ):
    """Inverse of :func:`freq_to_velocity_kms`."""
    v_kms = np.asarray(v_kms, dtype=np.float64)
    return f_rest_hz * (1.0 - v_kms / C_LIGHT_KM_S)
