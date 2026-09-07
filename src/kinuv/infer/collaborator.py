"""Conditioned bounded coordinates for the collaborator MAP/NUTS campaign."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinuv.infer.s3 import S3_PARAMETER_NAMES


@dataclass(frozen=True)
class CampaignTransform:
    full_bounds: tuple[tuple[float, float], ...]
    active: tuple[int, ...]
    log_kinematic_indices: tuple[int, ...]

    @property
    def names(self):
        return tuple(S3_PARAMETER_NAMES[index] for index in self.active)

    @property
    def physical_log_indices(self):
        indices = {0, 3, *self.log_kinematic_indices}
        if 15 in self.active:
            indices.add(15)
        return tuple(index for index in self.active if index in indices)

    def full_to_bounded(self, full):
        q = np.asarray(full, dtype=np.float64).copy()
        for index in self.log_kinematic_indices:
            if q[index] <= 0.0:
                raise ValueError(f"{S3_PARAMETER_NAMES[index]} must be positive")
            q[index] = np.log(q[index])
        return q

    def bounded_to_full_numpy(self, bounded):
        z = np.asarray(bounded, dtype=np.float64).copy()
        for index in self.log_kinematic_indices:
            z[index] = np.exp(z[index])
        return z

    def solver_bounds(self):
        bounds = list(self.full_bounds)
        for index in self.log_kinematic_indices:
            lo, hi = bounds[index]
            bounds[index] = (np.log(max(float(lo), 1.0e-6)), np.log(float(hi)))
        return tuple(bounds)

    def active_initial_unconstrained(self, full):
        bounded = self.full_to_bounded(full)
        bounds = self.solver_bounds()
        values = []
        for index in self.active:
            lo, hi = bounds[index]
            fraction = (bounded[index] - lo) / (hi - lo)
            fraction = np.clip(fraction, 1.0e-9, 1.0 - 1.0e-9)
            values.append(np.log(fraction) - np.log1p(-fraction))
        return np.asarray(values, dtype=np.float64)

    def unconstrained_to_full_jax(self, unconstrained, fixed_full):
        import jax.nn as jnn
        import jax.numpy as jnp

        y = jnp.asarray(unconstrained)
        z = jnp.asarray(fixed_full)
        bounds = self.solver_bounds()
        log_jacobian = jnp.asarray(0.0)
        physical_measure = jnp.asarray(0.0)
        for position, index in enumerate(self.active):
            lo, hi = bounds[index]
            fraction = jnn.sigmoid(y[position])
            q = float(lo) + float(hi - lo) * fraction
            log_jacobian = (
                log_jacobian
                + np.log(float(hi - lo))
                + jnn.log_sigmoid(y[position])
                + jnn.log_sigmoid(-y[position])
            )
            if index in self.physical_log_indices:
                physical_measure = physical_measure + q
            value = jnp.exp(q) if index in self.log_kinematic_indices else q
            z = z.at[index].set(value)
        return z, log_jacobian + physical_measure


def campaign_transform(full_bounds, active, *, uses_rings):
    active = tuple(index for index in active if index not in (13, 14))
    logs = tuple(range(9, 13)) if uses_rings else (7, 8)
    logs = tuple(index for index in logs if index in active)
    return CampaignTransform(tuple(full_bounds), active, logs)


__all__ = ["CampaignTransform", "campaign_transform"]
