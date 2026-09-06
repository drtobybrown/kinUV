---
id: DEC-KINUV-VISLIK
status: accepted
date: 2026-09-06
owner: astra
---
# Visibility-only inference boundary

kinUV fits molecular-gas kinematics directly to complex visibilities. Its scientific likelihood is the declared visibility-domain chi-square on the selected visibility cells and covariance model.

The forward and inference dependency graph may contain angular kinematic velocity profiles, emitting-gas geometry, surface brightness, line broadening, primary-beam response, spectral response, visibility transforms, and declared statistical priors. It must not contain dark-matter halo profiles, stellar or gas mass components, gravitational-potential solvers, cosmology, physical-radius grids, or baryonic/halo decomposition.

A rotation profile returned by kinUV is a kinematic measurement. Converting angular radius to physical radius or decomposing that profile into mass components is a downstream analysis. Such an analysis consumes an immutable fit product and writes a separate provenance record; it cannot participate in likelihood evaluation, initialization, convergence decisions, or promotion of the visibility fit.

The forward model exposes a caller-supplied velocity-profile callable so future kinematic parameterizations do not require a gravitational interpretation. The Stage A arctangent and Stage B ring profiles remain supported for reproducibility.

KinMS remains an image-cube comparator. Its fitting process and outputs are downstream diagnostics and never replace or augment the kinUV likelihood.
