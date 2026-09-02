# Restoring-beam / CLEAN-matched residuals (not a cube fitter)

S1 already showed vis Stage A recovers inject \(r_t=0.25''\) where the CLEANed cube does not (M1 slope 94.7 vs truth 236.7 km/s/arcsec; M2 56 vs 8). 3DBarolo was not on PATH. That table: [`docs/diagnostics/s1-mock.md`](../../../../diagnostics/s1-mock.md).

The three-way moments/spectra in the parent folder are CLEAN-matched cubes of vis models. They are not inverse Fourier transforms of residual visibilities (type-1 NUFFT is not implemented). Captions: restoring-beam / dirty residual, not a second likelihood.
