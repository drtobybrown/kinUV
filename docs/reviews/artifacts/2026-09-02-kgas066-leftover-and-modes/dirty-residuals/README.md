Not KinMS. Not uvkin. S1 was restoring-beam M1/M2; 3DBarolo was not on PATH.

The sibling `moments.png` / `spectra.png` / `pv_*.png` panels are CLEAN-matched Data | Model | Residual of the same sky (10 km/s cube, restoring beam). They are **not** a type-1 adjoint of residual visibilities. There is no \(F^{-1}\{\Delta V\}\) on the 881×95 irregular \((u,v)\) array in this repo (type-2 degrid only). Do not call those PNGs a KinMS posterior.

S1 vis vs CLEAN-beam (`docs/diagnostics/s1-mock.md`):

- vis recovered \(r_t=0.254''\) vs truth \(0.25''\); \(\Delta V_0=+0.19\) km/s; \(\Delta\sigma=-0.11\) km/s
- CLEAN M1 inner slope 94.7 vs truth 236.7 km/s/arcsec
- CLEAN M2 56.1 vs injected \(\sigma=8\) km/s

That is the vis vs image-plane claim. This folder restates it; it does not install KinMS.
