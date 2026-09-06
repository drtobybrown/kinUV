# Canonical kinUV–KinMS benchmark

`scripts/run_canonical_kinms_benchmark.py` runs the downstream image-domain comparison for KGAS066 and KGAS007 by default. It consumes official KILOGAS 10 km/s cubes, promoted kinUV model cubes, and independently generated KinMS cubes. It never imports a kinUV optimizer or sampler.

Target paths, comparison geometry, KinMS initialization, and KinMS bounds live in `configs/benchmarks/canonical-kinms.json`. KinMS remains an external optional dependency. Point `KINUV_KINMS_PYTHON` at an environment containing KinMS, NumPy, SciPy, and Astropy:

```bash
KINUV_KINMS_PYTHON=/path/to/kinms-env/bin/python \
  python scripts/run_canonical_kinms_benchmark.py
```

The runner reuses the retained KGAS066 KinMS fit. If KGAS007 has no cached comparator, it invokes `external/_kinms_best_worker.py` using the versioned fit configuration. Use `--target KGAS066` or `--target KGAS007` for a focused diagnostic, and `--no-fit` to require cached KinMS products.

Each accepted target bundle under `results/production/<target>/<run_id>/benchmark/` contains:

- `benchmark.json`: source paths and matched-voxel residual metrics;
- `kinms_model_k.fits`: KinMS cube converted to kelvin on the official grid;
- `moments.npz`: moments 0, 1, and 2 for data, kinUV, KinMS, and residuals;
- `profiles.npz`: aperture spectra, major-axis PVDs, and image-derived rotation profiles;
- comparison figures for moments, spectra, channel maps, major-axis PVDs, and rotation profiles;
- the KinMS fit configuration, result, and bounded worker log when a fit is generated.

These metrics assess image-domain residual behavior. The promoted scientific score remains visibility chi-square. The benchmark does not refit a multiplicative amplitude during scoring, preventing a diagnostic from hiding a flux mismatch.
