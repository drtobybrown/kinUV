# kinUV

High-throughput visibility-plane molecular-gas kinematic fitter. Current canonical targets: **KGAS066 and KGAS007**.

**Start here (human):** [`../results/MANIFEST.md`](../results/MANIFEST.md) identifies accepted products and archived runs. Scientific interpretation and closed experiments: [`docs/PRODUCTION_RECORD.md`](docs/PRODUCTION_RECORD.md) and [`docs/methodology.md`](docs/methodology.md). Agents: [`AGENTS.md`](AGENTS.md) → [`field-guide/index.md`](field-guide/index.md) → [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md) → [`docs/reviews/BOARD.md`](docs/reviews/BOARD.md). Physics: [`docs/decisions/`](docs/decisions/). Data extraction boundary: [`docs/diagnostics/data-preparation.md`](docs/diagnostics/data-preparation.md). Image-plane check: [`docs/diagnostics/stage-b-vs-imaging.md`](docs/diagnostics/stage-b-vs-imaging.md). Figure style: [`docs/diagnostics/plotting.md`](docs/diagnostics/plotting.md).

kinUV has no CASA or legacy-package dependency. Calibrated Measurement Sets are exported by the separately installed [`ms2kinuv`](../ms2kinuv/) companion; kinUV ingests its versioned NPZ tables.

The production objective is visibility-domain `chi2`. Kinematic velocity profiles are fitted directly in angular coordinates; cosmology and any baryonic/halo mass interpretation are downstream postprocessing and are absent from the forward-model, likelihood, and sampler dependency graph.

## Status (066)

Stage A (arctan) and Stage B (N=7 rings, lambda=0) MAP are done on the aggregated visibilities. Geometry is frozen at Stage A for Stage B. The official arctan product remains Stage A; AIC prefers Stage B on vis chi2. Laplace-MH is not NUTS; S2 SBC failed 68/95 coverage.

Imaging comparison uses the **10 km/s** v1.3 cube, not 30 km/s (see the diagnostic note). Ico / vis-trim stay on 30 km/s.

## Install / tests

```bash
pip install -e ".[io,test]"
export PYTHONPATH=src
pytest
```

NUFFT extras: `pip install -e ".[nufft]"`. Diagnostic figures need matplotlib (`pip install matplotlib`) and the CANFAR `/arc` paths in the diagnostic note. Standard leftover `chi2` (+ optional moments): `python scripts/plot_fit_diagnostics.py`. Do not use `native_diagonal`; the operator is `kinuv.response.spectral.hann_then_bin`.

The canonical downstream KinMS comparison selects KGAS066 and KGAS007 by default and never launches kinUV inference:

```bash
KINUV_KINMS_PYTHON=/path/to/kinms-env/bin/python \
  python scripts/run_canonical_kinms_benchmark.py
```

See [`docs/diagnostics/kinms-benchmark.md`](docs/diagnostics/kinms-benchmark.md).
