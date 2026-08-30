# kinUV

Visibility-plane kinematic fitter for KILOGAS. First target: **KGAS066**.

**Start here (human):** [`docs/methodology.md`](docs/methodology.md). Agents: [`AGENTS.md`](AGENTS.md) → [`field-guide/index.md`](field-guide/index.md) → [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md) → [`docs/reviews/BOARD.md`](docs/reviews/BOARD.md). Physics: [`docs/decisions/`](docs/decisions/). Image-plane check: [`docs/diagnostics/stage-b-vs-imaging.md`](docs/diagnostics/stage-b-vs-imaging.md). Figure style: [`docs/diagnostics/plotting.md`](docs/diagnostics/plotting.md). kinUV vs uvkin: [`docs/diagnostics/repos.md`](docs/diagnostics/repos.md). Changelog: [`CHANGELOG.md`](CHANGELOG.md).

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
