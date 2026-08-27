# kinUV

Visibility-plane kinematic fitter for KILOGAS. First target: **KGAS066**.

Start here: [`AGENTS.md`](AGENTS.md) → [`field-guide/index.md`](field-guide/index.md) → [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Physics lives in [`docs/decisions/`](docs/decisions/). Image-plane check of the Stage B MAP: [`docs/diagnostics/stage-b-vs-imaging.md`](docs/diagnostics/stage-b-vs-imaging.md).

## Status (066)

Stage A (arctan) and Stage B (N=7 rings, λ=0) MAP are done on the aggregated visibilities. Geometry (flux, PA, vsys, σ, dx, dy) is frozen at Stage A for Stage B. The official arctan product remains Stage A; AIC prefers Stage B on vis χ². NUTS has not been run.

Imaging comparison uses the **10 km/s** v1.3 cube, not 30 km/s (see the diagnostic note). Ico / vis-trim stay on 30 km/s.

## Install / tests

```bash
pip install -e ".[io,test]"
export PYTHONPATH=src
pytest
```

NUFFT extras: `pip install -e ".[nufft]"`. Plotting the Stage B vs imaging figures needs matplotlib (`pip install matplotlib`) and the CANFAR `/arc` paths in the diagnostic note.
