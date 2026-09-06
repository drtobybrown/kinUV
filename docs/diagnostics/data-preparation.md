# Data preparation boundary

kinUV is the standalone modeling and inference repository. It reads NPZ visibility tables and has no CASA, casatasks, pyuvdata, casacore, or companion-package dependency.

`ms2kinuv` is a separate ETL repository for environments that can read calibrated CASA Measurement Sets. Its command is:

```bash
ms2kinuv convert /path/to/calibrated.ms -o KILOGAS066.npz
```

## NPZ contract

| Key | Dtype | Shape | Required by kinUV |
|---|---|---|---|
| `u_m`, `v_m` | float32 | `(n_row,)` | yes |
| `vis` | complex64 | `(n_row, n_chan)` | yes |
| `weights` | float32 | `(n_row, n_chan)` | yes |
| `freqs` | float64 | `(n_chan,)` | yes |
| `time` | float64 | `(n_row,)` | yes for production time averaging |
| `baseline` | int64 | `(n_row,)` | yes for production time averaging |
| `phase_dir_rad` | float64 | `(2,)` | yes for phase-centre validation |
| `field_id` | int64 | scalar | optional provenance |
| `reference_dir_rad` | float64 | `(2,)` | optional provenance |
| `schema_version` | Unicode | scalar | written as `ms2kinuv-npz-v1` |

Baselines stay in metres. kinUV derives wavelengths independently for every channel as `(u_m, v_m) * frequency / c`. Flags are represented by zero weights. Extraction does not trim the line, average rows, bin uv cells, apply the spectral response, construct a model, or evaluate a likelihood; those operations belong to kinUV.

The existing KGAS007 product predates this contract and stores `u` and `v` at a reference frequency without time/baseline metadata. `kinuv.io.vis.load_visibility_table` contains an explicit historical reader so the accepted result remains reproducible. New products must use the metres-based ms2kinuv contract.
