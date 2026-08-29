# kinUV vs uvkin

Two repos. Do not copy forward-model code between them.

| | **kinUV** (this repo) | **uvkin** |
|---|---|---|
| Role | Production visibility fitter for 066 and the survey path | Legacy KinMS + UVfit/emcee campaign and science-matrix ops |
| Spectral operator | `kinuv.response.spectral.hann_then_bin` only | uvkin `bin_channels` / YAML `spectral_bin_factor` (not SPECRESP) |
| Fourier sign | `NPZ_UV_SIGN = -1` in `kinuv.transforms.dft` | Do not re-derive; if you FT a cube onto the 066 npz, match kinUV |
| Ico east | `fits_image_east_north` in `kinuv.forward.sb` | Imaging preflight only; not the vis MAP |
| Likelihood | `chi2 = s * sum w \|d-m\|^2` (`kinuv.likelihood.chi2`) | UVfit/KinMS chi2; not the Stage A product |
| Official 066 MAP | `kinuv-KGAS066-uvsign-map` | Not a product pointer |

`native_diagonal` is removed. Calling `kinuv.response.spectral.native_diagonal` raises. Gate 2 and S1 assert Hann+bin **before** mock vis.

uvkin stays for ms2uvfit conversion, science-matrix bookkeeping, and historical KinMS runs. New kinematics, leftover `chi2`, and survey MAPs go in kinUV.
