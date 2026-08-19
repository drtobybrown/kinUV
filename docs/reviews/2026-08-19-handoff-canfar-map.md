---
role: proposer
date: 2026-08-19
agent: chat-A
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-ZEROMODEL
  - DEC-OPS-AUTH
  - DEC-066-OSCMETRIC
verdict: propose
---

# Handoff: CANFAR CPU Stage A MAP for KGAS066 (066-11)

**Read this file after** `AGENTS.md` → `field-guide/index.md` → `docs/architecture/STATUS.md`. Canon generation 4. Do not create a `DEC-*` id. You are a Cursor agent on a CANFAR Skaha **CPU** session. You **git clone** this repo, then run **one Stage A MAP**. You do not NUTS, dynesty, rings, or 066-9.

## First actions (clone, do not copy from the laptop)

```bash
git clone git@github.com:drtobybrown/kinUV.git
cd kinUV
git fetch origin
git checkout dev
git pull origin dev
git rev-parse HEAD
git rev-parse --short=6 HEAD   # sha6 for DEC-OPS-AUTH
```

Repo (private): `https://github.com/drtobybrown/kinUV`. Branch **`dev`**.  
Session name: `kinuv-KGAS066-{sha6}-map` with sha6 from **this clone’s HEAD**. Do not invent a sha.

## Why you exist (do not redo science)

Laptop Stage A is **done** on `dev`. Δχ² vs V=0 = **+26213**. Interior: vsys=8098.7 km/s (radio), PA=381.86°≡21.9°, V_0=268.4 km/s, σ=11.7 km/s, (dx,dy)=(−0.10″, −0.06″), flux=60.6 Jy. **`r_t` is on the 0.5″ floor** — record it; do not add a V_0 prior or a new DEC.

This job is the **066-11 official MAP artifact** on `/arc` (same model, CANFAR paths). It is not a new inference. If Δχ² is not within ~10% of +26213 or V_0 hits 0, **stop** and write a note; do not NUTS.

## Hardware: CPU, not GPU

| Option | Verdict |
|---|---|
| **Skaha CPU, 4 cores, 16 GB RAM, no GPU** | **Do this.** Local wall ~28–30 min (nfev=1350, eval ~1 s). Session timeout **≥ 4 h**. Peak RAM ≪ 16 GB (1 GB npz mmap + ~92² cube). |
| GPU / `astroml-cuda` / `jax[cuda12]` | **No.** `jax-finufft` CUDA is unproven (`benchmarks/bench_gpu_canfar.py` is not evidence until it runs). Queue + image risk exceeds a 30 min CPU job. Same science product. |
| Local M1 again | Already have the vector. This session is the `/arc` record. |

Do not import uvkin or KinMS. Do not Hann the data. Do not fit 1920 native channels. Do not freeze `(dx, dy)`. Do not apply a vis phase ramp after PB.

## Paths (laptop defaults will fail on CANFAR)

`load_kgas066` / `load_sb_template` default to `/Users/thbrown/...`. **Always pass CANFAR paths.**

| File | CANFAR (try in order) |
|---|---|
| Native npz (997305244 bytes) | `/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz` |
| Ico moment-0 | `/arc/projects/KILOGAS/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits` then `ls /arc/projects/KILOGAS/analysis/**/KGAS66_Ico_K_kms-1.fits` |
| Clipped cube (VOPT trim) | sibling `KGAS66_clipped_cube.fits` (optional: loader has hardcoded 8034–8536 if missing) |

**Abort if Ico is missing.** Exponential fallback is not the 066 product.

Fit array must be **881 × 95**, N=4, Δv≈5.080 km/s, s≈0.514 (sanity 0.3–1.5). Not YAML 0.5, not `12/29`.

## Env

Do **not** conda-forge numpy/scipy + pip `jax-finufft` on a machine that already showed dual `libomp` (see `environment.yml`). On Skaha Linux, use a **venv + pip** stack:

```bash
python3.11 -m venv $HOME/kinuv-venv
source $HOME/kinuv-venv/bin/activate
pip install numpy scipy astropy "jax>=0.4.30" jax-finufft pytest
export PYTHONPATH=/path/to/kinUV/src
python -c "import jax_finufft, jax; print(jax.devices())"
```

Expect CPU devices. `cadc-get-cert` **before** `canfar create` (DEC-OPS-AUTH). Cert check: `openssl x509 -enddate -noout -in ~/.ssl/cadcproxy.pem`.

## Run (copy this)

```python
#!/usr/bin/env python3
"""066-11 Stage A MAP on CANFAR. Do not NUTS. Do not rings."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from kinuv.forward.sb import load_sb_template
from kinuv.infer.map import image_grid_for_vis, run_stage_a_map
from kinuv.io.vis import load_kgas066

NPZ = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/visibilities/KILOGAS066.npz")
ICO = Path("/arc/projects/KILOGAS/analysis/kinms_test/kgas066/KGAS66_Ico_K_kms-1.fits")
CUBE = Path("/arc/projects/KILOGAS/analysis/kinms_test/kgas066/KGAS66_clipped_cube.fits")
# Replace SHA6 with `git rev-parse --short=6 HEAD` from the clone.
OUT = Path("/arc/projects/KILOGAS/analysis/toby_sandbox/results/KILOGAS066/kinuv-KGAS066-SHA6-map")

def main() -> None:
    if not NPZ.is_file():
        raise FileNotFoundError(NPZ)
    if not ICO.is_file():
        raise FileNotFoundError(f"Ico required (no exponential fallback): {ICO}")
    cube = CUBE if CUBE.is_file() else None
    data = load_kgas066(NPZ, cube_path=cube)
    print(
        f"fit array {data.vis.shape} N={data.n_bin} dv={data.dv_kms:.4f} s={data.s:.4f}",
        flush=True,
    )
    if data.vis.shape != (881, 95):
        raise RuntimeError(f"unexpected fit shape {data.vis.shape}")
    grid = image_grid_for_vis(data)
    tmpl = load_sb_template(grid, ico_path=ICO)
    rec = run_stage_a_map(data, template=tmpl, grid=grid)
    OUT.mkdir(parents=True, exist_ok=True)
    payload = {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in asdict(rec).items()}
    (OUT / "stage_a_map.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2), flush=True)
    if rec.delta_chi2 <= 0.0:
        raise SystemExit("MAP_LOSES_TO_ZERO")
    if rec.v0_kms < 1.0:
        raise SystemExit("MAP_STILL_COLLAPSED")

if __name__ == "__main__":
    main()
```

Log: `stage_a_map.json` plus stdout. Expected vs laptop: Δχ² ~ **2.6e4**, vsys ~ **8100** km/s radio, PA ~ **22°** or **382°**, V_0 ≫ 0, σ not 50, `|dy|` not 2″. `r_t` may sit at 0.5″.

## Forbidden

- NUTS / dynesty / `nuts` session name
- Stage B rings / `run_lambda_reg_campaign` (still `NotImplementedError`; `λ_reg` uncalibrated — DEC-066-VC)
- Changing DEC-066-PA seed
- YAML `obs_freq_range` as the spectral trim
- Growing files past 400 lines of Python (`BLOATED:`)

## After this job (not you)

Gate 4: 20×5 `λ_reg` campaign (`select_lambda_reg` exists; the campaign loop does not). Then Stage B only if AIC beats Stage A. Then NUTS (`kinuv-KGAS066-{sha6}-nuts`) if R̂/ESS gates. XX+YY is 066-9.

## STATUS updates required

- `last_propose:` this file
- Session `kinuv-KGAS066-{sha6}-map` when created (sha6 = clone HEAD)
- `next_role: proposer`
