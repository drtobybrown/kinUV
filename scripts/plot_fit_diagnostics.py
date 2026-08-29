#!/usr/bin/env python3
"""Standard MAP diagnostic suite (leftover chi2 + optional imaging D/M/R).

Not a likelihood. Hann+bin vis chi2 is the fit. Figures follow
``docs/diagnostics/plotting.md`` via ``kinuv.diagnostics.style``.

1. leftover chi2 vs uv-distance and velocity
2. moments / spectra / PV if a model cube is present (``--imaging``)
3. chi2 slices are S1-only (expensive); use ``scripts/run_s1_mock.py``

Official MAP: ``kinuv-KGAS066-uvsign-map``. Preview writes default to
``docs/reviews/artifacts/fit-diagnostics/`` (gitignored).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from kinuv.diagnostics.s1 import CANFAR_NPZ, MAP_DIR
from kinuv.diagnostics.s1 import assert_hann_bin_operator

REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / "docs" / "reviews" / "artifacts" / "fit-diagnostics"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--npz", type=Path, default=CANFAR_NPZ)
    p.add_argument("--map-dir", type=Path, default=MAP_DIR)
    p.add_argument("--out", type=Path, default=SCRATCH)
    p.add_argument(
        "--imaging",
        action="store_true",
        help="Also write moments/spectra/PV via plot_stage_b_vs_imaging",
    )
    args = p.parse_args(argv)
    assert_hann_bin_operator()
    args.out.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from plot_leftover_chi2 import main as leftover_main

    leftover_main(["--npz", str(args.npz), "--map-dir", str(args.map_dir), "--out", str(args.out)])
    if args.imaging:
        from plot_stage_b_vs_imaging import main as imaging_main

        imaging_main(
            [
                "--stage-a",
                str(args.map_dir / "stage_a_map.json"),
                "--model-cube",
                str(args.map_dir / "stage_b_model_cube.fits"),
                "--out-dir",
                str(args.out),
                "--matched-fits",
                str(args.out / "model_on_10kms.fits"),
            ]
        )
    print(f"diagnostics -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
