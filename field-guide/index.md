# kinUV Field Guide (066)

Inject at start. Budget: 80 lines. Essays: `docs/decisions/`. Rank: `DEC-066-INDEX`. Handshake: `DEC-066-AGENTS` (dual-review board).

## Mailbox

Read `STATUS.md` every turn. Parent **proposes**; two independent sub-agents write `review-a` / `review-b`. Dual `accept` → implement the named stages; no third review. User reviews **final fit plots**, not gates. Rubber-stamp is a process failure. Human science: `docs/methodology.md`. Board: `docs/reviews/BOARD.md`. `code_freeze: false`. No new `DEC-*` id. Official MAP: `kinuv-KGAS066-uvsign-map`. Sampler label `laplace_mh` is the MH path; 066 receding product is `sampler: nuts`. G3 receding NUTS landed. Leftover + PA 25.2 in flight. S1/methodology notes landed (not ADR). Do not start G4.

## Gates (implementer decides)

Prefer the DEC. If you leave it, STATUS one-liner and continue. Do not wait for the user. Still forbidden: new `DEC-*`, in-place overwrite of `kinuv-KGAS066-uvsign-map`, labeling Laplace-MH as NUTS, committing secrets. Split files if BLOATED; do not stop the build.

## 066 gates (in order)

1. Analytic Gaussian + thin-ring, transform error < 1e-7.
2. Mock on real 066 uv: recover flux, PA, vsys, 0.3″ (dx,dy).
3. Real MAP: report Δχ² vs V=0, not reduced χ².
4. Injected V_c within beam-scale covariance (S1 vis recovered; cube did not).
5. Then sampling: report `R_hat`/`ESS`; do not call Laplace-MH "NUTS". Laplace SBC failed 68/95.

## DEC ids (closed)

| ID | Answer |
|---|---|
| INDEX | ADRs > Field Guide > STATUS > PLAN.md > Cursor plans |
| AGENTS | parent proposes; dual-board accept; user ties; no new DEC ids |
| TARGET | KGAS066 only |
| INC | 43.9° freeze, ±5° |
| PA | fit; seed 205.2° receding |
| SB | Wiener; pad ≥2×; clip only if centroid shift <0.01″ |
| PB | A after image (dx,dy); FWHM=1.13 λ/D ≈25.9″ |
| VC | arctan then 6–8 rings; outer flat; inner solid-body |
| OSCMETRIC | r0≥0.5 BMAJ; Ω/Δv<0.3; 20×5 λ; AIC |
| SHIFT | Fourier/spline image shift then PB; σ=0.5″ |
| INFER | MAP then sample; Laplace CIs not calibrated |
| VIS | aggregated npz; record N, Δv |
| SPECRESP | Hann native+guards then bin; empirical s |
| WEIGHT | s=2/⟨w\|V|²⟩; not 0.5; not 12/29 |
| POL | XX+YY re-export |
| GRID | Nyquist vs 305 kλ; npz `(u,v)→(−u,−v)` in −2πi kernel |
| ZEROMODEL | V=0; Δχ² |
| REPO | standalone kinUV |
| OPS-AUTH | `kinuv-KGAS066-{sha6}-{map\|nuts}` |
| HIER-SELFUNC | deferred Phase 5 |

## Imaging products (CANFAR)

Root: `/arc/projects/KILOGAS/products/v1.3/original/by_galaxy/KGAS66/`
- Ico / vis-trim / SB: `30kms/`
- Stage B vs imaging: `10kms/`. Method: `docs/diagnostics/stage-b-vs-imaging.md`. Style: `docs/diagnostics/plotting.md`.

## Git

Branch `dev`. Commit and push `origin/dev` after each propose, board tally, and stage deliverable. Conventional subject; do not skip hooks. High-frequency I/O: [`docs/diagnostics/scratch.md`](../docs/diagnostics/scratch.md) (`/scratch`, not `/arc`). CPU headless NUTS (flexible default; 4×1-chain parallel): [`docs/diagnostics/canfar-cpu-parallel.md`](../docs/diagnostics/canfar-cpu-parallel.md). GPU rejected: [`docs/architecture/notes/2026-09-02-gpu-rejection-cpu-parallel.md`](../docs/architecture/notes/2026-09-02-gpu-rejection-cpu-parallel.md).
