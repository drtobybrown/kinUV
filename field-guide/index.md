# kinUV Field Guide (066)

Inject at start. Budget: 80 lines. Essays: `docs/decisions/`. Rank: `DEC-066-INDEX`. Handshake: `DEC-066-AGENTS`.

## Mailbox (two agents)

Read `docs/architecture/STATUS.md` every turn. Follow `next_role`. Propose and review only via `docs/reviews/` (see `_template.md`). Rubber-stamp ACK is a process failure. Cursor plans without `canon_generation` matching STATUS are stale. `code_freeze: true` → no Python. Do not create a `DEC-*` id.

## Stop conditions

- Need a new design choice → stop; do not create a `DEC-*` id.
- MAP cannot beat zero model → stop; do not try dynesty/NUTS.
- Restored Ico into FINUFFT as intrinsic SB → stop (DEC-066-SB).
- Hann on already-binned channels → stop (DEC-066-SPECRESP).
- `(dx,dy)` frozen at 0 before MAP → stop (DEC-066-SHIFT).
- Growing a file past 400 lines of Python → flag `BLOATED:`, do not add more.

## 066 gates (in order)

1. Analytic Gaussian + thin-ring, transform error < 1e-7.
2. Mock on real 066 uv: recover flux, PA, vsys, 0.3″ (dx,dy).
3. Real MAP: report Δχ² vs V=0, not reduced χ².
4. Injected V_c within beam-scale covariance.
5. Then NUTS: R̂<1.01, ESS>200 on PA, vsys, flux.

## DEC ids (closed)

| ID | Answer |
|---|---|
| INDEX | ADRs > Field Guide > STATUS > PLAN.md > Cursor plans |
| AGENTS | adversarial propose/review; user ties; no new DEC ids |
| TARGET | KGAS066 only |
| INC | 43.9° freeze, ±5° |
| PA | fit; seed 205.2° receding |
| SB | Wiener restoring beam; K=(σ/I_peak)²; taper 0.05 |
| PB | A(x,y) before NUFFT; FWHM=1.13 λ/D ≈25.9″ |
| VC | arctan then 6–8 rings |
| OSCMETRIC | Ω_k/Δv<0.3; 20 mocks × 5 λ; AIC vs arctan |
| SHIFT | (dx,dy) in MAP; ramp; σ=0.5″ |
| INFER | MAP then NUTS |
| VIS | aggregated npz; record N, Δv |
| SPECRESP | Hann native then bin; empirical s only |
| WEIGHT | s=2/⟨w\|V\|²⟩; not 0.5; not 12/29 |
| POL | XX+YY re-export |
| GRID | Nyquist vs 305 kλ; no imaging override |
| ZEROMODEL | V=0; Δχ² |
| REPO | standalone kinUV |
| OPS-AUTH | `kinuv-KGAS066-{sha6}-{map\|nuts}` |
| HIER-SELFUNC | deferred Phase 5 |

## Git

Branch `kgas066-slice`. First commit: this guide + `docs/decisions/` + `.gitignore`. No `src/` scaffolding. One component id per commit. Code starts only after architecture is approved.
