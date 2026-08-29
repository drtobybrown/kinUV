# kinUV — design plan and evidence base

**Status:** architecture closed after pre-build review (2026-08-18). No production code until approved.
**Date:** 2026-08-17 (reviewed 2026-08-18; ADR generation 2; methodology scorecard 2026-08-29)
**Source of truth:** mailbox [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md); ranking in [`DEC-066-INDEX`](docs/decisions/DEC-066-INDEX.md) (ADRs > Field Guide > STATUS > this file > Cursor plans). Handshake: [`DEC-066-AGENTS`](docs/decisions/DEC-066-AGENTS.md). This document is evidence, not policy.
**Context:** KILOGAS DR1, ALMA CO(2–1), 452 galaxies (283 MaNGA + 169 SAMI, z = 0.016–0.060).

**2026-08-29 evidence (not policy).** Phase-1 transform exit is met (FINUFFT T2 vs analytic Gaussian; error ≪ noise floor). Mission 10× is **not** that plot: it is S1 (sub-beam \(V_c\) and \(\sigma\) vs CLEAN) plus S2 (calibrated vis posteriors) plus Y1 (\(10^5\) evals in minutes). 066 vis MAP beats \(V=0\) and matches the 10 km/s cube after the CASA uv-sign / Ico-east fixes; S1/S2/Y1 are **not** shown (`r_t` on 0.5″ floor; no NUTS; ~2 eval/s CPU). Propose: [`docs/reviews/2026-08-29-propose-methodology.md`](docs/reviews/2026-08-29-propose-methodology.md).

---

## 0. Executive summary

I measured the existing `ms2uvfit → uvfit → uvkin` stack against analytic ground
truth and against the real `KILOGAS007.ms`. The headline result is that **this is
a correctness project before it is a performance project.**

The current forward model's numerical error is **2.7–9.6× larger than the
statistical noise floor** of the data it is being fitted to. The fits are
therefore systematics-limited, and the dominant systematics are numerical, not
astrophysical. Speed is a real but secondary problem.

The proposed engine is a **deterministic-quadrature forward model coupled to a
true NUFFT**, replacing Monte-Carlo cloud sampling plus FFT-and-bilinear-
interpolation. This simultaneously:

- removes the largest error term (Monte-Carlo cloud shot noise),
- improves degridding accuracy by ~5 orders of magnitude (measured),
- makes the model differentiable, unlocking HMC/NUTS,
- and runs faster on the same hardware (measured 4.9× before any GPU work).

Every number below is measured on this machine and reproducible via
[`benchmarks/`](benchmarks/).

---

## 1. Measured baseline

### 1.1 The data (KILOGAS007.ms, measured)

| Quantity | Value |
|---|---|
| Line SPW | one (`DATA_DESC_ID` 25), 1920 channels |
| Frequency range | 219.0655 – 220.9395 GHz |
| Channel width | 0.9766 MHz = **1.331 km/s** |
| Rows (line SPW) | 44,550 |
| Correlations | 2 |
| **Raw complex visibilities** | **171,072,000** |
| Flagged fraction | 9.7 % |
| Baseline lengths | 0 – 414 m (median 90 m) |
| uv distance | 0 – **305 kλ** (median 67 kλ) |
| max abs(w) | 216 m |

After the production aggregation (`uv_bin 10 m`, `time_bin 30 s`,
`spectral_bin_factor 8`), measured directly by counting occupied (time, uv-cell)
bins:

| Quantity | Value |
|---|---|
| Rows | 44,550 → **8,304** (×5.36 reduction) |
| Channels | 450 line channels → **56** |
| **Fitted visibilities** | **465,024** |
| σ per raw visibility | 0.879 Jy |
| σ per aggregated visibility | **94.9 mJy** |
| Source peak flux density | ~60 mJy / channel |
| **Per-visibility SNR** | **0.64** |
| Coherent noise floor σ/√N | **0.139 mJy** |

The per-visibility SNR of 0.64 is the crucial number: the science signal is
recovered only by coherently averaging ~5 × 10⁵ visibilities. In that regime a
*coherent* model error of fractional size ε contributes ε × 60 mJy, which must be
compared against 0.139 mJy — not against the per-visibility noise.

### 1.2 Accuracy against analytic ground truth

An elliptical Gaussian has a closed-form visibility function, so the degridder
can be scored absolutely rather than by comparison to another approximation.
Configuration is the production one (256², 0.1″, 56 channels, 15,000 rows,
uv ≤ 309 kλ).

| Method | Wall time | Throughput | max err | rms err |
|---|---|---|---|---|
| scipy bilinear (current), 256² @ 0.1″ | 369.6 ms | 2.27 Mvis/s | **2.21e-2** | **6.18e-3** |
| scipy bilinear, 128² @ 0.2″ | 214.4 ms | 3.92 Mvis/s | 2.21e-2 | 6.18e-3 |
| **FINUFFT type 2, tol 1e-6, plan reused** | **75.3 ms** | **11.15 Mvis/s** | **4.05e-8** | **8.27e-9** |
| FINUFFT type 3 (grid-free), 20k clouds | 254.0 ms | 3.31 Mvis/s | 1.67e-2 | 3.66e-3 |
| FINUFFT type 3 (grid-free), 100k clouds | 369.0 ms | 2.28 Mvis/s | 9.11e-3 | 2.06e-3 |

Two non-obvious findings:

1. **The bilinear error is identical for 256²@0.1″ and 128²@0.2″.** Both grids
   have the same 25.6″ field of view, hence the same uv-grid spacing
   `du = 1/(N·cell)`. Bilinear interpolation error scales as `du²·|V''|`, so it is
   controlled by the **image FoV, not the cell size**. Shrinking cells to chase
   accuracy does nothing; only padding the image helps. A real NUFFT achieves the
   same effect via internal oversampling plus deapodization, at a fraction of the
   cost.
2. **Type-3 (grid-free) error is Monte-Carlo shot noise**, falling as 1/√N_cloud
   (1.67e-2 → 9.11e-3 for 20k → 100k). This is the same error that KinMS carries.
   It is *not* removed by fixing the random seed.

### 1.3 The error budget — why this is a correctness project

Treating model error as coherent across all 465,024 aggregated visibilities and
comparing to the 0.139 mJy noise floor:

| Error source | fractional | coherent error | vs noise floor |
|---|---|---|---|
| FINUFFT @ tol 1e-6 | 4.05e-8 | 0.0000 mJy | **0.0×** |
| bilinear degridding (rms) | 6.18e-3 | 0.373 mJy | **2.7×** |
| bilinear degridding (max) | 2.21e-2 | 1.333 mJy | 9.6× |
| KinMS 20k clouds | 1.67e-2 | 1.008 mJy | **7.2×** |
| KinMS 100k clouds | 9.11e-3 | 0.550 mJy | 3.9× |

**Caveat, stated honestly:** this is the worst case. Degridding error varies with
uv position, so it partially averages down; the true factor lies between 1× and
the quoted value. But it does not average to zero, because the error is a smooth
deterministic function of uv position and of the model parameters. Monte-Carlo
cloud noise is worse in a subtler way: it is a fixed pseudo-random pattern that
*changes discontinuously as parameters move the clouds*, which both biases the
posterior and destroys any gradient signal.

The conclusion stands: **current numerical error is at or above the statistical
precision of the data.**

### 1.4 Speed

Production configuration (256² @ 0.1″, 56 channels), measured:

| Stage | Time |
|---|---|
| `KinMS.generate_cube` | 173 ms |
| degrid, 5,000 rows (280k vis) | 209 ms |
| degrid, 15,000 rows (840k vis) | 426 ms |
| degrid, 40,000 rows (2.24M vis) | 925 ms |
| **Total per likelihood** | **0.38 – 1.10 s** |

For emcee at 64 walkers × 5,000 steps = 320,000 evaluations: **34–98 core-hours
per galaxy** (4–12 h on 8 cores). At 64 × 20,000 steps: 136–390 core-hours. For
452 galaxies this is 1.5 × 10⁴ – 4.4 × 10⁴ core-hours.

Where the time goes, measured by reimplementing the identical bilinear math in
numba (agreement 1.0e-7, so this is a pure implementation comparison):

| Implementation | Time |
|---|---|
| scipy `RegularGridInterpolator`, rebuilt per channel | 334 ms |
| numba fused bilinear, parallel | 87.7 ms (3.8×) |
| — of which the **FFT alone** | **73.2 ms** |
| — of which interpolation | ~14 ms |

So once the Python/scipy object churn is removed, **the FFT dominates**, and the
FFT is oversized: 0.1″ cells give a uv grid reaching 1031 kλ while the data stop
at 305 kλ. That is 3.4× oversampling per axis — ~11× wasted transform work — and
it cannot be reduced by coarsening cells without incurring aliasing, because
the FoV must stay large for bilinear accuracy (§1.2). A NUFFT breaks this
deadlock: accuracy is set by a tolerance parameter, decoupled from both FoV and
cell size.

### 1.5 Two smaller findings

**The w-term is negligible and can be rigorously dropped.** With max abs(w) = 216 m,
the maximum phase error from the 2D approximation is 0.017° at 5″ from phase
centre, 0.067° at 10″, and 0.151° at 15″. No w-projection, w-stacking, or
faceting is needed for this array configuration. This is worth stating explicitly
because it removes a large amount of would-be complexity — but it must be
re-checked if KILOGAS ever includes extended-configuration data.

**`gnfw_circular_velocity` is grid-dependent.** It normalises the curve to the
maximum of `v_over_r` *on the supplied radius grid*, and integrates the enclosed
mass with `cumulative_trapezoid` from r = 0. Measured, for identical physical
parameters (vmax 200, r_s 3″, γ 1):

| grid r_max | V(1″) |
|---|---|
| 10″ | 144.58 |
| 30″ | 144.22 |
| 120″ | 137.71 |

and with r_max fixed at 15″, V(1″) moves from 143.11 (50 points) to 144.61 (2000
points). The sensitivity worsens with γ: at γ = 1.5 the inner velocity shifts
2.5 % between grids. This is a 1–2.5 % systematic on the inner rotation curve —
the exact quantity that discriminates cusp from core — and it is not reproducible
across configurations. gNFW has a closed-form enclosed mass (a hypergeometric
function); it should be used instead of quadrature on an arbitrary grid.

---

## 2. Questions I need answered before building

These are ordered by how much they change the design. Several are blocking.

### Blocking

**Q1 — Scope and repo politics.** There are already three repos: `ms2uvfit`
(MS I/O), `uvfit` (engine), `uvkin` (KILOGAS application layer), with a
documented separation doctrine in `uvkin/AGENTS.md`. Is kinUV:
 (a) a new engine that replaces `uvfit`'s internals, with `uvkin` retargeted at it;
 (b) a from-scratch stack that subsumes all three; or
 (c) a new `uvfit` backend selectable by config, so existing runs stay valid?
I recommend **(c) then (a)** — it lets the accuracy fix land in production within
days and keeps every existing result reproducible for comparison. But this is
your call and it determines everything downstream.

**Q2 — GPU hardware, honestly.** This machine is an M1 Pro with 16 GB and no
CUDA. The requirement for "native CUDA/ROCm acceleration" cannot be developed or
tested here. Do you have GPU access on CANFAR (which GPU, how many, what session
limits)? If not, the realistic target is a CPU-optimal engine with a CUDA
backend written behind an abstraction but validated only when hardware appears.
JAX on Apple Silicon runs on CPU reliably; `jax-metal` does not, and I would not
stake the project on it. **This single answer determines whether "1–2 orders of
magnitude" is achievable or whether the honest target is 5–15× on CPU.**

**Q3 — What is the actual throughput target?** The requirement says 10⁷–10⁹
visibilities. Your data is 1.7 × 10⁸ raw per galaxy but 4.7 × 10⁵ after the
aggregation you actually fit. These imply completely different designs:
 (a) *Fit aggregated data, 452 galaxies* — the bottleneck is per-galaxy latency
     and job orchestration; the win is a fast accurate kernel + gradients.
 (b) *Fit unaggregated data, per galaxy* — 1.7 × 10⁸ visibilities per likelihood.
     Currently 33 s per evaluation; achievable at ~0.3–3 s with the new engine,
     but only worth it if aggregation is actually costing you science.
 (c) Both.
Which is it? Related: **is the 10⁹ figure aspirational, or do you have a specific
dataset in mind** (a large program, an ALMA archive reprocessing) that I should
be designing for?

**Q4 — Baryons.** `uvkin/docs/agents/science-hypothesis.md` states that "in the
centres of KILOGAS galaxies, the stellar disk dominates the observed CO rotation
curve." The current model fits a **pure gNFW halo**. If baryons dominate the inner
rotation curve and the model has no baryonic component, then γ is not a halo
inner slope — it is absorbing the stellar disk. The cusp/core result would be an
artefact of model misspecification, and no amount of numerical accuracy fixes it.
Do you want kinUV to do a **full mass decomposition** (stellar disk + bulge from
MaNGA/SAMI MGE with M/L as a free or prior-constrained parameter, plus gas
self-gravity from the CO map itself, plus halo)? This is the single largest
*scientific* change and it substantially expands the parameter space. I think it
is mandatory for the stated science goal, but it is your hypothesis to defend.

### Statistical

**Q5 — Hanning covariance.** `weight_scale_factor: 0.5` is a scalar fudge for
what is really a **tridiagonal channel-channel covariance** (Hanning weights
0.25, 0.5, 0.25). The exact treatment is a banded GLS likelihood with a Cholesky
factor of the band matrix, which costs almost nothing (O(N_chan) per row). Do you
want exact, or is the scalar approximation acceptable? Note the requirement
document says "statistically exact Gaussian χ²", which the scalar factor is not.
Also: after ×8 spectral binning, adjacent *binned* channels are nearly
independent again, so the 0.5 factor may now be over-correcting — I can quantify
this precisely if you want.

**Q6 — Spectral response function.** Related and, I suspect, currently missing:
is the *model* convolved with the Hanning spectral response before comparison to
the data? If the data are Hanning-smoothed and the model is not, there is a
systematic line-width bias that maps directly onto `gas_sigma` and onto the
inner rotation curve via beam smearing. Can you confirm whether the DR1
visibilities are Hanning-smoothed, and whether online channel averaging was
applied?

**Q7 — Are the CASA weights trustworthy?** Measured median σ = 0.879 Jy per raw
visibility. ALMA weights are frequently mis-scaled after pipeline processing.
Should kinUV fit a per-SPW (or per-EB) weight rescaling nuisance parameter, and
report the empirical scatter of residuals as a check? I would default to yes:
it is cheap, and a wrong global weight scale directly corrupts Bayesian evidence
and all parameter uncertainties.

**Q8 — Aggregation is itself a systematic.** Time-averaging to 30 s and uv-binning
to 10 m causes amplitude decorrelation (time and bandwidth smearing) that is
baseline- and offset-dependent. Do you want kinUV to (a) forward-model the
smearing, (b) restrict aggregation to a regime where it is provably negligible,
or (c) fit unaggregated? I can quantify the current decorrelation for your
configuration in about an hour of work.

### Physical modeling

**Q9 — Primary beam.** Currently absent from the model. The ALMA 12 m primary
beam FWHM at 220 GHz is ~26″, and your model FoV is 25.6″. For galaxies of
10–20″ extent this is a 5–15 % attenuation in the outer disk that maps onto the
surface-brightness profile and hence the outer rotation curve. Include a
per-channel PB (Airy with blockage, or the CASA PB model)? Also: are these
data single-pointing or mosaics?

**Q10 — Continuum.** Is continuum subtracted in the uv-plane before export, or
should kinUV fit a joint continuum component? For CO(2–1) in these galaxies the
continuum is likely faint, but an unmodelled continuum biases the line wings and
hence `gas_sigma`.

**Q11 — Which non-circular motions actually matter for KILOGAS?** Radial
inflow/outflow, bar-driven m=2 streaming, warps, and lopsidedness are all
listed in the requirements. Implementing all of them well is a lot of surface
area. Given a MaNGA/SAMI sample at z ~ 0.02–0.06 with ~1.3″ beams, which do you
actually expect to constrain, and which are there to demonstrate extensibility?

### Process

**Q12 — What would convince you?** I would like to build a synthetic ALMA
benchmark with `simobserve` using your real uv coverage, inject a known
rotation curve, and require recovery within stated tolerances. Do you have an
existing truth dataset, or should I generate one? Relatedly: **is there an
existing result you want reproduced** (a KGAS007/066 fit whose posterior we
should recover, or deliberately improve on)?

**Q13 — Precision policy.** `uvfit` enforces a single-precision pipeline with
float64 accumulation in χ². With 4.7 × 10⁵ – 1.7 × 10⁸ visibilities, do you want
float64 throughout the transform (2× memory, ~2× time on CPU, up to 30× slower on
consumer GPUs), or float32 transforms with float64 reductions and a documented
error bound? I recommend the latter with a measured bound.

## 2.1 Resolved questions (from architecture review 2026-08-18)

The following questions from §2 have been resolved by the architecture review and supporting research.

**Q1 — Scope:** kinUV is a standalone fitter (option b, simplified). The uvfit selectable backend is deferred until the 066 MAP exists. Integration with the uvkin stack is deferred.

**Q2 — GPU:** This Mac is CPU only (M1 Pro, no CUDA). GPU = CANFAR only, astroml-cuda images. Local dev is JAX CPU. jax-metal is unsupported. Default survey job is CPU (class A: 4 CPU, 16 GB). GPU (class B) is the same model, faster, not a different science product.

**Q3 — Throughput:** Fit aggregated data (~5×10⁵ vis per galaxy) for DR1. Full-resolution (~1.7×10⁸) is a validation test after MAP, not the production path. The 10⁹ figure is aspirational for future large programs.

**Q4 — Baryons:** YES, mass decomposition is mandatory for the survey. For the 066 slice it is **deferred**. V_c(r) is the DR product, not γ. A later mass-model boundary separates cached `v_star_unit(r)` from the visibility hot path. γ is a post-hoc projection of V_c posteriors onto gNFW + v_star + gas, with Υ★ from IFU SPS and IMF treated as a systematic. No `massmodel.py` is in the 066 critical path.

**Q5 — Hanning covariance:** See DEC-066-SPECRESP. After Hann + bin-8, adjacent-bin correlation is **ρ = 3/58 ≈ 5.2%** (not 1.6%). Small enough for a diagonal MAP; not ignored if NUTS error bars are the product. Phase 1 weights are the empirical `s` only (Q7). Theoretical `12/29` is a synthetic unit test, not a 066 multiplier. Phase 2: banded GLS after 066 MAP.

**Q6 — Spectral response:** Hann `[0.25, 0.5, 0.25]` on the **native MS channel grid**, then the same software bin as the npz. ALMA default correlator window is Hann. **Forbidden:** Hann along an already-binned axis. See DEC-066-SPECRESP.

**Q7 — CASA weights:** Pre-MAP diagnostic: `s = 2.0 / ⟨w|V|²⟩_line-free` on the fit npz (sanity 0.3–1.5). Forensic 066: ⟨w|V|²⟩ = 2.42 → `s ≈ 0.826`. Cycle 3+ WEIGHT already uses EFFECTIVE_BW, so do **not** also apply `12/29`. See DEC-066-WEIGHT.

**Q8 — Aggregation:** Production fits aggregated data. The aggregated-vs-full-resolution agreement test runs once after MAP (gate 4 of the validation ladder), not before. Time/bandwidth smearing at 30s/10m is negligible for KILOGAS compact configs (max baseline 414m, time smearing <0.1% at 15" offset).

**Q9 — Primary beam:** **Superseded.** Mandatory for 066 because the Ico template is `pbcor=True`. Re-apply `A(x,y)` in the image plane before FINUFFT; never PB-correct visibilities. FWHM = `1.13 λ/D` ≈ 25.9″ at 224.5 GHz. At r=7.5″, A≈0.79 (~21% suppression). See DEC-066-PB.

**Q10 — Continuum:** UV-plane continuum subtraction is done in the ALMA pipeline (uvcontsub). Verify by checking that line-free channels in the .npz have zero mean. If residual continuum is detected, fit a per-row constant or first-order baseline.

**Q11 — Non-circular motions:** For 066 (regular disk, no bar): circular rotation only. Radial flow, warps, and m=2 streaming are deferred. Do not treat untracked scaffolding `src/kinuv/geometry.py` as existing product.

**Q12 — Truth dataset:** Validation ladder defined in the architecture (§4). No simobserve needed — the analytic thin-ring oracle (spatial J₀, spectral double-horn) is the reference. Mock recovery uses real 066 (u,v,ν) with injected parameters.

**Q13 — Precision:** Float32 transforms with float64 reductions and a measured error bound. The NUFFT at tolerance 1e-6 in float64 gives 4e-8 error; float32 gives ~1e-6. Both are far below the noise floor. Float64 accumulation in χ² is mandatory (float32 accumulation of ~5×10⁵ terms loses ~3 digits).

## 2.2 New decisions from the architecture review

Closed answers live in `docs/decisions/`. Mailbox: [`docs/architecture/STATUS.md`](docs/architecture/STATUS.md). Index: [`field-guide/index.md`](field-guide/index.md). Generation-2 physics is **provisional** until the other agent files a review.

| ID | Question | Answer | Owner |
|---|---|---|---|
| DEC-066-INDEX | Who wins on conflict? | ADRs > Field Guide > STATUS > PLAN.md > Cursor plans | planner |
| DEC-066-AGENTS | How do two chats share architecture? | Adversarial propose/review via `docs/reviews/`; user ties; no new DEC ids | planner |
| DEC-066-ZEROMODEL | Zero model for Δχ²? | V=0; report Δχ² not reduced χ² | 066-8-map |
| DEC-066-SPECRESP | Hanning? | Hann on native grid then bin; empirical `s` only; 12/29 is a unit test | 066-6-likelihood |
| DEC-066-WEIGHT | Weight scale? | `s=2/⟨w\|V\|²⟩`; not 0.5; not 12/29 | 066-6-likelihood |
| DEC-066-SB | Ico template? | Wiener restoring beam; K=(σ/I_peak)²; taper 0.05; observed ν | 066-2-template |
| DEC-066-PB | Primary beam? | Mandatory for 066 (pbcor Ico); A(x,y) before NUFFT | 066-2-template |
| DEC-066-OSCMETRIC | λ_reg ringing? | Ω_k/Δv_npz < 0.3; 20 mocks × 5 λ; spline prior not SE-GP | 066-4-rings |
| DEC-OPS-AUTH | CANFAR names/auth? | `kinuv-KGAS066-{sha6}-{map\|nuts}`; cert at launch; campaign cron deferred | ops |
| DEC-HIER-SELFUNC | Hierarchical selection? | Deferred Phase 5 | planner |

---

## 3. Proposed architecture

### 3.1 The central algorithmic idea

Replace **Monte-Carlo clouds → histogram into a cube → FFT → bilinear
interpolation** with **deterministic quadrature → NUFFT**.

The observable is

  I(l, m, ν) = ∫ ds ρ(x, y, z) · φ(v(ν) − v_los(x, y, z); σ_v(R, z)) · e^(−τ)

evaluated on a set of quadrature nodes in the **galaxy frame** (Gauss–Legendre in
R, uniform in azimuth φ, Gauss–Hermite in z), which are then mapped through
inclination, position angle, and warp to the sky frame and Fourier-transformed
to the (u, v) points of the data.

Why this is the right core:

- **Convergence is spectral, not 1/√N.** KinMS's Monte-Carlo error falls as
  1/√N_cloud — measured above as 1.67e-2 at 20k clouds and 9.11e-3 at 100k. A
  tensor-product Gauss quadrature on a smooth integrand converges exponentially
  in the number of nodes. This eliminates the single largest error term rather
  than throwing 100× more clouds at it.
- **It is differentiable.** Quadrature node positions and weights are smooth,
  closed-form functions of the parameters, so the whole map from parameters to
  visibilities is differentiable. Histogramming Monte-Carlo clouds into a cube is
  piecewise-constant and has zero gradient almost everywhere; that is the reason
  the current stack cannot use HMC, independent of which autodiff library is used.
- **It is deterministic and reproducible.** No seed dependence, no jitter in the
  likelihood surface.

### 3.2 The transform: three candidate paths, with a decision gate

I do **not** want to pick this on aesthetics. The plan is to implement the
abstraction and benchmark all three against analytic truth.

| Path | Description | Pros | Cons |
|---|---|---|---|
| **T2** | Deposit quadrature nodes onto a uniform sky grid with a smooth kernel, then FINUFFT **type 2** per channel | fastest measured on CPU (75 ms); mature autodiff via jax-finufft | reintroduces a grid, hence an aliasing bound to control |
| **T3** | FINUFFT **type 3** directly from galaxy-frame nodes to uv points | no grid, no aliasing, no pixelation, no band limit | 3–5× slower on CPU as measured; autodiff support needs verification |
| **DFT** | Direct exact sum on GPU | exact by construction, trivially differentiable | O(N_vis × N_node); measured 0.28 Gterm/s on 8 CPU cores — viable only on GPU |

Measured today: T2 with plan reuse is 75.3 ms at 4.05e-8 error; T3 at 20k nodes
is 254 ms and limited by node count, not by the transform. My expectation is that
**T2 wins on CPU and T3 or DFT becomes attractive on GPU**, but the benchmark
decides. The engine will expose a single `VisibilityTransform` interface so the
backend is swappable and the choice is empirical.

### 3.3 Differentiability

JAX is the primary target: `jax-finufft` provides CUDA-capable type-1/type-2
transforms with autodiff, and JAX gives HMC/NUTS (NumPyro/BlackJAX), vmap over
galaxies, and a single codebase for CPU and GPU.

Risk to manage explicitly: `jax-finufft` has to build against a CUDA FINUFFT, and
type-3 autodiff support needs verification (**an early spike task, not an
assumption**). The fallback is cheap and well-defined: the adjoint of a type-2
transform *is* a type-1 transform, and derivatives with respect to node positions
have closed forms (multiply by −2πi u). So a custom `jax.custom_vjp` wrapper is a
bounded piece of work if the upstream package disappoints.

### 3.4 Likelihood

Exact Gaussian likelihood with spectral response and weight calibration:

  −2 ln L = Σ_rows (Δv)† W_eff (Δv) + const

**Phase 1 (066):** Evaluate the model at native MS channels, convolve with Hann `[0.25, 0.5, 0.25]`, then bin with the same operator as the data (DEC-066-SPECRESP). **Forbidden:** Hann on the already-binned axis. After bin-8, inter-bin correlation is ρ ≈ 5.2%. Diagonal MAP: `W_eff = diag(w × s)` with empirical `s` only.

The weight calibration is a **pre-MAP** diagnostic:
1. Select line-free channels (outside the CO window).
2. Compute ⟨w|V|²⟩. Expect 2.0 if weights match Re/Im variance.
3. `s = 2.0 / ⟨w|V|²⟩_line-free` per SPW (sanity 0.3–1.5).
4. Forensic KGAS066: ⟨w|V|²⟩ = 2.42 → `s ≈ 0.826`. Do **not** also multiply by 12/29.

**Phase 2:** Full banded GLS after 066 MAP exists. Adjacent-bin covariance at N=8 is ρ = 3/58. Banded Cholesky is O(N_chan) per row.

**Phase 2:** Full banded GLS with the exact channel covariance from Hanning + binning. The covariance matrix C is tridiagonal after Hanning (bandwidth 1) and remains narrow-banded after bin-8 (bandwidth 1, since only adjacent bins share correlated samples). Banded Cholesky factorisation costs O(N_chan) per row.

Includes:
- Per-SPW weight rescaling nuisance parameters.
- Float64 accumulation (mandatory for ~5×10⁵ terms).
- The ln det C term retained for posterior normalisation (needed for Savage-Dickey density ratio if γ posteriors are ever computed; not for discrete model comparison via Bayes factors).

### 3.5 Scaling and memory

The log-likelihood is a sum over visibility chunks, so it is exactly
decomposable. Implementation: chunked iteration over a memory-mapped or Zarr
visibility store, with per-chunk model evaluation and χ² accumulation, and a
fixed memory ceiling independent of dataset size. On a 16 GB machine this is
mandatory even for a single galaxy at full resolution (171M visibilities ×
complex64 = 1.4 GB for the data alone, before model and residual arrays).

Across the survey, 452 galaxies are embarrassingly parallel — the orchestration
layer matters more than intra-galaxy parallelism for total throughput, which is
why **Q3** matters so much.

### 3.6 Module layout (proposed)

```
kinuv/
  io/          MS reader (python-casacore), .npz back-compat, Zarr store, FITS cubes
  geometry/    sky <-> galaxy frame transforms, warps, PA(r), i(r) — pure, tested
  profiles/    surface brightness, rotation curves, dispersion, scale height
               (registry + decorator for user extensions)
  dynamics/    mass components: disk, bulge, gas, NFW/gNFW/core, SMBH; v_c from potential
  lineshape/   Gaussian, Gauss-Hermite, custom; optical depth
  quadrature/  deterministic node generation, convergence control
  transform/   T2 / T3 / DFT backends behind one interface; CPU + CUDA
  response/    primary beam, spectral response, channelisation
  likelihood/  banded GLS, weight calibration, evidence-ready normalisation
  inference/   NUTS, VI, emcee, nested sampling; diagnostics (R-hat, ESS, tau)
  diagnostics/ uv-domain residuals, synthesized residual cubes, posterior products
  validate/    analytic benchmarks, convergence tests, gradient checks
```

The `profiles/` and `dynamics/` registries are where the "plug in your own
component via a decorator" requirement is satisfied.

---

## 4. Verification strategy

Non-negotiable, because the entire argument for this project is numerical
correctness.

1. **Analytic visibility tests.** Gaussian, uniform disc, and thin ring all have
   closed-form visibility functions. Assert agreement to < 1e-7 (already
   demonstrated: 4.05e-8). This is the test the current engine fails at 2.2e-2.
2. **Transform adjointness.** `<F x, y> == <x, F† y>` to machine precision.
3. **Quadrature convergence.** Refine node counts and assert the model
   visibilities converge at the expected rate and plateau at the transform
   tolerance — the property Monte Carlo cannot have.
4. **Gradient verification.** Every analytic/AD gradient checked against
   high-order finite differences and against complex-step differentiation where
   applicable.
5. **Parameter recovery.** Synthetic data generated by an *independent* code path
   (ideally `simobserve` with real KILOGAS uv coverage), fitted blind, requiring
   unbiased recovery within stated credible intervals.
6. **Statistical calibration.** Simulation-based calibration: rank statistics of
   truth within the posterior must be uniform. This is the only real proof that
   the "exact posteriors" claim holds.
7. **Regression against existing results.** Reproduce a current KGAS007/066 fit
   and explain every difference, so we can tell an improvement from a bug.
8. **CI policy inherited from uvkin:** synthetic data only, no DR1 in CI.

### 4.1 Performance budget (refined)

MAP on 066 (aggregated, ~881 rows × 125 channels):
- FINUFFT T2: ~75 ms per eval (measured)
- Quadrature + profile evaluation: ~5 ms (768 nodes, 4 profiles)
- Channel integration (erf) at **native** channels, then Hann, then bin (cost scales with N, not with the binned axis)
- Do not budget “Hanning ~0.1 ms on 125 channels” — that is the forbidden binned-axis convolution
- χ² accumulation: ~2 ms
- **Total per likelihood eval: ~110 ms** (measured FINUFFT dominates)
- L-BFGS with 100-200 gradient evaluations: **15-30 seconds**
- With JAX JIT warmup on first call: **1-5 minutes total**

If MAP exceeds 30 minutes, investigate: either the grid is too large (Nyquist violation), or a non-smooth component is in the model.

---

## 5. Phased roadmap

Each phase has a measurable exit criterion. Phases 1–2 deliver science value
before any GPU work happens, which is deliberate: the accuracy fix is worth more
than the speed fix, and it is not hardware-blocked.

**Phase 0 — Decisions and harness (days).** Resolve Q1–Q4. Stand up the repo,
CI, and the analytic benchmark suite. *Exit: benchmarks run in CI and reproduce
the numbers in §1.*

**Phase 1 — Exact transform, drop-in (1–2 weeks).** FINUFFT-backed
`VisibilityTransform` with the banded-covariance likelihood, exposed as an
alternative backend inside `uvfit` (per Q1c). CPU only.
*Exit: < 1e-7 against analytic truth, ≥ 4× faster than the current degridder,
and a rerun of KGAS066 with a documented explanation of every posterior shift.*

**Phase 2 — Deterministic differentiable forward model (2–3 weeks).** Quadrature
disk model replacing KinMS Monte Carlo; analytic gNFW; baryonic components
(pending Q4); PB and spectral response (Q6, Q9).
*Exit: model error below the noise floor by ≥ 10×, convergence demonstrated to
be spectral, KinMS agreement in the limit of many clouds.*

**Phase 3 — Gradients and modern inference (2 weeks).** JAX port, NUTS via
NumPyro/BlackJAX, R-hat/ESS diagnostics, nested sampling for evidence.
*Exit: NUTS and emcee posteriors agree; ESS per wall-clock second improves by
≥ 10×; SBC calibration passes.*

**Phase 4 — GPU (hardware-dependent, see Q2).** CUDA backend via
cufinufft/jax-finufft; batched multi-galaxy execution.
*Exit: measured order-of-magnitude throughput gain on real hardware, bitwise-
comparable results to CPU within tolerance.*

**Phase 5 — Scale-out (1–2 weeks).** Chunked/streaming visibility access,
unaggregated fitting, CANFAR orchestration for 452 galaxies.
*Exit: a full-resolution single-galaxy fit within a fixed memory ceiling; survey
run costed and demonstrated on a subset.*

**Phase 6 — Physics breadth.** Non-circular motions, warps, multipole expansions,
SMBH Keplerian cores, Gauss-Hermite lineshapes, optical depth (Q11).

---

## 6. Risks

| Risk | Mitigation |
|---|---|
| No GPU access materialises (Q2) | CPU-optimal engine still delivers ~5–15×; state the honest target rather than promising 100× |
| `jax-finufft` CUDA build or type-3 autodiff fails | Custom `jax.custom_vjp`; adjoint is a type-1 transform, closed-form position derivatives |
| Quadrature struggles on sharp features (rings, truncations) | Adaptive/panelled quadrature; convergence test in CI catches it |
| Baryon decomposition (Q4) enlarges parameter space beyond identifiability | Prior-constrained M/L from MaNGA/SAMI; explicit degeneracy analysis in the posterior |
| Rewrite diverges from a working production pipeline | Ship as a selectable backend first (Q1c); regression-test against current results |
| macOS OpenMP clash between the FINUFFT wheel and the conda stack | Observed and worked around during benchmarking; install FINUFFT from conda-forge so a single `libomp` is linked |

---

## 7. What I deliberately did not propose

- **w-projection / faceting.** Measured max phase error 0.15° (§1.5). Not needed
  for this array configuration. Revisit for extended-configuration data.
- **Image-plane deconvolution of any kind.** Correctly excluded by the science goal.
- **A bespoke NUFFT implementation.** FINUFFT is state of the art, is what the
  performance target implicitly refers to, and reimplementing it would be a
  months-long detour with no scientific payoff.
