---
role: proposer
date: 2026-08-21
agent: canfar-066-12-diagnosis
canon_generation: 4
ids:
  - DEC-066-OSCMETRIC
  - DEC-066-VC
  - DEC-066-INFER
verdict: propose
---

# Gate 4 is a metric–physics incompatibility, not a missing λ

**Read after** `AGENTS.md` → field-guide → STATUS → `docs/reviews/2026-08-21-report-gate4-stage-b.md`. Canon generation 4. No new `DEC-*` id (field-guide stop: new design choice). No NUTS. No rings on real 066.

## Scope

Close the executable 066-12 path. Name the blocker at the equation level. Recommend (do not implement) what a user-authored DEC would have to change if Stage B is still wanted.

## Verdict

**There is no path forward inside DEC-066-OSCMETRIC as written.** Densifying λ, bumping N_rings again, or running Stage B “anyway” are all forbidden *and* would fail on the physics. Official 066 kinematics remain **Stage A MAP**. Next code change that reopens rings requires a **user-created** DEC; this chat must not invent one.

## The blocker (why λ cannot exist)

OSCMETRIC defines

`Ω_k = |V_{k+1} − 2 V_k + V_{k−1}| / Δv_chan`  accept if `max_k Ω_k < 0.3`

On the 066-6 fit array, `Δv_chan ≈ 5.080 km/s`, so the gate is **|Δ²V| < 1.52 km/s** between adjacent knots.

That quantity is a **raw second difference**, not `d²V/dr²`, and it is **not** residual ringing. A concave-down turnover (every arctan, every rising-then-flat disk) has Δ²V < 0 of order the amplitude of the bend.

Sample the **noise-free calibration truth** `(V₀=200 km/s, r_t=3″)` on OSCMETRIC knots (`r_0=0.65″`, `r_last=7.5″`):

| N | Δr | max |Δ²V| | max Ω (Δv=5.08) | passes 0.3? |
|---|---|---|---|---|
| 6 | 1.37″ | 16.0 km/s | **3.15** | no |
| 7 | 1.14″ | 11.4 km/s | **2.24** | no |
| 8 | 0.98″ | 8.4 km/s | **1.65** | no |

The true curve is already **5–10×** above the gate. λ_reg’s prior *is* a Gaussian on those same second differences. Driving max Ω below 0.3 **requires suppressing the turnover**, i.e. pulling V(r) toward a straight line (Δ²V→0). Endpoint-linear rings on the N=7 knots recover **V₀≈258 km/s, r_t≈6.4″** — the same direction as the λ=100 campaign (V₀≈217, r_t≈3.56). That bias is the regulariser doing what Ω asked, not a fitter bug.

Campaign numbers are then obligatory:

- λ≤1: ⟨V₀⟩≈201 (scientifically recovered) but ⟨max Ω⟩≈1.3–2.5, **0%** Ω-pass (the fitter is allowed to keep truth curvature, which the gate calls “ringing”).
- λ=10: Ω still ~0.74 (fail); V₀ already +6–8 km/s.
- λ=100: Ω-pass **100%**; joint 1σ (V₀,r_t) recovery **0%**.

No value between 10 and 100 can satisfy both: Ω=0.3 sits in the already-biased regime. N=8 vs 7 only shrinks Δ²V as (Δr)²; reaching Ω_truth<0.3 at this Δv would need Δr≈0.4″ ≈ **17 knots**, outside DEC-066-VC (6–8). Using YAML Δv=10.6 km/s still leaves N=8 truth Ω≈0.79>0.3.

A second, independent failure of criterion 2: even at λ=1, mock-scatter “1σ coverage” is ~0.40–0.45 < 0.68, because a ~1 km/s discretisation bias is comparable to the ~1 km/s noise scatter. That test would fail **even if Ω were ignored**. It is a coverage test on shared-uv systematics, not a 10 km/s science tolerance (at λ≤10, |ΔV₀|<10 km/s in 100% of mocks).

OSCMETRIC §4 already names this: *if the three criteria conflict at all ring counts, that is a model-specification failure, not a licence to drop the metric.* N=7 and N=8 conflict the same way. Stop.

## Path forward that was rejected (and why)

| Action | Why not |
|---|---|
| Densify λ (OSCMETRIC §2) | Authorized *only* when criteria conflict **and a passing λ might exist**. Truth Ω>0.3 ⇒ empty set. |
| N_rings=6 | Increases |Δ²V|. ADR was increase, not decrease. |
| N_rings>8 | New DEC (VC cap). ~17 rings to make *truth* pass absolute Ω. |
| Drop Ω or raise 0.3 | New DEC. Dropping Ω is explicitly forbidden. |
| Stage B on real vis with λ=1 or λ=100 | VC: calibrate before real vis. λ=1 fails Ω; λ=100 fails recovery. AIC-vs-A is moot. |
| NUTS | INFER order: MAP rings only after λ selected. |
| Redefine Ω in code without a DEC | Field-guide stop. |

## What *is* executable (done)

1. Keep **Stage A** (`kinuv-KGAS066-f47bc9-map/stage_a_map.json`): Δχ²=+26213, V₀=268 km/s, r_t on the 0.5″ floor. That floor is a Stage A bound, not a Stage B product.
2. Lock the incompatibility: `test_truth_arctan_omega_exceeds_gate_at_oscmmetric_knots`.
3. This propose: mailbox for a reviewer. Do not ACK in this turn.

## If the user still wants Stage B (recommend, do not implement)

A new DEC would have to replace **absolute** Ω with **excess curvature relative to a smooth curve**, e.g. Ω on `V_k − V_arctan(r_k)` (or vs a low-order polynomial / the λ→0 MAP), so a turnover is not scored as ringing. Optionally replace mock-scatter coverage with a science window (e.g. |ΔV₀|<10 km/s and |Δr_t|<0.5″ in ≥68% of mocks). Knot count and Δv stay as VC/VIS. Until that DEC exists and is ACK’d, **do not reopen the campaign**.

## Residual risks

1. Stage A `r_t=0.5″` means the 066 arctan is a beam-floor scale, not a measured inner radius. Rings were supposed to relax that; they are not licensed.
2. A reviewer might argue Δv should be native 1.27 km/s — that makes Ω *worse*.
3. Two-agent handshake: this propose must be challenged or ACK’d in a **later** turn. User is the only one who may add a DEC stub.

## STATUS updates required

- `next_role: reviewer`
- `pending: []` (no new id opened)
- `last_propose:` this file
- Official product: Stage A only
- `deadlocks:` none (spec failure is decisive; not a two-agent disagreement)
