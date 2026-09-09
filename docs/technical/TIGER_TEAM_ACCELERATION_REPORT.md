# kinUV Tiger Team Acceleration Report

**Date:** 2026-09-09  
**Scope:** CANFAR headless visibility-domain NUTS for KGAS066 and KGAS007  
**Team:** HPC/topology, NUTS/likelihood, GPU/surrogates

## Priority ranking

Speedups are realistic wall-time ranges relative to the current shared-session,
1,000-warm-up plus 1,000-draw campaign. They are not promises and are not all
multiplicative.

| Rank | Optimization Action | Area (HPC/NUTS/AI/GPU) | Est. Speedup | Implementation Time | Risk/Complexity |
|---:|---|---|---:|---:|---|
| 1 | Run one flexible CANFAR headless session per chain; cap math-library threads to the CPUs actually visible inside that session | HPC | 3-4x wall time; potentially up to 8x versus a heavily contended target session | 30-90 min; launcher is already landed | Low |
| 2 | Start with 200 warm-up plus 500 retained draws per chain, evaluate R-hat/ESS, and extend only compatible chains that miss the primary-parameter gate | NUTS | 2.5-2.9x fewer iterations | 15-30 min | Low-medium; convergence must decide extension |
| 3 | Seed a regularized dense NUTS metric from the MAP curvature in the unconstrained sampling chart; keep adaptation enabled and use heuristic initial step-size search | NUTS | 1.3-3x fewer leapfrog evaluations | 2-4 h | Medium |
| 4 | Use separate warm-up and sampling depth caps, initially `(7, 8)`, and record warm-up `num_steps`; raise only when measured saturation affects primary mixing | NUTS | 1x on ordinary trajectories; 2-8x on saturated warm-up steps | Under 1 h | Medium |
| 5 | After preconditioning, pilot `target_accept=0.85`; retain it only with zero divergences and acceptable energy diagnostics | NUTS | 1.1-1.4x | Under 1 h | Medium |
| 6 | Profile the warmed objective and gradient, then benchmark NUFFT tolerance and render-once visibility chunking on the exact target shapes | HPC | 1.2-2x if the NUFFT is confirmed dominant | 1-3 days | Medium |
| 7 | Prototype a fully device-resident GPU render/NUFFT/likelihood only for substantially larger batched workloads | GPU | Current evidence is a 5.5-6.5x regression; future gain unmeasured | 2-5 days for a bounded benchmark | Medium-high |
| 8 | Use a surrogate or normalizing flow only to initialize or propose, with exact-likelihood correction | AI | Unknown; useful only after repeated-run amortization | 1-2 weeks | High |

Topology plus the shorter initial budget plausibly reduces wall time by about
8-12x. A well-conditioned metric could raise the combined practical gain to
roughly 8-20x. Measure complete effective samples per wall hour before claiming
more.

## Evidence from the live and historical runs

- The metadata-complete visibility exports are faster to ingest. KGAS066/KGAS007
  v2 load times are 3.74/4.01 s, versus 99.87/81.93 s for the old files. The
  effective fit arrays are equal or smaller. NPZ I/O is not the current
  bottleneck.
- The historical independent-session KGAS007 campaign completed four chains in
  3,478-4,209 s per chain using 200 warm-up and 600 draws. Merge time was 0.016 s.
  See `docs/reviews/artifacts/2026-09-05-kgas007-nuts/wall.json`.
- The collaborator campaign used 1,000 warm-up and 1,000 draws, 2.5 times the
  historical iteration count, with a 10/11-dimensional likelihood instead of
  the old six-dimensional posterior.
- The fixed `0.01` unconstrained perturbation raised KGAS066 energy by
  171.9-616.6 and gradient norms to 31,175-62,321. All four chains completed
  zero checkpoints in 20 hours. Commit `8f11648` bounds the initial energy
  increase to 10; its replacement starts use `0.00125` jitter and energy deltas
  of 2.76-9.78.
- The retained KGAS007 chain is slow but is not saturating depth 10. Its first
  500 draws have zero divergences, median/mean/maximum leapfrog counts of
  15/21.8/47, and no 1,023-step trajectory. Do not attribute all elapsed time
  to tree-depth saturation.
- KGAS066 replacement warm-up moved rapidly through the first 36 steps and then
  spent minutes in individual steps around 37-38. This is consistent with deep
  early adaptation trajectories, but the current worker does not persist
  warm-up `num_steps`; instrument that value before choosing a permanent cap.
- The selected MAP curvature is nontrivial: reported active-space condition
  numbers are 462 for KGAS066 and 609 for KGAS007. KGAS007 retained draws show
  absolute correlations of roughly 0.37-0.65 and variances spanning about 700x.
  Identity/diagonal initialization therefore wastes warm-up.
- The current likelihood is already JAX/XLA compiled. `build_s3_objective`
  returns a jitted value-and-gradient, the chain transition is jitted, and the
  production path uses batched `jax-finufft` without a host bounce. Adding
  Numba, Cython, or another JAX wrapper does not address the measured problem.
- Checkpoint files and draw archives are tiny relative to trajectory compute.
  The scratch-first design and 100-step durable copy interval should remain.
- Existing evidence rejects GPU as an immediate fix at this size. CPU JAX
  achieved 3.01 likelihood evaluations/s and a 0.43 s six-axis gradient. The
  H100 MIG trial achieved 0.55 evaluations/s and a 2.80 s gradient. See
  `docs/PRODUCTION_RECORD.md` under validation and benchmark evidence.

## Implementation notes

The one-session-per-chain launcher and energy-bounded starts are already in
`scripts/launch_collaborator_canfar.py`,
`scripts/run_collaborator_nuts_chain_headless.sh`, and
`scripts/run_collaborator_nuts_chain.py`. Future runs should retire the shared
four-worker target topology in `scripts/run_collaborator_nuts_headless.sh` and
`scripts/run_collaborator_nuts_controller.py`.

The shorter budget is an initial allocation, not a relaxed posterior gate. Run
four chains, compute diagnostics on 500 retained draws each, and append draws
without repeating warm-up when primary R-hat or ESS alone fails. Do not thin
retained samples; thinning saves no compute and discards effective samples.

For preconditioning, compute or persist the full curvature matrix; the current
MAP JSON stores singular values but not eigenvectors. Form curvature in the
unconstrained `CampaignTransform` coordinates, including its Jacobian. A direct
`jax.hessian` currently fails through the installed `jax-finufft` batching rule,
so use finite differences of the already-jitted gradient unless that dependency
is upgraded. Symmetrize, floor small/nonpositive eigenvalues, pass the resulting
metric to NumPyro with `dense_mass=True`, retain mass adaptation, and fall back
to a diagonal metric on any non-SPD or pilot failure.

Judge a 50-warm-up/50-draw preconditioning pilot by wall seconds, mean/maximum
`num_steps`, divergences, finite energy, and agreement of the sampled potential.
It is a throughput test, not a scientific posterior product.

For kernel work, profile the native-cube render, Fourier shift, NUFFT, spectral
response, and covariance reduction separately after compilation. If visibility
rows are chunked, render the sky cube once and partition only visibility
sampling/reduction. Re-rendering the cube per chunk will erase the expected
gain. Test a looser NUFFT tolerance only against the existing visibility closure
and likelihood-identity gates.

GPU execution requires a CUDA-enabled JAX and `jax-finufft` build and a workload
large enough to amortize launch and transfer overhead. Removing
`JAX_PLATFORMS=cpu` is not a GPU implementation. Any later GPU pilot must keep
the full render, NUFFT, spectral response, and residual reduction on device and
compare warmed gradients plus effective samples per wall hour.

## Do This Next

- Keep the progressing KGAS007 session and the energy-bounded KGAS066 replacement
  sessions running. Use one flexible session per chain for every new or restarted
  chain; do not return to four chains inside one target container.
- Change the next campaign defaults to 200 warm-up plus 500 retained draws and
  implement compatible draw extension driven by the existing primary-parameter
  R-hat/ESS gates.
- Run one bounded 50/50 pilot with a regularized dense MAP metric, heuristic
  step-size search, and warm-up `num_steps` logging. Promote those sampler
  settings only if they reduce leapfrog work without divergences or potential
  mismatch.

Do not start a neural surrogate, a duplicate Numba/Cython forward model, or a
production GPU migration to solve the current run. Those are lower-return work
than topology, budget, and preconditioning.
