---
role: literature
field: computer-science
date: 2026-09-03
canon_generation: 4
ids:
  - DEC-066-INFER
  - DEC-066-PA
  - DEC-066-ZEROMODEL
---

# Rank

Not an ADR. If this file disagrees with a `DEC-*`, the DEC wins. Rank below `docs/architecture/STATUS.md` and the board log. INDEX is unchanged. Do not paste this essay into the Field Guide. Official MAP `kinuv-KGAS066-uvsign-map` is read-only. Do not start G4. Do not call Laplace-MH "NUTS".

# Question

When NUTS/HMC is initialized far from a basin, or the posterior is multimodal / funnel-like, what does the computational-statistics literature recommend? Is nested sampling (`dynesty`) the right next tool for KGAS066, or MAP + reject exploded chains + do not average modes?

# What NUTS does, and what long trajectories mean

Hoffman and Gelman (2014, JMLR 15:1593–1623, arXiv:1111.4246) introduced NUTS as HMC that builds a balanced binary tree of leapfrog steps and stops when the trajectory U-turns **or** when the simulated energy error becomes extremely large. Dual averaging tunes the step size \(\epsilon\) during warmup. Neither rule is a license to start far from the typical set. A large mean number of leapfrog steps is NUTS spending \(2^j\) gradient evaluations because the trajectory has not U-turned; Stan treats hitting `max_treedepth` as an **efficiency** warning, not a validity fix (Stan Development Team, *How to Diagnose and Resolve Convergence Problems*). Raising tree depth or `adapt_delta` is a last resort; it does not repair a bad init or a second basin.

Betancourt (2017/2018, arXiv:1701.02434) is the geometry paper. Probability mass lives on the **typical set**, not at the MAP and not in the far tails. MCMC has three phases: drift toward the typical set (biased), first sojourn (bias collapses), then refinement. Warmup is supposed to discard the first phase **and** adapt \(\epsilon\) there. If warmup never reaches the typical set, the adapted step size is the curvature of the wrong neighborhood. When a symplectic integrator then hits unresolved curvature, the numerical trajectory **diverges and flies toward infinite energy** (Betancourt Fig. 29; case study *Diagnosing Biased Inference with Divergences*). That is a validity failure: geometric ergodicity is lost and expectations are biased. Funnel geometry (Neal 2003 slice-sampling funnel; Betancourt and Girolami 2015, arXiv:1312.0906) is the textbook case of position-dependent curvature that a single global \(\epsilon\) cannot cover. The recommended CS move is reparameterization or a better start, not a longer tree.

# Parallel chains diagnose modes; they do not mix them

Gelman and Rubin (1992) and Vehtari, Gelman, Simpson, Carpenter, and Bürkner (2021, *Bayesian Analysis*, arXiv:1903.08008) run **at least four** chains from dispersed starts so that between-chain variance can expose non-mixing. \(\hat{R}\) near 1 and bulk/tail ESS \(\gtrsim 400\) (four chains) are the pass. High \(\hat{R}\) with well-separated chain means is the diagnostic **working**: the ensemble is not one stationary distribution. For separated modes, between-chain ESS collapses toward the number of modes found; pooling those draws and quoting a posterior mean is not sampling \(\pi(\theta\mid y)\). Occupancy of four independent chains is an artifact of init and seed, not of posterior mass.

Stan’s own warning page: high \(\hat{R}\) without divergences is commonly multiple well-separated modes. The literature tool for *crossing* those modes is **tempering**, not averaging: replica exchange / parallel tempering (Swendsen and Wang 1986; Geyer 1991) and Neal’s tempered transitions (1996, *Statistics and Computing* 6:353–366). Hot replicas flatten \(\pi^\beta\) so local kernels can jump; swaps propagate location back to \(\beta=1\). Independent NUTS chains with no swaps are not a tempering schedule. Yao, Vehtari, Simpson, and Gelman (2018) stacking averages **models** for prediction; it is not a recipe for stacking unmixed HMC chains of one misspecified likelihood.

# Nested sampling is for \(Z\), at a live-point price

Skilling (2006, *Bayesian Analysis* 1:833–860) made the evidence \(Z=\int L\,\pi\,\mathrm{d}\theta\) the **prime** result; posterior samples are an optional byproduct of likelihood-constrained prior sampling. Speagle (2020, *MNRAS* 493:3132, arXiv:1904.02180, `dynesty`) keeps that contract and adds dynamic live-point allocation. Default static \(K=500\). Iterations scale as \(N\sim K H\) with \(H\) the KL information in nats; each replacement is one or more likelihood-restricted prior draws (slice / random-walk / ellipsoidal), so wall clock is \(K\times H\times\) (LRPS evals), not one NUTS chain. Chopin and Robert (2010, *Biometrika*) give exact-NS cost \(O(d^3/e^2)\) when iteration cost is \(O(d)\). Buchner (2023, arXiv:2101.09675) reviews the same scaling: \(O(N)\) in live points, \(O(d^2)\)–\(O(d^3)\) in dimension for MCMC/slice LRPS; ellipsoidal NS is typically for \(d\lesssim 30\). Stage A 066 is \(d=6\) sampled (\(d=8\) if \((dx,dy)\) unfrozen), so dimension is not the veto. The veto is purpose and wall: Speagle’s benefits are (i) \(Z\), (ii) unsupervised multi-modal navigation, (iii) no MCMC burn-in. The drawbacks he states are prior-transform dependence, runtime set by prior volume, and that standard NS **cannot** prioritize posterior over \(Z\).

`DEC-066-INFER` already answers the tool question: MAP first; NUTS only after \(\Delta\chi^2\) vs \(V=0\) is real; **do not switch to dynesty because MAP failed**.

# Map onto the 066 approaching failure

| Fact | CS reading |
|---|---|
| Receding NUTS (`sd3ckpf2`) \(\hat{R}\sim 1\), ESS \(\ge 889\), mean leapfrog steps \(\sim 10\) | Typical-set HMC after a real MAP start. Hoffman dual averaging saw the right curvature. |
| Approaching init = receding MAP with only `pa_deg=25.2` overwritten; \(\sim 31\mathrm{k}\) \(\chi^2\) from the vis minimum | Warmup never started on the approaching typical set. Betancourt phase-1 init, not a sampler bug. |
| Mean steps 23–195; c4 flux \(10^{262}\) | Long trees (efficiency) plus a symplectic divergence to infinity (validity). Drop the exploded chain; do not retune `max_treedepth` from this init. |
| c1/c3 PA \(\sim 15^\circ\), c2 \(\sim 64^\circ\); merged \(\hat{R}(\mathrm{PA})=22\) | Parallel chains did their job. Do not average. Do not call the merge NUTS. |
| Official two-start L-BFGS: PA=205.2 \(\Delta\chi^2=35553\) vs PA=25.2 \(\Delta\chi^2=4260\) (\(\sim 8\times\)) | Modes are not equal mass. Chain occupancy is not \(Z\). MAP already ranked them (`DEC-066-PA`, `DEC-066-ZEROMODEL`). |
| \(\sim 3\) vis eval/s on \(881\times 95\) | One `dynesty` default-\(K\) compression is \(10^4\)–\(10^5\) likelihoods before LRPS inefficiency. Nested sampling is not cheaper than a MAP. |
| Leftover structured (vel span \(>\) uv span); S2 Laplace SBC failed 68/95 | \(Z\) for this likelihood is not a model-selection number. Ignore leftover and you are answering a different problem. |

# CS recommendation for **this** 066 failure

1. **MAP first** (already `DEC-066-INFER`). Approaching NUTS from a receding \(\theta\) with PA overwritten was not that order. Do not relaunch NUTS from the same init.
2. **Drop exploded chains.** A flux \(10^{262}\) draw is a divergent leapfrog, not a mode. Merge hygiene rejects non-finite shards; it does not manufacture \(\hat{R}\approx 1\) from c1–c3.
3. **Do not average modes.** Four independent chains are a diagnostic. Tempering would be the literature mixer if 066 needed one posterior over both PAs; 066 does not: the official two-start already discarded the approaching start on \(\Delta\chi^2\).
4. **Nested sampling only if you need \(Z\)** and you are willing to **ignore leftover**. `dynesty` is the right next tool for evidence and unsupervised mode-finding on a well-specified, cheap likelihood. It is the wrong next tool for a 3 eval/s vis \(\chi^2\) whose approaching basin is already \(8\times\) weaker and whose leftover is SB-dominated. Live-point wall is not a substitute for L-BFGS on PA=25.2.

Receding NUTS remains the 066 sampler product. Approaching is a failed init plus a weaker MAP, not a reason to change the inference engine.
