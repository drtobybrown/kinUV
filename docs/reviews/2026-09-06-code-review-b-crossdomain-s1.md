---
role: reviewer
seat: software-reproducibility
phase: implementation
date: 2026-09-06
reviewer: s1-review-b
canon_generation: 11
campaign_id: crossdomain-recovery-s1
proposal: docs/decisions/DEC-KINUV-CROSSDOMAIN-RECOVERY.md
reviewed_commit: 1b629a1fab9c64a25cb870d8f5ad43102b55cb0b
verdict: accept-with-required-changes
---
# Independent code review B: cross-domain recovery S1

I reviewed exact commit `1b629a1fab9c64a25cb870d8f5ad43102b55cb0b`
without reading or contacting Reviewer A. This verdict covers the S1
intrinsic-renderer boundary, common measurement operator, and durable closure
dossier. It does not assess S2 or license a fit, posterior, or superiority
claim.

## Attempted falsification

I checked package imports with KinMS and every legacy/CASA package blocked,
checked the isolated KinMS environment for the inverse dependency, traced the
kinUV and comparator operator call paths, inspected KinMS 3.0.13's installed
`cleanOut` behavior, exercised the versioned NPZ contract with contradictory
metadata, verified the durable manifest recursively, and repeated an external
nominal render twice.

- Full CPU suite: `230 passed, 8 skipped` in 123.48 seconds.
- Focused S1, standalone, architecture, and exact-replay suite: `40 passed, 3
  skipped`.
- Official KGAS066 likelihood replay after the operator refactor: `3 passed`.
- kinUV import with `kinms`, `uvkin`, `uvfit`, `ms2kinuv`, CASA, casacore, and
  pyuvdata blocked: passed.
- Isolated KinMS environment: KinMS 3.0.13 imports; `kinuv` is absent.
- All 28 S1 dossier entries match manifest sizes and SHA-256 hashes. The current
  worker, environment lock, source snapshots, target configs, S0 input, KinMS
  module, worker configs, logs, sidecars, and NPZ products match their recorded
  hashes.
- Two reruns of the KGAS007 nominal external render produced byte-identical NPZ
  and sidecar files and numerically identical cubes.

The strongest contract falsification succeeded. I built a checksum-valid NPZ
whose embedded metadata declared `clean_out=false`, all three forbidden
operators applied, wrong units and axis order, and rendered flux 999. Its
sidecar claimed the approved values and rendered flux 1. The loader accepted
the product, returned the 320-flux cube, and exposed the false sidecar claim to
the flux gate. Only `config_sha256` is compared between the hash-bound embedded
metadata and sidecar.

## Findings

1. **Required — The external product contract does not fail closed.**
   [`load_intrinsic_kinms_cube`](../../src/kinuv/diagnostics/comparator.py)
   validates the mutable sidecar's operator flags, units, axis order, and
   `clean_out`, but checks only `config_sha256` in the metadata embedded inside
   the checksum-bound NPZ. A contradictory embedded record is accepted. The
   target flux gate then trusts `integrated_flux_jy_kms_rendered` from that
   sidecar instead of summing the loaded cube. Validate the embedded schema and
   every contract-critical field, require agreement with the sidecar, recompute
   integrated flux from the cube, and use the recomputed quantity for the gate.
   Add rejection tests for each forbidden operator, missing/mismatched embedded
   metadata, nonfinite axes, and claimed-versus-actual flux. The current
   dossier's six products are internally consistent when checked independently;
   this is a consumer-path defect rather than evidence that those files are
   corrupt.

2. **Required — The dossier runner can overwrite or mix a validated run.**
   [`run_s1_crossdomain_closure.py`](../../scripts/run_s1_crossdomain_closure.py)
   creates the fixed output root with `exist_ok=True`, overwrites configs,
   logs, cubes, sidecars, and metrics, and only rewrites the manifest at the
   end. It neither rejects a nonempty root nor stages a new attempt. A failed
   rerun can therefore leave an old passing manifest beside a mixture of new
   partial products. It also records a dirty tree but does not reject it. Require
   a clean exact commit, a new empty attempt directory, and staged/atomic
   publication of the completed dossier. Fail before touching a nonempty
   validated directory. Generate the source snapshot and manifest as part of
   that recorded workflow rather than through an undocumented post-run step.

3. **Required — Environment and upstream-input locks are recorded but not
   enforced.** `_environment` saves `pip freeze`, but the runner never compares
   it with `external/requirements-kinms-s1.txt`; a different KinMS or NumPy
   environment can pass. It hashes the S0 metrics and current target configs but
   does not validate the S0 manifest/schema/accepted commit or verify that the
   current scientific inputs match the S0 input hashes before reusing the S0
   parameters. Enforce the checked-in lock, the S0 manifest and schema, target
   config identity, and input checksums before rendering. Record and validate
   the worker's KinMS version/module hash against the frozen environment. The
   reviewed dossier happens to satisfy these equalities, which I verified
   manually; the executable does not guarantee them.

4. **Required — The frozen latency and memory provenance gate is incomplete.**
   The accepted decision requires deterministic full-pipeline latency and peak
   memory on declared hardware/backend, with optimizer budgets stated. The
   metrics contain per-worker wall time, operator time, replay time, and a
   cumulative `RUSAGE_CHILDREN.ru_maxrss`. They contain no top-level command,
   total pipeline wall time, parent/full-process peak RSS, CPU/device identity,
   thread settings, or NUFFT backend, and do not explicitly mark optimizer
   budgets as not applicable. `child_max_rss_kib` is the maximum over all prior
   children, not an individual worker delta. Record these quantities with clear
   scope and units; do not infer total latency by summing partial timers.

5. **Advisory — Keep the full operator orchestration in one callable.** Both
   paths now call the same PB/NUFFT implementation and the same `hann_then_bin`
   function, and baseline replay is exact. The binned orchestration remains
   duplicated between `infer.map.predict_binned` and
   `forward.operators.sample_intrinsic_cube_binned`. Routing both intrinsic
   cubes through one shared binned callable, or adding a call-path equivalence
   test, would prevent future ordering or guard-channel drift.

6. **Advisory — Harden the external API shim.** The deterministic renderer
   replaces KinMS's pre-generated `randompick_vdisp` attribute. Pinning KinMS
   3.0.13 and recording the module hash limits current risk, but this behavior
   is not checked by an integration test and can change across KinMS releases.
   The environment file pins versions but not package-distribution hashes.

## Gate assessment

The numerical dossier reports passes within every frozen S1 threshold:
analytic relative visibility L2 `4.70e-14`, component RMS `1.04e-12`,
zero-baseline flux error `3.70e-15`, centroid error below `1e-9` channel, and
doubled-sampling delta chi-square `2.40e-5`. For KGAS066/KGAS007 respectively,
nominal-versus-high rendering RMS is `1.34e-4`/`3.35e-5` thermal SD,
independent-high RMS is `6.76e-5`/`1.63e-5`, high-cloud absolute chi-square
change is `0.04294`/`0.02802`, and baseline replay error is zero. These values
and their thresholds are machine-readable, and the current manifest is
internally intact.

The architecture correctly keeps KinMS in an isolated external process and
keeps KinMS, CASA, ms2kinuv, and legacy packages out of kinUV runtime imports.
The worker uses KinMS `cleanOut=True`, no spectral response, and no primary
beam; kinUV applies the shared PB, NUFFT, and Hann/bin operators once.

S1 cannot close at this commit because the consumer can accept a contradictory
external contract, the runner can overwrite a passed dossier and execute under
unlocked inputs/environments, and the frozen full-pipeline latency/memory report
is absent. STATUS is accurate about the measured numerical values and pending
review, but “passes its principal numerical gates” must not be interpreted as
completion of the omitted reproducibility gates.

## Residual risks

- The current dossier directory remains writable; its manifest provides
  integrity detection but no independent immutable anchor.
- Deterministic quadrature and the pinned KinMS module produced byte-identical
  reruns in this environment, but cross-platform byte identity is not claimed.
- The S1 artifact establishes operator and renderer closure. It does not yet
  establish matched/native comparative truth recovery or real-data
  superiority.
- The isolated worker has no routine end-to-end test in the main suite; the
  durable six-render dossier is the only integration evidence.

## Verdict rationale

**Accept with required changes.** The shared-operator refactor, external KinMS
isolation, numerical closure, deterministic rendering, legacy replay, and
current artifact hashes are strong. The required findings are bounded software
and provenance repairs within the frozen S1 specification. They do not alter
physics or relax a gate, but they prevent this exact commit from closing the
software/reproducibility seat. Reviewer B must inspect a revised exact commit
and regenerated immutable dossier before acceptance.
