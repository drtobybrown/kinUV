# Reviewer B: S4 recovery software and reproducibility review

**Verdict:** `changes-requested`  
**Review role:** Reviewer B, software and reproducibility  
**Candidate range:** `3462efd05481aa8ae4c51d8e21c5cc5badf505c2..17dc9d7b576b2fede281a935d3b69a23e6d25fa8`  
**Scientific evidence commit declared by the final dossier:** `f0d3067562e3b65b07a6a748ad869ea6656f9c0e`

The numerical S4 gates pass as recorded, the evidence files are intact, and the
CASA/legacy boundary is clean. Promotion is blocked by the synthetic runner's
checkpoint provenance contract. The final dossier stamps one clean exact
commit while containing a retained KinMS checkpoint made before that commit,
and it does not retain the exact source-input hashes required by the binding PI
decision.

## Evidence that passes

* All five manifests were independently rehashed. The final dossier has 50 of
  50 registered payloads and 69,145,565 registered bytes; the nested synthetic
  dossier has 49 of 49; the grouped dossier has 9 of 9; and the two supporting
  replay dossiers each have 83 of 83. Every byte count and SHA-256 matches, with
  no missing or unregistered payloads under each manifest's exclusion rule.
* The grouped source summary records a clean `dev` checkout at `403d248` and the
  synthetic summary records a clean `dev` checkout at `f0d3067`. The grouped
  lower 95 percent bounds are +0.0317859235 for KGAS066 and +0.00290303756 for
  KGAS007. All ten held-out folds favor kinUV.
* The three declared synthetic seeds, 7001--7003, are present for both targets.
  All twelve optimizers report success. The aggregate projected-velocity RMSE
  ratios are 0.01180267 for KGAS066 and 0.04569431 for KGAS007, both below the
  binding 0.90 maximum.
* The four named CASA scratch environments are absent. No CASA/tclean log is
  present in the repository, no CASA process was found, and neither S4 runner
  nor `src/` imports `casatasks`, `uvkin`, or `uvfit`. The synthetic runner uses
  Python generation and the KinMS worker only. The standalone boundary is
  preserved.
* Focused verification passes: `34 passed` for
  `tests/test_s4_benchmark.py`, `tests/test_hotpath_architecture.py`,
  `tests/test_standalone.py`, and `tests/test_s1_comparator.py` using the S0
  validation environment. The recorded broader deterministic result is 263
  passed and 5 skipped, with the unlicensed unstable NUTS smoke excluded.
* The decision, STATUS, and PLAN consistently prohibit training-fold CASA
  imaging, identify the standard-use and Python-native gates, keep S5 frozen
  pending review, and preserve earlier failed evidence as supporting history.

## Blocking reproducibility findings

1. **A resumed fit is accepted without proving that it belongs to the current
   realization.** In `run_s4_synthetic_recovery.py`, `_kinms_fit` returns any
   existing `kinms_fit_result.json` whose `success` field is true. It does not
   compare the current commit, seed, target, worker hash, fit-config hash, mock
   cube hash, truth-cube hash, or mask hash. The outer `--resume` path then
   regenerates the truth, noise cube, and kinUV fit, incorporates the retained
   KinMS parameters, and writes a new summary and manifest under the current
   commit. No test exercises rejection of a stale or mismatched checkpoint.

2. **The sealed dossier demonstrates the mixed-generation condition.** The
   KGAS066 seed-7001 KinMS result and model cube were written at 17:20 UTC,
   before commit `f0d3067` at 17:28:46 UTC. Its paired truth and mock cube were
   regenerated at 17:29--17:30 UTC, and `recovery.json` was rewritten at 17:30.
   The nested summary and manifest nevertheless attribute the complete dossier
   to `f0d3067`. The earlier checkpoint may be scientifically identical, but
   the retained record does not prove that identity. Exact-commit provenance
   therefore is not established.

3. **Required source inputs are not checksum-bound.** The PI decision requires
   seeds, parameter support, noise, masks, inputs, code commit, and checksums in
   the final dossier. Output files are checksum-bound, but the synthetic
   records omit hashes for the target configs, canonical visibility NPZ files,
   covariance metrics, diagnostic cube/mask/error inputs, KinMS worker, and
   executable/environment identity. The per-target `load` block contains only
   aggregation counts and settings. The grouped artifact similarly omits its
   fixed bootstrap seed from the result: seed 4404 exists only as a source-code
   default while draws and resampling unit are serialized.

## Required correction

Create a new immutable evidence directory. Before reusing a checkpoint, bind it
to a checkpoint record containing at least target ID, seed, code commit, worker
hash, fit-config hash, and hashes of the exact cube, truth, and mask inputs;
reject or recompute on any mismatch. Record and checksum all external/source
inputs required to regenerate both synthetic target runs, and serialize the
grouped bootstrap seed. Add focused tests for matching-resume reuse and stale
checkpoint rejection. Re-seal the final manifest from one reproducible run or
from explicitly provenance-bound checkpoints, then resubmit Reviewer B.

The arithmetic gates themselves do not need to be changed or relaxed.

## Revision review: authenticated candidate

**Verdict:** `changes-requested`<br>
**Implementation:** `f2f6a22b6749f46da70e55f57e599f919190c9f8`<br>
**Status packet:** `879a04b896df6a22b72437e98bb937385b0a9e68`

The first submission's mixed-generation defect is repaired. Every regenerated
synthetic product timestamp follows `f2f6a22`; all six fit configurations bind
the exact commit, realization seed, mock-cube hash, truth-cube hash, mask hash,
and KinMS-worker hash; and all bound values independently verify. Retained
results are reused only when their complete worker configuration is equal to
the newly constructed configuration. The synthetic summary and manifest both
record a clean `dev` checkout at `f2f6a22`.

The complete manifest chain verifies again: 50 of 50 final payloads totaling
69,156,464 bytes, 49 of 49 synthetic payloads, 9 of 9 grouped payloads, and 83
of 83 payloads in each supporting replay. All registered sizes and SHA-256
values match. The final metrics serialize grouped bootstrap seed 4404. The
exact `f2f6a22` checkout passes the declared deterministic suite with 263 tests
passed and 5 skipped. The CASA-free and standalone boundaries remain intact.

Two reproducibility requirements remain open in this revision:

1. Each target passes the distinct 30 km/s `fit_window_cube` to
   `load_target_vis`, so that file affects the selected visibility channels.
   It is not one of the recorded input hashes. Hashing the target configuration
   records its path, not the content of that external FITS input.
2. The added test checks checkpoint code with source-text assertions. It does
   not execute a matching-checkpoint reuse or demonstrate rejection after a
   commit, seed, or cube-hash mismatch, as required by the first review.

Add the fit-window FITS hash and behavioral checkpoint identity tests, then
regenerate and reseal the dossier at the resulting clean implementation
commit. No scientific metric or gate change is requested.
