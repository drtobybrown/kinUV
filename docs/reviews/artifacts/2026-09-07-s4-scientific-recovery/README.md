# S4 scientific recovery review packet

**Candidate implementation:** `f2f6a22b6749f46da70e55f57e599f919190c9f8`<br>
**Binding comparison:** `DEC-PI-S4-STANDARD-USE-BENCHMARK`<br>
**Dossier:** `results/validation/crossdomain-recovery-s4-final-20260907/`

## Promotion evidence

| Target | Held-out delta chi-square per component | Lower 95% bound | Python-truth kinUV RMSE | Python-truth KinMS RMSE | Ratio | Gate |
|---|---:|---:|---:|---:|---:|---|
| KGAS066 | +0.04265074 | +0.03178592 | 0.08836 km/s | 7.48616 km/s | 0.01180 | pass |
| KGAS007 | +0.00456083 | +0.00290304 | 0.39607 km/s | 8.66773 km/s | 0.04569 | pass |

All five real-data folds favor kinUV for both targets. The stock KinMS
comparator was frozen from the canonical full-data science cube and therefore
received information from the held-out groups; kinUV was refitted on each
training fold. The positive bounds are consequently conservative for the
standard-use comparison required by the PI.

The synthetic suite uses three fixed noise seeds per target. A projected,
axisymmetric exponential emissivity distribution and arctan rotation curve are
shared by the analytic cube and visibility generator. kinUV fits the complex
visibilities; stock KinMS fits the Python-generated restored cube. The
projected profile `u(r)` is scored over 24 common radii. Both aggregate ratios
are below the required 0.90.

## Supporting evidence

* The repaired frozen-ancestor replay is
  `results/validation/crossdomain-recovery-s4-repair-20260907-r2/`.
* The smooth-emissivity fit and replay are under
  `results/validation/crossdomain-recovery-s4-remediation-20260907-r1/`.
* The grouped visibility audit is
  `results/validation/crossdomain-recovery-s4-remediation-20260907-r2/grouped/`.
* The final dossier contains 50 checksum-bound files totaling 69,156,464
  bytes. All referenced source manifests were revalidated while sealing it.
* Reviewer B rejected the first submission at `72a7d35` because resume could
  mix an older KinMS checkpoint into a newer stamped dossier. The revised
  runner authenticates commit, realization seed, runner, worker, cube, truth,
  and mask before reuse. Target config, covariance, visibility, diagnostic
  inputs, and bootstrap seed 4404 are serialized. The entire dossier was then
  deleted and all six fits rerun from scratch at the candidate commit; every
  scientific metric reproduced exactly.
* The deterministic and architecture suite passes: 263 passed, 5 skipped.
  The known unstable NUTS smoke module is outside this authorized MAP/recovery
  run and was not executed.

## CASA boundary

No CASA task contributed to any promoted metric or product. The abandoned
experiment produced no science image. Its background processes were stopped,
four scratch environments were removed, and three untracked CASA logs were
deleted. S4 no longer requires training-fold `tclean` imaging. kinUV remains a
visibility engine; Measurement Set extraction remains isolated in ms2kinuv.

## Review questions

Reviewer A should decide whether the standard-use and matched-family evidence
supports the physical superiority claim without over-interpreting restored
cube diagnostics. Reviewer B should verify the code boundary, clean exact
commit, manifest chain, deterministic tests, and resumable external-fit
execution. Dual `accept` closes S4 and permits S5 sealing.
