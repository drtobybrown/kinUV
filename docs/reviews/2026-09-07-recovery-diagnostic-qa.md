# Recovery diagnostic QA

## 2026-09-07 diagnostic coordinate and checkpoint repair

The PI licensed this bounded diagnostic repair after discovering that the PVD
helper reversed astronomical east. Implementation `fdcbf165cdd8b1f9b6d89d71cfd5353ce4ac9e7c` uses celestial
spherical offsets with east increasing RA and PA east of north. Off-cardinal
slit tests measure the expected physical derivative; 36 focused imaging,
style, and S4 tests pass. The former KGAS066 major/minor slits were displaced
by 39.457741 deg modulo 180 and cannot diagnose radial motions.

Active products now consume the exact selected checkpoints and matched cubes
from `results/validation/crossdomain-recovery-s4-remediation-20260907-r1/`,
which the sealed S4 dossier explicitly binds. KGAS066 uses
`two_zone_dispersion`; KGAS007 uses `supported_rings`. Checkpoint parameters,
cube hashes, observed-frequency PB accounting, one native Hann response,
channel edges, and native-TOPO to optical-LSRK conversion are retained.
The exact fit configuration is recovered from replay commit `efe5433` and
verified against its saved SHA; later configuration path changes are routing
metadata. The deprecated active posterior routes were removed at `e5639e5`;
the exact routing configs used for this plot pass are preserved in the archive.
No optimization, inference, new scoring, or new gate was performed.

`best_model/` contains that checkpoint, parameters, exact restored model cube,
configuration and provenance. Four direct PDF/PNG pairs cover moments,
major/minor PVDs, spectra and the selected rotation profile; five benchmark
pairs cover matched moments, PVDs, spectra, rotation and retained synthetic
recovery. There is no posterior for these selected checkpoints. The older
fixed-inclination arctan posterior and corner are historical archive products,
not covariance of the displayed model. Every active manifest verifies 30 files.

Astra inspected both targets' moment footprints, PVD geometry and spectra,
including matched KinMS comparisons, and accepted science completeness and
the bounded replacement. Moment framing encloses the detected emission:
12.0 arcsec half-width for KGAS066 and 7.450614 arcsec for KGAS007.
S0--S5 remain closed; residual image structure remains visible as a diagnostic.

| Target | Production manifest SHA-256 | Archived previous tree | Archive bytes |
|---|---|---|---:|
| KGAS066 | `f60c01822bb7a589f9e12a48e49bb15fea8856ad78ad9ab798a7305c8d60a279` | `results/archive/KGAS066/20260907_pre-checkpoint-diagnostics.tar.gz` | 12848967 |
| KGAS007 | `813b22cbfe4a769b096d392680f63f37257ded2b17bedc347b27bf0288eea1d6` | `results/archive/KGAS007/20260907_pre-checkpoint-diagnostics.tar.gz` | 10136900 |

Each archive was verified against all 44 original file hashes before removing
the superseded tree. The external archive manifest records full archive and
unchanged checkpoint/cube hashes.

The scientific claim is visibility-domain inference and the measured synthetic
advantage within the registered family. For KGAS066 the retained turnover-error
and inner projected-velocity RMSE ratios are 0.02493 and 0.01438. Restored
real cubes are supporting diagnostics, not known truth. Spatial beam convolution
does not itself smooth across channels. Visibility modeling retains phase-derived centroid constraints without
requiring CLEAN inversion; for short baselines of a marginally
resolved source, `phi(q,v) approximately -2 pi q dot xbar(v)`, with q in
wavelengths and xbar in radians. Differential phase uses the centroid difference
from a reference channel ([Lachaume 2003](https://arxiv.org/abs/astro-ph/0304259)).
Clumps influence flux-weighted centroids too; this is complementary information,
not exact morphology/kinematics independence. Native channels retain measured
adjacent correlation (C1 rho approximately 0.2977), modeled in the likelihood.

Implementation verification: Sol. Independent science completeness: Astra, by visual inspection of both target suites. Verdict: accept bounded diagnostic replacement. No new scientific fit or scoring acceptance is implied.
