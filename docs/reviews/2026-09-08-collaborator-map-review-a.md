# Reviewer A: collaborator MAP-only packet v2

**Verdict:** `accept`

**Reviewed packet:** `results/production/meeting_packets/kinuv-collaborator-20260908-map-v2/`

Reviewer A independently verified the physical and scientific record. The
selected MAP parameters equal the derived checkpoints and replay parameters;
the delivered cubes are byte-identical to the replay cubes. C1 covariance,
spectral response, primary-beam frequency, celestial coordinate convention,
and native-TOPO-radio to LSRK-optical conversion remain consistent.

The first packet revision was rejected because it mislabeled new joint MAP fits
as unchanged S4 checkpoints and understated the remaining real-cube residuals.
Revision v2 correctly records `map_fit_performed=true`, distinguishes rendering
from fitting, scopes the inherited synthetic evidence per target, preserves the
old replay blockers as historical fields, and discloses the KGAS066 spectrum,
PVD, moment-1, reduced-chi-square, and profile-RMSE deficits.

All five packet entries, both 32-entry target manifests, 49 S4 synthetic
entries, seven S5 entries, and the process ledger verified. The required MAP
moments, PVDs, spectra, rotation curves, and synthetic comparisons are complete
and readable. Posterior status is `RUNNING` and is outside this review.

| Artifact | SHA-256 |
|---|---|
| Packet `INDEX.json` | `64976fb021903f9576f921ab99712c8862b444891ef52456bef330a7bbad79ae` |
| Packet `MANIFEST.json` | `bd030b2346d696b8db58dec51c223e2b9e199add9617dce7a2eff541dc8d2a09` |
| KGAS066 packet manifest | `f815f87dbc4cc869b875547c5a839e7611e0ca22bc58d84fca578ddd83f9b3fb` |
| KGAS066 render manifest | `44f1ed8ed26448ad366ce0fd5ef01b52e4a7a9d800d23210bf162414a064566e` |
| KGAS007 packet manifest | `89b84da9c816551d5761316c71149fba684baf5ebb5040c4b0290079fbefa71a` |
| KGAS007 render manifest | `a7e48eaa7b0e61a96ceddb56f6a3dd23f94674aacd30ac18c9654a14b12f28e2` |
| Attempt ledger | `5fafd4615d05e188bac7f9bea8b396efd2cac6307c9e726be549411b4cda64bf` |

Selected MAPs are KGAS066 start 4 at chi-square 166317.442848 and
KGAS007 start 3 at chi-square 105421.994976.
