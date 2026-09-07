# Astra science-completeness review: production figures

- Review date: 2026-09-07 UTC
- Reviewed code: `c6822b10e274513d4b59f728ecfe5af99b75a885`
- Targets: KGAS066 and KGAS007
- Verdict: **accept**

## Evidence reviewed

Astra inspected all eight regenerated PNGs, both product manifests, both
science-deliverable notes, and the plotting path that binds retained model
products to the figures. Each target contains:

- masked moment 0/1/2 data, kinUV, and residual panels with the restoring beam;
- major- and minor-axis data, model, and residual PVDs;
- a conditional-MAP rotation curve with analytic and KinMS context, a one-beam
  scale, a qualified turnover marker, and restored-cube centroid diagnostics;
- the three-realization matched-family kinUV/KinMS synthetic recovery plot,
  including sub-beam turnover error and inner-beam projected-velocity RMSE.

No additional production plot is required because the accepted bundles
already retain aperture spectra, channel maps, and visibility residuals.
Approaching/receding profiles are useful future asymmetry diagnostics but are
not a completeness condition for this release.

## Scientific corrections confirmed

The production generator no longer reads the pre-repair
`benchmark/profiles.npz` centroid arrays. It recomputes centroids from the
retained moment maps with the current celestial-WCS tangent-plane coordinate
contract and records the result in `rotation_centroids.npz`.

Both target notes state that the plotted real-galaxy curves are conditional
MAP diagnostics, posterior intervals are uncalibrated, and neither a real
inner slope nor sub-beam turnover is promoted. A formal rotation claim still
requires a fitted non-rotating emitting disk and complete-refit bootstrap.
KGAS066 identifies its displayed turnover as an arctan-equivalent diagnostic
compression of the selected Stage B ring curve. KGAS007 records the Stage B
rejection and displays the accepted Stage A curve.

The review found no missing science-worthy diagnostic and accepts both suites
for the production record.
