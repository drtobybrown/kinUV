# Production layout and diagnostic QA

- Implementation commit: `e3a54fcdd2d61e9bf283374cd4b35043c9921fb7`
- Targets: KGAS066 and KGAS007
- Result: **pass**

The generator completed from both the legacy source layout and the new
standardized source layout. Each final target contains exactly `best_model/`,
`plots/`, and `benchmarks/`, plus its README and target manifest. Each manifest
verifies 43 files. Active figures comprise five direct-diagnostic and five
matched-benchmark PDF/PNG pairs; no PNG or PDF exists outside the two figure
directories.

Visual QA covered all 20 PNGs. Moment panels use shared data/model scaling,
symmetric residual scaling, white masks, a registered beam, and an adaptive
crop containing the detected 5-percent moment-0 support plus one BMAJ. The
registered half-width is 12.0 arcsec for KGAS066 and 7.5111 arcsec for
KGAS007. Spectra report aperture-integrated Jy values per matched channel.
Posterior contours use the five sampled primary dimensions and state the fixed
inclination explicitly; intervals remain labeled uncalibrated.

The deterministic suite passed with 261 tests and 5 skips. The complete prior
bundles were gzip-tested, tar-index-tested, and SHA-256 verified before their
active directories were removed.
