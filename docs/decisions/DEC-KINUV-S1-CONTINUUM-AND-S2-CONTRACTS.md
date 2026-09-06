---
id: DEC-KINUV-S1-CONTINUUM-AND-S2-CONTRACTS
status: accepted
date: 2026-09-06
authority: Astra
supersedes_transform: ESC-KINUV-S1-RENDERER-CONVERGENCE
implementation: licensed-by-pragmatic-s1-directive
---
# Continuum comparator and S2 contracts

## Decision

S1 will use a labeled **KinMS continuum adapter** for matched-domain numerical
closure. It implements KinMS's thin circular-disk geometry with continuous
projected cloud coordinates and mean line-of-sight velocities, but replaces
KinMS's nearest-cell cube population and stochastic/finite Gauss-Hermite
dispersion samples. It is not described as the stock KinMS renderer. The stock
KinMS image-fitting benchmark remains a separate best-practice comparator.

This adapter is external validation infrastructure. It does not enter kinUV's
likelihood, gradients, sampler, or runtime dependency graph, and it does not
constrain kinUV to use clouds or an image grid internally.

## Continuum renderer

Pin KinMS 3.0.13. Generate deterministic face-on radial/azimuthal quadrature
clouds and reproduce its thin-disk projection and circular line-of-sight
velocity equations directly, without invoking its nearest-cell cube builder.
Apply the
verified boundary conversion `kinms_posang=(360-kinuv_pa)%360`. Apply the
declared sky-centre and systemic-velocity offsets explicitly to the returned
continuous coordinates; do not infer that returned clouds contain cube-bin
offsets.

Deposit nonnegative cloud flux on the spatial lattice with the separable
cardinal cubic B-spline

`K_h(x,y) = h^-2 B3(x/h) B3(y/h)`, where

- `B3(t)=(4-6|t|^2+3|t|^3)/6` for `|t|<1`;
- `B3(t)=(2-|t|)^3/6` for `1<=|t|<2`;
- `B3(t)=0` otherwise.

Use exact partition-of-unity weights and pad for the entire two-pixel kernel
support. Record input, retained, spatially excluded, spectrally excluded, and
jointly retained flux before any normalization. Renormalization after clipping
is prohibited.

For a cloud with mean velocity `mu`, positive dispersion `sigma`, integrated
flux `F`, and channel edges `v_lo,v_hi`, its channel-integrated emission is

`E = F [Phi((v_hi-mu)/sigma) - Phi((v_lo-mu)/sigma)]` in Jy km/s.

Use stable normal-CDF tail evaluation. The value supplied to the visibility
operator is the channel-average flux density `S=E/abs(v_hi-v_lo)` in Jy.
Native top-hat integration, Hanning response, and weighted software binning are
distinct operations and each is applied exactly once. Radio/optical conversion
passes through frequency. Remove dispersion-cloud replication and mutation of
KinMS random draws.

The matched kinUV branch uses the same analytic native-channel integral. The
corrected operator has a new schema/version and requires new baseline fits;
old chi-square values are not recovered by changing data, weights, or target
parameters. Historical results retain their original recorded operator.

## S1 convergence and evidence

S1 uses proportional verification on the production path. A second ungridded
direct-cloud Fourier engine is out of scope while target refinement closes.
It may be introduced only if empirical refinement still fails without an
isolated cause. The following gates apply:

| Gate | Limit |
|---|---:|
| Successive-refinement relative complex-visibility L2 | `<=1e-4` |
| Actual-data absolute chi-square refinement sensitivity | `<=0.1` |
| Pre-normalization physical-flux relative error | `<=0.001` |
| Controlled spectral-centroid error | `<=0.02` native channel |

Test spatial spacing, radial quadrature, azimuthal quadrature, and analytic
spectral subdivision independently on both canonical targets. Repeat with an
independent quadrature phase. Gaussian dispersion order is retired as a
production convergence axis. The velocity profile contains the explicit point
`R=0, v_c=0`. A passing target matrix closes S1 immediately through an atomic
implementation commit and STATUS update.

## S2 geometry and data contract

KGAS007 is KILOGAS catalogue target 7, MaNGA `1-55227`,
`J094618.26+025303.62`. The historical `28.9 deg` is an initialization value,
not an uncertainty-bearing prior. Until independently measured optical disk
geometry with source, PSF treatment, uncertainty, intrinsic-shape assumptions,
and gas/stellar alignment assessment is registered, use the explicitly
authorized isotropic-orientation prior `p(cos(i))=1` on `0<cos(i)<1`. Profile
inclination and promote only identifiable `u(R)=v_c(R) sin(i)`, not intrinsic
`v_c(R)`.

The canonical visibility export must preserve schema version, MS row identity,
observation/execution block, array, scan, field, SPW/data-description,
polarization, antenna pair, time/interval, UVW metres, channel centres and
edges with frame, flags, calibrated Jy units, weight convention, calibration
and smoothing history. Every averaging product records contributing source
rows and coefficients. Assign folds before aggregation and never mix training
and validation contributors in one aggregate. Real held-out claims remain
blocked when this provenance is absent.

For this campaign, `BMAJ_ref` is the major-axis FWHM of the hash-frozen official
v1.3 original 10-km/s diagnostic cube: `1.254196 arcsec` for KGAS007 and
`1.040022 arcsec` for KGAS066. Record BMIN and BPA. BMAJ defines audit scales,
not a visibility convolution. Diagnostic restoration uses the full elliptical
beam covariance and an enclosing common beam when beams vary.

Keep the geometrically thin disk. Use `0<=u(R)<=500 km/s` as the canonical
computational search envelope; boundary pressure blocks promotion. Analytic
LOSVD accounting covers at least `mu +/- 8 sigma` and is checked against
`+/-10 sigma`, with all response guards retained and no window
renormalization. Existing registered centre, systemic-velocity, and positive
constant-dispersion bounds remain until changed through review.

## S2 covariance and optimization

The candidate covariance family is:

1. `C0`: positive diagonal pre-response thermal covariance propagated through
   recorded averaging, Hanning, and bin operators, with scale fitted per
   execution-block/SPW/polarization stratum.
2. `C1`: `C0` plus pre-response spectral AR(1), `|rho|<=0.5`, estimated from
   training line-free data only.

Select with grouped inner-validation predictive log likelihood including
`logdet(C)` and both complex components. Use the one-standard-error rule and
prefer `C0`. Freeze selection before signal fitting and outer-fold scoring.
Failure of both candidates or material residual time/antenna/cross-component
correlation returns to Astra.

Use dimensionless coordinates: log flux, log dispersion, `u/100 km/s`,
turnover and offsets divided by `BMAJ_ref`, systemic offset divided by fitted
channel width, PA in radians, and `cos(i)`. For

`f(z)=0.5 r^T C^-1 r - log p(theta(z)) + registered regularization`, define

`g_P = z - projection_Z(z-gradient_z f)`.

Require `norm(g_P, infinity)<=1e-3` without dividing objective or gradient by
data count. Treat PA periodically and report likelihood, prior, and
regularization separately.

Use at least five held-out groups formed from disjoint time/scan blocks across
all baselines, with boundaries wider than measured time and spectral-response
correlations. Group by execution block when calibration correlations demand
it. Random UV-cell folds are prohibited.

At each registered turnover ratio run twelve deterministic starts: 72
fixed-turnover fits, followed by twelve joint fits with turnover released.
Every outer fold repeats the complete procedure. At least three joint solutions
must be within `0.1` chi-square of the best and pass the projected-gradient
gate. Endpoint selection triggers the previously specified extended audit.

## Promotion boundary

The S4 superiority and non-regression gates in
[`DEC-KINUV-CROSSDOMAIN-RECOVERY`](DEC-KINUV-CROSSDOMAIN-RECOVERY.md) remain
unchanged. No NUTS campaign or calibrated-interval claim is licensed.

## Implementation outcome

The localized implementation at `97548f7` and its second refinement correction
at `c55c985` removed the discrete-dispersion failure. The r4 canonical matrix
passes spatial, azimuthal, spectral, phase, flux, coordinate, and centroid
checks. Radial refinement remains above both frozen thresholds for KGAS066 and
KGAS007. S1 therefore remains open under the two-iteration escalation rule.
