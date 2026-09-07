"""Visibility loading, cube-window trim, and time/uv aggregation.

Source is the native ``43240×1920`` npz. Fit product is time-averaged 30 s,
uv-binned 10 m, then software-binned ``N=4``. Data are already
correlator-Hann'd — this module never Hanns visibilities.

The Ico cube is ``VOPT``; visibilities are radio vs rest CO via
:func:`kinuv.constants.freq_to_velocity_kms`. The trim converts the cube
window to radio so the same sky frequencies are selected.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from kinuv.constants import C_LIGHT_KM_S, C_LIGHT_M_S, freq_to_velocity_kms
from kinuv.decisions import requires
from kinuv.likelihood.chi2 import empirical_s
from kinuv.response.spectral import bin_channels

DEFAULT_NPZ = Path("/Users/thbrown/kilogas/DR1/visibilities/KILOGAS066.npz")
DEFAULT_CUBE = Path(
    "/Users/thbrown/kilogas/analysis/kinms_test/kgas066/KGAS66_clipped_cube.fits"
)

# KGAS66_clipped_cube.fits CTYPE3=VOPT-W2W channel-centre span (dispatch 8034–8536).
CUBE_VOPT_LO_KMS = 8034.059711054924
CUBE_VOPT_HI_KMS = 8535.65720660806

N_BIN = 4
TIME_BIN_S = 30.0
UV_BIN_M = 10.0
TRIM_MARGIN_NATIVE = 3
N_GUARD = 1
NATIVE_N_ROW = 43240
NATIVE_N_CHAN = 1920
MS2KINUV_SCHEMA_VERSION = "ms2kinuv-npz-v1"
MS2KINUV_PROVENANCE_SCHEMA_VERSION = "ms2kinuv-npz-v2"

MS2KINUV_V2_KEYS = (
    "uvw_m",
    "row_id",
    "antenna1",
    "antenna2",
    "scan_number",
    "observation_id",
    "array_id",
    "state_id",
    "field_id",
    "data_desc_id",
    "time_centroid",
    "interval",
    "flags",
    "channel_width_hz",
    "channel_edges_hz",
    "spectral_window_id",
    "polarization_id",
    "polarization_index",
    "correlation_type",
    "source_polarization_indices",
    "source_correlation_types",
    "polarization_combination_json",
    "frequency_reference_code",
    "frequency_frame",
    "visibility_unit",
    "weight_convention",
    "history_json",
    "smoothing_history_json",
    "extraction_json",
    "source_row_id",
    "aggregation_coefficients",
)


@dataclass
class VisData:
    """Fit-array visibilities plus the native model axis (trim + guards)."""

    u_m: np.ndarray
    v_m: np.ndarray
    vis: np.ndarray
    weights: np.ndarray
    freqs: np.ndarray
    vel: np.ndarray
    freqs_native: np.ndarray
    vel_native: np.ndarray
    n_bin: int
    dv_kms: float
    s: float
    phase_dir_rad: np.ndarray
    line_free_mask: np.ndarray
    n_guard: int = N_GUARD
    weights_native: np.ndarray | None = None
    v_lo_line: float = 0.0
    v_hi_line: float = 0.0


@dataclass
class NativeVisTable:
    """Native spectral visibility table before kinUV aggregation."""

    u_m: np.ndarray
    v_m: np.ndarray
    vis: np.ndarray
    weights: np.ndarray
    freqs: np.ndarray
    time: np.ndarray | None
    baseline: np.ndarray | None
    phase_dir_rad: np.ndarray | None
    schema: str
    uv_ref_hz: float | None
    uvw_m: np.ndarray | None = None
    row_id: np.ndarray | None = None
    antenna1: np.ndarray | None = None
    antenna2: np.ndarray | None = None
    scan_number: np.ndarray | None = None
    observation_id: np.ndarray | None = None
    array_id: np.ndarray | None = None
    state_id: np.ndarray | None = None
    field_id: np.ndarray | None = None
    data_desc_id: np.ndarray | None = None
    time_centroid: np.ndarray | None = None
    interval: np.ndarray | None = None
    flags: np.ndarray | None = None
    channel_width_hz: np.ndarray | None = None
    channel_edges_hz: np.ndarray | None = None
    spectral_window_id: int | None = None
    polarization_id: int | None = None
    polarization_index: int | None = None
    correlation_type: int | None = None
    source_polarization_indices: np.ndarray | None = None
    source_correlation_types: np.ndarray | None = None
    polarization_combination: dict | None = None
    frequency_reference_code: int | None = None
    frequency_frame: str | None = None
    visibility_unit: str | None = None
    weight_convention: str | None = None
    history: dict | None = None
    smoothing_history: dict | None = None
    extraction: dict | None = None
    source_row_id: np.ndarray | None = None
    aggregation_coefficients: np.ndarray | None = None

    @property
    def fold_safe(self) -> bool:
        """Whether the table carries the complete native S2 grouping contract."""
        return self.schema == MS2KINUV_PROVENANCE_SCHEMA_VERSION


def require_s2_provenance(table: NativeVisTable) -> None:
    """Fail closed unless an unaveraged, grouping-complete v2 export is loaded."""
    if not table.fold_safe:
        raise ValueError(
            "S2 requires an ms2kinuv-npz-v2 export with native row provenance"
        )
    if table.extraction.get("source_row_identity_preserved") is not True:
        raise ValueError("S2 requires preserved Measurement Set row identity")
    if table.extraction.get("row_averaging") != "none":
        raise ValueError("S2 folds must be assigned before row averaging")
    if table.extraction.get("channel_averaging") != "none":
        raise ValueError("S2 folds must be assigned before channel averaging")
    if not str(table.visibility_unit).lower().startswith("jy"):
        raise ValueError("S2 requires calibrated visibility flux density in Jy")
    if "2_over_complex_noise_variance" not in str(table.weight_convention):
        raise ValueError(
            "S2 requires w=2/E[|complex noise|^2]=1/sigma_component^2"
        )


def load_visibility_table(path) -> NativeVisTable:
    """Load canonical ms2kinuv NPZ or the retained historical wavelength schema."""
    npz_path = Path(path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)
    with np.load(npz_path, mmap_mode="r", allow_pickle=False) as z:
        required = ("vis", "weights", "freqs")
        missing = [key for key in required if key not in z.files]
        if missing:
            raise KeyError(f"{npz_path} missing keys {missing}")
        freqs = np.asarray(z["freqs"], dtype=np.float64).ravel()
        if "u_m" in z.files and "v_m" in z.files:
            if "schema_version" in z.files:
                version = str(np.asarray(z["schema_version"]).item())
                if version not in (
                    MS2KINUV_SCHEMA_VERSION,
                    MS2KINUV_PROVENANCE_SCHEMA_VERSION,
                ):
                    raise ValueError(
                        f"{npz_path}: unsupported schema_version {version!r}; "
                        "expected an ms2kinuv v1 or v2 schema"
                    )
            u_m = np.asarray(z["u_m"], dtype=np.float64)
            v_m = np.asarray(z["v_m"], dtype=np.float64)
            schema = version if "schema_version" in z.files else "unversioned-metres"
            uv_ref_hz = None
        elif "u" in z.files and "v" in z.files:
            uv_ref_hz = float(np.mean(freqs))
            scale = C_LIGHT_M_S / uv_ref_hz
            u_m = np.asarray(z["u"], dtype=np.float64) * scale
            v_m = np.asarray(z["v"], dtype=np.float64) * scale
            schema = "historical-reference-wavelengths"
        else:
            raise KeyError(f"{npz_path} missing u_m/v_m coordinates")
        vis = np.asarray(z["vis"], dtype=np.complex128)
        weights = np.asarray(z["weights"], dtype=np.float64)
        time = (
            np.asarray(z["time"], dtype=np.float64).ravel()
            if "time" in z.files
            else None
        )
        baseline = (
            np.asarray(z["baseline"], dtype=np.int64).ravel()
            if "baseline" in z.files
            else None
        )
        phase_dir = (
            np.asarray(z["phase_dir_rad"], dtype=np.float64).reshape(2)
            if "phase_dir_rad" in z.files
            else None
        )
        provenance = {}
        if schema == MS2KINUV_PROVENANCE_SCHEMA_VERSION:
            missing_v2 = [key for key in MS2KINUV_V2_KEYS if key not in z.files]
            if missing_v2:
                raise KeyError(f"{npz_path} missing v2 provenance keys {missing_v2}")
            row_int = (
                "row_id",
                "antenna1",
                "antenna2",
                "scan_number",
                "observation_id",
                "array_id",
                "state_id",
                "field_id",
                "data_desc_id",
                "source_row_id",
            )
            for key in row_int:
                provenance[key] = np.asarray(z[key], dtype=np.int64).ravel()
            for key in ("time_centroid", "interval", "aggregation_coefficients"):
                provenance[key] = np.asarray(z[key], dtype=np.float64).ravel()
            provenance.update(
                uvw_m=np.asarray(z["uvw_m"], dtype=np.float64),
                flags=np.asarray(z["flags"], dtype=bool),
                channel_width_hz=np.asarray(z["channel_width_hz"], dtype=np.float64),
                channel_edges_hz=np.asarray(z["channel_edges_hz"], dtype=np.float64),
            )
            for key in (
                "spectral_window_id",
                "polarization_id",
                "polarization_index",
                "correlation_type",
                "frequency_reference_code",
            ):
                provenance[key] = int(np.asarray(z[key]).item())
            provenance["source_polarization_indices"] = np.asarray(
                z["source_polarization_indices"], dtype=np.int64
            ).ravel()
            provenance["source_correlation_types"] = np.asarray(
                z["source_correlation_types"], dtype=np.int64
            ).ravel()
            for key in ("frequency_frame", "visibility_unit", "weight_convention"):
                provenance[key] = str(np.asarray(z[key]).item())
            for key, output_key in (
                ("history_json", "history"),
                ("smoothing_history_json", "smoothing_history"),
                ("extraction_json", "extraction"),
                ("polarization_combination_json", "polarization_combination"),
            ):
                provenance[output_key] = json.loads(str(np.asarray(z[key]).item()))
    n_row, n_chan = vis.shape
    if weights.shape != vis.shape:
        raise ValueError(f"weights shape {weights.shape} != vis shape {vis.shape}")
    if u_m.shape != (n_row,) or v_m.shape != (n_row,):
        raise ValueError("visibility coordinates must have one value per row")
    if freqs.shape != (n_chan,):
        raise ValueError("freqs must have one value per visibility channel")
    if (time is None) != (baseline is None):
        raise ValueError("time and baseline metadata must be present together")
    if schema == MS2KINUV_PROVENANCE_SCHEMA_VERSION:
        for key in (
            "row_id",
            "antenna1",
            "antenna2",
            "scan_number",
            "observation_id",
            "array_id",
            "state_id",
            "field_id",
            "data_desc_id",
            "time_centroid",
            "interval",
            "source_row_id",
            "aggregation_coefficients",
        ):
            if provenance[key].shape != (n_row,):
                raise ValueError(f"{key} must have one value per visibility row")
        if provenance["uvw_m"].shape != (n_row, 3):
            raise ValueError("uvw_m must have shape (n_row, 3)")
        if provenance["flags"].shape != vis.shape:
            raise ValueError("flags must match the visibility array")
        if provenance["channel_width_hz"].shape != (n_chan,):
            raise ValueError("channel_width_hz must match freqs")
        if provenance["channel_edges_hz"].shape != (n_chan, 2):
            raise ValueError("channel_edges_hz must have shape (n_chan, 2)")
        if provenance["source_polarization_indices"].size == 0 or (
            provenance["source_polarization_indices"].shape
            != provenance["source_correlation_types"].shape
        ):
            raise ValueError("source polarization indices and types must align")
        if np.any(weights[provenance["flags"]] != 0.0):
            raise ValueError("flagged cells must have zero weight")
        if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
            raise ValueError("weights must be finite and nonnegative")
        if not np.array_equal(
            provenance["uvw_m"][:, 0].astype(np.float32), u_m.astype(np.float32)
        ) or not np.array_equal(
            provenance["uvw_m"][:, 1].astype(np.float32), v_m.astype(np.float32)
        ):
            raise ValueError("u_m/v_m disagree with uvw_m")
        expected_baseline = (
            provenance["antenna1"] << np.int64(32)
        ) | provenance["antenna2"]
        if not np.array_equal(baseline, expected_baseline):
            raise ValueError("baseline encoding disagrees with antenna pair")
        if np.any(provenance["channel_width_hz"] <= 0.0):
            raise ValueError("channel widths must be positive")
        edges = provenance["channel_edges_hz"]
        if np.any(edges[:, 0] >= freqs) or np.any(edges[:, 1] <= freqs):
            raise ValueError("each channel centre must lie within ordered edges")
        if not np.array_equal(provenance["row_id"], provenance["source_row_id"]):
            raise ValueError("unaveraged v2 export must preserve source row identity")
        if not np.all(provenance["aggregation_coefficients"] == 1.0):
            raise ValueError("unaveraged v2 export must have unit coefficients")
    return NativeVisTable(
        u_m=u_m,
        v_m=v_m,
        vis=vis,
        weights=weights,
        freqs=freqs,
        time=time,
        baseline=baseline,
        phase_dir_rad=phase_dir,
        schema=schema,
        uv_ref_hz=uv_ref_hz,
        **provenance,
    )


def optical_to_radio_kms(v_opt_kms, c_kms: float = C_LIGHT_KM_S):
    """``v_radio = v_opt / (1 + v_opt/c)``. Cube is VOPT; vis are radio."""
    v = np.asarray(v_opt_kms, dtype=np.float64)
    return v / (1.0 + v / c_kms)


def radio_to_optical_kms(v_rad_kms, c_kms: float = C_LIGHT_KM_S):
    """Inverse of :func:`optical_to_radio_kms`."""
    v = np.asarray(v_rad_kms, dtype=np.float64)
    return v / (1.0 - v / c_kms)


def cube_vopt_window_kms(cube_path: Path | None = DEFAULT_CUBE) -> tuple[float, float]:
    """Channel-centre VOPT min/max from the Ico cube, or the dispatch span."""
    path = Path(cube_path) if cube_path is not None else DEFAULT_CUBE
    if path.is_file():
        try:
            from astropy.io import fits
        except ImportError:
            return CUBE_VOPT_LO_KMS, CUBE_VOPT_HI_KMS
        header = fits.getheader(path)
        n = int(header["NAXIS3"])
        vel = float(header["CRVAL3"]) + (
            np.arange(1, n + 1, dtype=np.float64) - float(header["CRPIX3"])
        ) * float(header["CDELT3"])
        return float(np.min(vel)), float(np.max(vel))
    return CUBE_VOPT_LO_KMS, CUBE_VOPT_HI_KMS


def _weighted_row_reduce(u_m, v_m, vis, weights, inv, n_grp):
    """Weighted mean vis per group ``inv``; uv centroid uses row-sum weights."""
    u64 = np.asarray(u_m, dtype=np.float64)
    v64 = np.asarray(v_m, dtype=np.float64)
    w64 = np.asarray(weights, dtype=np.float64)
    vis_c = np.asarray(vis, dtype=np.complex128)
    w_row = np.sum(w64, axis=1)
    u_acc = np.zeros(n_grp, dtype=np.float64)
    v_acc = np.zeros(n_grp, dtype=np.float64)
    w_uv = np.zeros(n_grp, dtype=np.float64)
    np.add.at(u_acc, inv, u64 * w_row)
    np.add.at(v_acc, inv, v64 * w_row)
    np.add.at(w_uv, inv, w_row)
    safe = np.maximum(w_uv, 1e-40)
    u_out = u_acc / safe
    v_out = v_acc / safe
    numer = np.zeros((n_grp, vis_c.shape[1]), dtype=np.complex128)
    denom = np.zeros((n_grp, vis_c.shape[1]), dtype=np.float64)
    np.add.at(numer, inv, vis_c * w64)
    np.add.at(denom, inv, w64)
    vis_out = np.divide(
        numer, denom, out=np.zeros_like(numer), where=denom > 0.0
    )
    return u_out, v_out, vis_out, denom


def average_time_steps(
    u_m,
    v_m,
    vis,
    weights,
    time_s,
    bin_s: float,
    baseline_ids,
):
    """Average visibilities in ``bin_s``-second bins per physical baseline."""
    if bin_s <= 0.0:
        raise ValueError(f"bin_s must be positive; got {bin_s}")
    vis = np.asarray(vis)
    weights = np.asarray(weights)
    if vis.shape != weights.shape:
        raise ValueError("vis and weights must have the same shape")
    nrow = vis.shape[0]
    if u_m.shape != (nrow,) or v_m.shape != (nrow,):
        raise ValueError("u_m, v_m must be 1D with length n_row")
    time_s = np.asarray(time_s, dtype=np.float64).ravel()
    baseline_ids = np.asarray(baseline_ids, dtype=np.int64).ravel()
    if time_s.shape[0] != nrow or baseline_ids.shape[0] != nrow:
        raise ValueError("time_s and baseline_ids must match vis row count")

    t_rel = time_s - np.min(time_s)
    tb = np.floor(t_rel / bin_s).astype(np.int64)
    stack = np.column_stack([tb, baseline_ids])
    _, inv = np.unique(stack, axis=0, return_inverse=True)
    n_grp = int(inv.max()) + 1 if inv.size else 0
    return _weighted_row_reduce(u_m, v_m, vis, weights, inv, n_grp)


def bin_uv_plane(u_m, v_m, vis, weights, bin_size_m: float):
    """Grid-average visibilities in ``bin_size_m``-metre UV cells."""
    if bin_size_m <= 0.0:
        raise ValueError(f"bin_size_m must be positive; got {bin_size_m}")
    vis = np.asarray(vis)
    weights = np.asarray(weights)
    if vis.shape != weights.shape:
        raise ValueError("vis and weights must have the same shape")
    nrow = vis.shape[0]
    if u_m.shape != (nrow,) or v_m.shape != (nrow,):
        raise ValueError("u_m, v_m must be 1D with length n_row")

    u64 = np.asarray(u_m, dtype=np.float64)
    v64 = np.asarray(v_m, dtype=np.float64)
    iu = np.floor(u64 / bin_size_m).astype(np.int64)
    iv = np.floor(v64 / bin_size_m).astype(np.int64)
    pairs = np.column_stack([iu, iv])
    _, inv = np.unique(pairs, axis=0, return_inverse=True)
    n_bins = int(inv.max()) + 1 if inv.size else 0
    return _weighted_row_reduce(u_m, v_m, vis, weights, inv, n_bins)


def _trim_and_guard_indices(vel, v_lo_line, v_hi_line, *, margin, n_guard):
    dv = float(np.median(np.abs(np.diff(vel)))) if vel.size > 1 else 1.0
    v_lo_trim = float(v_lo_line) - margin * dv
    v_hi_trim = float(v_hi_line) + margin * dv
    trim = (vel >= v_lo_trim) & (vel <= v_hi_trim)
    if int(np.sum(trim)) < 2:
        raise ValueError(
            f"Cube trim [{v_lo_trim:.1f}, {v_hi_trim:.1f}] km/s leaves "
            f"{int(np.sum(trim))} native channels"
        )
    idx = np.flatnonzero(trim)
    i0, i1 = int(idx[0]), int(idx[-1])
    n_chan = int(vel.shape[0])
    g0 = i0 - n_guard
    g1 = i1 + n_guard
    dnu = float(np.median(np.diff(vel))) if vel.size > 1 else dv
    extra_lo = extra_hi = 0
    if g0 < 0:
        extra_lo = -g0
        g0 = 0
    if g1 > n_chan - 1:
        extra_hi = g1 - (n_chan - 1)
        g1 = n_chan - 1
    return i0, i1, g0, g1, dv, extra_lo, extra_hi, dnu


def _extend_axis(freq_core, vel_core, extra_lo, extra_hi, dvel):
    """If the npz has no extra channels, evaluate ν_edge ± Δν_native."""
    freqs = np.asarray(freq_core, dtype=np.float64)
    vel = np.asarray(vel_core, dtype=np.float64)
    if extra_lo:
        df = float(np.median(np.diff(freqs))) if freqs.size > 1 else 0.0
        lo_f = freqs[0] + df * np.arange(-extra_lo, 0, dtype=np.float64)
        lo_v = vel[0] + dvel * np.arange(-extra_lo, 0, dtype=np.float64)
        freqs = np.concatenate([lo_f, freqs])
        vel = np.concatenate([lo_v, vel])
    if extra_hi:
        df = float(np.median(np.diff(freqs))) if freqs.size > 1 else 0.0
        hi_f = freqs[-1] + df * np.arange(1, extra_hi + 1, dtype=np.float64)
        hi_v = vel[-1] + dvel * np.arange(1, extra_hi + 1, dtype=np.float64)
        freqs = np.concatenate([freqs, hi_f])
        vel = np.concatenate([vel, hi_v])
    return freqs, vel


def load_target_vis(
    path,
    *,
    cube_path,
    phase_dir_rad=None,
    row_mask=None,
    n_bin: int = N_BIN,
    time_bin_s: float = TIME_BIN_S,
    uv_bin_m: float = UV_BIN_M,
    trim_margin: int = TRIM_MARGIN_NATIVE,
    n_guard: int = N_GUARD,
) -> tuple[VisData, dict]:
    """Prepare a configured target, optionally from a native-row subset.

    ``row_mask`` is applied before every time/uv aggregation.  This ordering is
    required for correlation-aware train/validation splits: no averaged output
    row may mix native Measurement Set rows from different folds.
    """
    table = load_visibility_table(path)
    freqs_all = table.freqs
    vel_all = freq_to_velocity_kms(freqs_all)
    v_lo_opt, v_hi_opt = cube_vopt_window_kms(cube_path)
    v_lo_line = float(optical_to_radio_kms(v_lo_opt))
    v_hi_line = float(optical_to_radio_kms(v_hi_opt))
    i0, i1, g0, g1, dv_native, extra_lo, extra_hi, dvel = _trim_and_guard_indices(
        vel_all,
        v_lo_line,
        v_hi_line,
        margin=int(trim_margin),
        n_guard=int(n_guard),
    )
    sl = slice(i0, i1 + 1)
    if row_mask is None:
        rows = slice(None)
        selected_native_rows = int(table.vis.shape[0])
    else:
        rows = np.asarray(row_mask, dtype=bool).ravel()
        if rows.shape != (table.vis.shape[0],):
            raise ValueError("row_mask must contain one value per native row")
        selected_native_rows = int(np.sum(rows))
        if selected_native_rows == 0:
            raise ValueError("row_mask selects no native visibility rows")
    vis = table.vis[rows, sl]
    weights = table.weights[rows, sl]
    u_m = table.u_m[rows]
    v_m = table.v_m[rows]
    freqs_trim = freqs_all[sl]
    vel_trim = vel_all[sl]
    freqs_native = freqs_all[g0 : g1 + 1]
    vel_native = vel_all[g0 : g1 + 1]
    freqs_native, vel_native = _extend_axis(
        freqs_native, vel_native, extra_lo, extra_hi, dvel
    )
    time_average = table.time is not None
    if time_average:
        u_m, v_m, vis, weights = average_time_steps(
            u_m,
            v_m,
            vis,
            weights,
            table.time[rows],
            float(time_bin_s),
            table.baseline[rows],
        )
    u_m, v_m, vis, weights = bin_uv_plane(
        u_m, v_m, vis, weights, float(uv_bin_m)
    )
    vis_b, w_b, vel_b, freqs_b, _ = bin_channels(
        vis, weights, vel_trim, freqs_trim, int(n_bin)
    )
    dv_kms = (
        float(np.median(np.abs(np.diff(vel_b))))
        if vel_b.size > 1
        else float(n_bin) * dv_native
    )
    line_free = (vel_b < v_lo_line) | (vel_b > v_hi_line)
    s = empirical_s(vis_b, w_b, line_free)
    phase = table.phase_dir_rad
    if phase is None:
        if phase_dir_rad is None:
            raise ValueError(
                "phase_dir_rad is absent from NPZ and no target fallback was given"
            )
        phase = np.asarray(phase_dir_rad, dtype=np.float64).reshape(2)
    data = VisData(
        u_m=u_m,
        v_m=v_m,
        vis=vis_b,
        weights=w_b,
        freqs=freqs_b,
        vel=vel_b,
        freqs_native=freqs_native,
        vel_native=vel_native,
        n_bin=int(n_bin),
        dv_kms=dv_kms,
        s=s,
        phase_dir_rad=phase,
        line_free_mask=line_free,
        n_guard=int(n_guard),
        weights_native=weights,
        v_lo_line=v_lo_line,
        v_hi_line=v_hi_line,
    )
    meta = {
        "schema": table.schema,
        "uv_ref_hz": table.uv_ref_hz,
        "time_average": time_average,
        "time_bin_s": float(time_bin_s) if time_average else None,
        "uv_bin_m": float(uv_bin_m),
        "native_rows_total": int(table.vis.shape[0]),
        "native_rows_selected": selected_native_rows,
        "row_subset_before_aggregation": row_mask is not None,
    }
    return data, meta


@requires("DEC-066-VIS", "DEC-066-SPECRESP", "DEC-066-WEIGHT", "DEC-066-ZEROMODEL")
def load_kgas066(
    path=DEFAULT_NPZ,
    *,
    cube_path=DEFAULT_CUBE,
    n_bin: int = N_BIN,
    time_bin_s: float = TIME_BIN_S,
    uv_bin_m: float = UV_BIN_M,
    trim_margin: int = TRIM_MARGIN_NATIVE,
    n_guard: int = N_GUARD,
) -> VisData:
    """Load native KGAS066, trim to the Ico cube + margin, aggregate, bin ``N``.

    Does **not** Hann the data. Records ``(n_row, n_chan, Δv_kms, N)`` on the
    returned :class:`VisData`. ``s`` is measured on line-free **fit** channels.
    """
    npz_path = Path(path)
    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    z = np.load(npz_path, mmap_mode="r")
    required = ("u_m", "v_m", "vis", "weights", "freqs", "time", "baseline")
    missing = [k for k in required if k not in z.files]
    if missing:
        raise KeyError(f"{npz_path} missing keys {missing}")

    freqs_all = np.asarray(z["freqs"], dtype=np.float64).ravel()
    vel_all = freq_to_velocity_kms(freqs_all)
    v_lo_opt, v_hi_opt = cube_vopt_window_kms(cube_path)
    v_lo_line = float(optical_to_radio_kms(v_lo_opt))
    v_hi_line = float(optical_to_radio_kms(v_hi_opt))
    i0, i1, g0, g1, dv_native, extra_lo, extra_hi, dvel = _trim_and_guard_indices(
        vel_all,
        v_lo_line,
        v_hi_line,
        margin=int(trim_margin),
        n_guard=int(n_guard),
    )

    sl = slice(i0, i1 + 1)
    vis = np.asarray(z["vis"][:, sl], dtype=np.complex128)
    weights = np.asarray(z["weights"][:, sl], dtype=np.float64)
    u_m = np.asarray(z["u_m"], dtype=np.float64)
    v_m = np.asarray(z["v_m"], dtype=np.float64)
    time_s = np.asarray(z["time"], dtype=np.float64).ravel()
    baseline = np.asarray(z["baseline"], dtype=np.int64).ravel()
    phase_dir = np.asarray(z["phase_dir_rad"], dtype=np.float64).ravel()
    freqs_trim = freqs_all[sl]
    vel_trim = vel_all[sl]
    freqs_native = freqs_all[g0 : g1 + 1]
    vel_native = vel_all[g0 : g1 + 1]
    freqs_native, vel_native = _extend_axis(
        freqs_native, vel_native, extra_lo, extra_hi, dvel
    )

    u_m, v_m, vis, weights = average_time_steps(
        u_m, v_m, vis, weights, time_s, float(time_bin_s), baseline
    )
    u_m, v_m, vis, weights = bin_uv_plane(u_m, v_m, vis, weights, float(uv_bin_m))

    vis_b, w_b, vel_b, freqs_b, _ = bin_channels(
        vis, weights, vel_trim, freqs_trim, int(n_bin)
    )
    dv_kms = (
        float(np.median(np.abs(np.diff(vel_b)))) if vel_b.size > 1 else float(n_bin) * dv_native
    )
    line_free = (vel_b < v_lo_line) | (vel_b > v_hi_line)
    s = empirical_s(vis_b, w_b, line_free)
    return VisData(
        u_m=u_m,
        v_m=v_m,
        vis=vis_b,
        weights=w_b,
        freqs=freqs_b,
        vel=vel_b,
        freqs_native=freqs_native,
        vel_native=vel_native,
        n_bin=int(n_bin),
        dv_kms=dv_kms,
        s=s,
        phase_dir_rad=phase_dir,
        line_free_mask=line_free,
        n_guard=int(n_guard),
        weights_native=weights,
        v_lo_line=v_lo_line,
        v_hi_line=v_hi_line,
    )
