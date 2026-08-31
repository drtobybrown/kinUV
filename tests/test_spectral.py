"""DEC-066-SPECRESP: native Hann then bin; not Hann on a binned axis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kinuv.response.spectral import (
    HANN_KERNEL,
    bin_channels,
    hann_native,
    hann_then_bin,
    native_diagonal,
    rho_bin,
    s_theory,
)

SRC = Path(__file__).resolve().parents[1] / "src" / "kinuv"


def test_s_theory_n4_is_6_over_13_n8_is_12_over_29():
    assert s_theory(4) == pytest.approx(6.0 / 13.0)
    assert s_theory(8) == pytest.approx(12.0 / 29.0)
    assert rho_bin(8) == pytest.approx(3.0 / 58.0)
    assert rho_bin(4) == pytest.approx(3.0 / 26.0)


def test_white_noise_hann_bin_recovers_s_theory_and_rho():
    rng = np.random.default_rng(20260818)
    n_rep, n_chan = 96, 4096
    x = rng.standard_normal((n_rep, n_chan + 2))
    h = hann_native(x, axis=-1)[:, 1:-1]
    assert np.var(h) == pytest.approx(0.375, rel=0.03)
    for n_bin in (4, 8):
        n_use = (h.shape[1] // n_bin) * n_bin
        z = h[:, :n_use].reshape(n_rep, -1, n_bin).mean(axis=2)
        s_hat = (np.var(h) / n_bin) / np.var(z)
        assert s_hat == pytest.approx(s_theory(n_bin), rel=0.05)
        rho = float(np.corrcoef(z[:, :-1].ravel(), z[:, 1:].ravel())[0, 1])
        assert rho == pytest.approx(rho_bin(n_bin), rel=0.12)


def test_hann_native_then_bin_matches_reference():
    n_trim, n_bin, n_guard = 24, 4, 1
    x = np.arange(n_trim + 2 * n_guard, dtype=np.float64)
    line = np.exp(-0.5 * ((x - 8.0) / 2.5) ** 2)
    got = hann_then_bin(line, n_bin, n_guard=n_guard)
    hann = np.convolve(line, HANN_KERNEL, mode="same")
    core = hann[n_guard : n_guard + n_trim]
    n_use = (n_trim // n_bin) * n_bin
    want = core[:n_use].reshape(-1, n_bin).mean(axis=1)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12)


def test_hann_on_binned_axis_does_not_match_native():
    n_trim, n_bin, n_guard = 24, 4, 1
    x = np.arange(n_trim + 2 * n_guard, dtype=np.float64)
    line = np.exp(-0.5 * ((x - 8.0) / 2.5) ** 2)
    correct = hann_then_bin(line, n_bin, n_guard=n_guard)
    core = line[n_guard : n_guard + n_trim]
    binned = core.reshape(-1, n_bin).mean(axis=1)
    wrong = np.convolve(binned, HANN_KERNEL, mode="same")
    rel = np.max(np.abs(wrong - correct)) / np.max(np.abs(correct))
    assert rel > 0.02


def test_edge_padding_gate_unpadded_fails_guards_pass():
    n_trim, n_bin, n_guard, pad = 16, 4, 1, 8
    i0 = pad
    x = np.arange(n_trim + 2 * pad, dtype=np.float64) - i0
    native_ext = np.exp(-0.5 * (x / 0.65) ** 2)
    hann_ext = np.convolve(native_ext, HANN_KERNEL, mode="same")
    truth = hann_ext[pad : pad + n_trim].reshape(-1, n_bin).mean(axis=1)

    guarded = native_ext[pad - n_guard : pad + n_trim + n_guard]
    got = hann_then_bin(guarded, n_bin, n_guard=n_guard)
    np.testing.assert_allclose(got, truth, rtol=1e-12, atol=1e-12)

    unpadded = native_ext[pad : pad + n_trim]
    bad = np.convolve(unpadded, HANN_KERNEL, mode="same").reshape(-1, n_bin).mean(
        axis=1
    )
    assert abs(bad[0] - truth[0]) > 0.05 * abs(truth[0])
    assert abs(got[0] - truth[0]) < 1e-12


def test_bin_channels_weighted_mean_and_summed_weights():
    vis = np.array([[1.0 + 0j, 3.0 + 0j, 5.0 + 0j, 7.0 + 0j]], dtype=np.complex128)
    w = np.array([[1.0, 1.0, 3.0, 1.0]], dtype=np.float64)
    vel = np.array([0.0, 1.0, 2.0, 3.0])
    freqs = np.array([4.0, 5.0, 6.0, 7.0])
    vb, wb, velb, fb, n_drop = bin_channels(vis, w, vel, freqs, 2)
    assert n_drop == 0
    np.testing.assert_allclose(vb[0, 0], 2.0 + 0j)
    np.testing.assert_allclose(vb[0, 1], (5.0 * 3.0 + 7.0) / 4.0)
    np.testing.assert_allclose(wb[0], [2.0, 4.0])
    np.testing.assert_allclose(velb, [0.5, 2.5])
    np.testing.assert_allclose(fb, [4.5, 6.5])


def test_hann_impulse_is_not_rigidly_shifted():
    """Kernel [0.25, 0.5, 0.25] is centred; peak stays in the same N=4 bin.

    An impulse at trimmed channel 16 is the first sample of bin 4. Hann leaks
    0.25 into channel 15 (bin 3), so the centroid is 3.75, not a whole bin.
    """
    n_trim, n_bin, n_g = 40, 4, 1
    peak = 16
    native = np.zeros(n_trim + 2 * n_g)
    native[n_g + peak] = 1.0
    out = hann_then_bin(native, n_bin, n_guard=n_g)
    assert int(np.argmax(out)) == peak // n_bin
    cent = float(np.sum(np.arange(out.size) * out) / np.sum(out))
    assert cent == pytest.approx(3.75, abs=1e-12)


def test_native_diagonal_is_removed():
    with pytest.raises(RuntimeError, match="hann_then_bin"):
        native_diagonal()
    import kinuv.likelihood as lik

    assert "hann_then_bin" not in lik.__all__


def test_spectral_sources_do_not_import_uvkin():
    for name in ("response/spectral.py", "io/vis.py", "likelihood/chi2.py"):
        text = (SRC / name).read_text(encoding="utf-8")
        assert "uvkin" not in text
        assert "uv_aggregate" not in text
    vis_src = (SRC / "io" / "vis.py").read_text(encoding="utf-8")
    assert "hann_native" not in vis_src
    assert "hann_then_bin" not in vis_src
