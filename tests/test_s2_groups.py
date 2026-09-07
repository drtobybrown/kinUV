"""S2 native-row grouping and fold isolation."""

from __future__ import annotations

import numpy as np

from kinuv.io.vis import MS2KINUV_PROVENANCE_SCHEMA_VERSION, NativeVisTable
from kinuv.validation.groups import build_grouped_visibility_folds


def _table(n_scan=10, rows_per_scan=6):
    n_row = n_scan * rows_per_scan
    scan = np.repeat(np.arange(n_scan), rows_per_scan)
    time = scan.astype(float) * 100.0 + np.tile(np.arange(rows_per_scan), n_scan)
    ant1 = np.tile(np.array([0, 0, 1, 1, 2, 2]), n_scan)
    ant2 = np.tile(np.array([1, 2, 2, 3, 3, 4]), n_scan)
    return NativeVisTable(
        u_m=np.zeros(n_row),
        v_m=np.zeros(n_row),
        vis=np.zeros((n_row, 2), dtype=np.complex128),
        weights=np.ones((n_row, 2)),
        freqs=np.array([1.0, 2.0]),
        time=time,
        baseline=(ant1 << np.int64(32)) | ant2,
        phase_dir_rad=np.zeros(2),
        schema=MS2KINUV_PROVENANCE_SCHEMA_VERSION,
        uv_ref_hz=None,
        row_id=np.arange(n_row),
        antenna1=ant1,
        antenna2=ant2,
        scan_number=scan,
        observation_id=np.zeros(n_row, dtype=int),
        array_id=np.zeros(n_row, dtype=int),
        state_id=np.zeros(n_row, dtype=int),
        field_id=np.zeros(n_row, dtype=int),
        data_desc_id=np.zeros(n_row, dtype=int),
        time_centroid=time,
        interval=np.ones(n_row),
        visibility_unit="Jy",
        weight_convention="ms_weight_equals_2_over_complex_noise_variance",
        extraction={
            "source_row_identity_preserved": True,
            "row_averaging": "none",
            "channel_averaging": "none",
        },
    )


def test_scan_groups_are_indivisible_and_cover_five_folds():
    table = _table()
    folds = build_grouped_visibility_folds(table, n_folds=5)
    assert set(folds.row_fold_id) == set(range(5))
    for scan in np.unique(table.scan_number):
        assert np.unique(folds.row_fold_id[table.scan_number == scan]).size == 1
    for fold_id in range(5):
        assert not np.any(
            folds.validation_mask(fold_id) & folds.training_mask(fold_id)
        )


def test_time_embargo_removes_neighboring_scan_blocks():
    table = _table()
    folds = build_grouped_visibility_folds(table, n_folds=5)
    held = folds.validation_mask(2)
    plain = folds.training_mask(2)
    embargoed = folds.training_mask(2, embargo_s=150.0)
    assert np.sum(embargoed) < np.sum(plain)
    assert not np.any(held & embargoed)


def test_too_few_scan_groups_fails_closed():
    table = _table(n_scan=4)
    with np.testing.assert_raises_regex(ValueError, "at least 5"):
        build_grouped_visibility_folds(
            table, n_folds=5, integrations_per_group=100
        )
