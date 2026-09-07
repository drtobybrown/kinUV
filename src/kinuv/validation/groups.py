"""Correlation-aware native-row groups for held-out visibility validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kinuv.io.vis import NativeVisTable, require_s2_provenance


@dataclass(frozen=True)
class VisibilityGroup:
    """One contiguous within-scan time block spanning all baselines."""

    group_id: int
    fold_id: int
    observation_id: int
    array_id: int
    scan_number: int
    time_block_index: int
    state_id: int
    field_id: int
    data_desc_id: int
    start_time_s: float
    end_time_s: float
    n_rows: int
    n_baselines: int


@dataclass(frozen=True)
class GroupedVisibilityFolds:
    """Native-row fold assignment with optional time-boundary embargo."""

    row_group_id: np.ndarray
    row_fold_id: np.ndarray
    groups: tuple[VisibilityGroup, ...]
    n_folds: int

    def validation_mask(self, fold_id: int) -> np.ndarray:
        self._check_fold(fold_id)
        return self.row_fold_id == int(fold_id)

    def training_mask(self, fold_id: int, *, embargo_s: float = 0.0) -> np.ndarray:
        """Exclude the validation fold and rows within its time embargo."""
        self._check_fold(fold_id)
        if embargo_s < 0.0:
            raise ValueError("embargo_s must be nonnegative")
        validation = [group for group in self.groups if group.fold_id == fold_id]
        keep_group = np.ones(len(self.groups), dtype=bool)
        for candidate in self.groups:
            if candidate.fold_id == fold_id:
                keep_group[candidate.group_id] = False
                continue
            for held_out in validation:
                same_stratum = (
                    candidate.observation_id == held_out.observation_id
                    and candidate.array_id == held_out.array_id
                    and candidate.field_id == held_out.field_id
                    and candidate.data_desc_id == held_out.data_desc_id
                )
                if not same_stratum:
                    continue
                separated = (
                    candidate.end_time_s < held_out.start_time_s - embargo_s
                    or candidate.start_time_s > held_out.end_time_s + embargo_s
                )
                if not separated:
                    keep_group[candidate.group_id] = False
                    break
        return keep_group[self.row_group_id]

    def _check_fold(self, fold_id: int) -> None:
        if not 0 <= int(fold_id) < self.n_folds:
            raise ValueError(f"fold_id must be in 0..{self.n_folds - 1}")


def build_grouped_visibility_folds(
    table: NativeVisTable,
    *,
    n_folds: int = 5,
    integrations_per_group: int = 5,
) -> GroupedVisibilityFolds:
    """Assign within-scan time blocks to contiguous, balanced folds.

    The indivisible key uses standard Measurement Set identities plus a rank
    over native integration timestamps. Antenna pair is deliberately excluded
    so every baseline observed within a time block stays in the same fold.
    Assignment happens before any time, uv, or channel aggregation.
    """
    require_s2_provenance(table)
    if n_folds < 2:
        raise ValueError("n_folds must be at least two")
    if integrations_per_group < 1:
        raise ValueError("integrations_per_group must be positive")
    scan_keys = np.column_stack(
        [
            table.observation_id,
            table.array_id,
            table.scan_number,
            table.state_id,
            table.field_id,
            table.data_desc_id,
        ]
    ).astype(np.int64, copy=False)
    _, scan_inverse = np.unique(scan_keys, axis=0, return_inverse=True)
    time_block = np.empty(table.time_centroid.size, dtype=np.int64)
    for scan_id in np.unique(scan_inverse):
        rows = scan_inverse == scan_id
        integrations, integration_index = np.unique(
            table.time_centroid[rows], return_inverse=True
        )
        if integrations.size == 0:
            raise ValueError("scan group contains no integrations")
        time_block[rows] = integration_index // int(integrations_per_group)
    keys = np.column_stack(
        [
            table.observation_id,
            table.array_id,
            table.scan_number,
            table.state_id,
            table.field_id,
            table.data_desc_id,
            time_block,
        ]
    ).astype(np.int64, copy=False)
    unique_keys, inverse = np.unique(keys, axis=0, return_inverse=True)
    n_group = unique_keys.shape[0]
    if n_group < n_folds:
        raise ValueError(
            f"need at least {n_folds} independent scan groups; found {n_group}"
        )

    starts = np.empty(n_group, dtype=np.float64)
    ends = np.empty(n_group, dtype=np.float64)
    counts = np.bincount(inverse, minlength=n_group).astype(np.int64)
    n_baselines = np.empty(n_group, dtype=np.int64)
    for group_id in range(n_group):
        rows = inverse == group_id
        starts[group_id] = float(np.min(table.time_centroid[rows] - table.interval[rows] / 2.0))
        ends[group_id] = float(np.max(table.time_centroid[rows] + table.interval[rows] / 2.0))
        pairs = np.column_stack([table.antenna1[rows], table.antenna2[rows]])
        n_baselines[group_id] = np.unique(pairs, axis=0).shape[0]

    group_fold = np.full(n_group, -1, dtype=np.int64)
    covariance_strata = unique_keys[:, [0, 1, 4, 5]]
    unique_strata, stratum_inverse = np.unique(
        covariance_strata, axis=0, return_inverse=True
    )
    for stratum_id, stratum_key in enumerate(unique_strata):
        positions = np.flatnonzero(stratum_inverse == stratum_id)
        if positions.size < n_folds:
            raise ValueError(
                "each covariance stratum needs at least "
                f"{n_folds} scan groups; stratum {stratum_key.tolist()} has "
                f"{positions.size}"
            )
        order = positions[
            np.lexsort((unique_keys[positions, 2], starts[positions]))
        ]
        for fold_id, block in enumerate(np.array_split(order, n_folds)):
            group_fold[block] = fold_id
    if np.any(group_fold < 0):
        raise RuntimeError("internal error: unassigned visibility group")
    row_fold = group_fold[inverse]

    groups = tuple(
        VisibilityGroup(
            group_id=group_id,
            fold_id=int(group_fold[group_id]),
            observation_id=int(unique_keys[group_id, 0]),
            array_id=int(unique_keys[group_id, 1]),
            scan_number=int(unique_keys[group_id, 2]),
            time_block_index=int(unique_keys[group_id, 6]),
            state_id=int(unique_keys[group_id, 3]),
            field_id=int(unique_keys[group_id, 4]),
            data_desc_id=int(unique_keys[group_id, 5]),
            start_time_s=float(starts[group_id]),
            end_time_s=float(ends[group_id]),
            n_rows=int(counts[group_id]),
            n_baselines=int(n_baselines[group_id]),
        )
        for group_id in range(n_group)
    )
    return GroupedVisibilityFolds(
        row_group_id=inverse.astype(np.int64, copy=False),
        row_fold_id=row_fold,
        groups=groups,
        n_folds=int(n_folds),
    )
