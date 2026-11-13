"""Voxel-map construction and map-quality metrics, identical for every method.

All methods share the same scans, poses, range mask and voxel resolution; they differ only in
which points they remove. Labels are used for scoring only.

Definitions (per method, over the whole window):
  occupied_voxels           voxels containing >= 1 kept point
  contaminated_voxels       voxels containing >= 1 kept annotated-moving point
  moving_points_retained    kept annotated-moving points
  static_point_retention    kept annotated-static points / annotated-static points
  static_coverage           voxels holding >= 1 kept static point / voxels holding >= 1 static point
                            (reference = all annotated-static points, i.e. the raw map's static part)
  growth                    cumulative occupied voxels after each frame
Ignored-label points (unlabeled/outlier) are mapped like any other point but count as neither
static nor moving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_BITS = 21
_OFF = 1 << (_BITS - 1)


def voxel_keys(xyz_world: np.ndarray, resolution: float) -> np.ndarray:
    """Pack integer voxel coordinates into int64 keys (valid for |coord| < 2^20 voxels)."""
    v = np.floor(np.asarray(xyz_world) / resolution).astype(np.int64) + _OFF
    if v.min(initial=_OFF) < 0 or v.max(initial=_OFF) >= (1 << _BITS):
        raise ValueError("coordinates out of packable range")
    return (v[:, 0] << (2 * _BITS)) | (v[:, 1] << _BITS) | v[:, 2]


@dataclass
class MapAccumulator:
    resolution: float
    occupied: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    contaminated: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    static_vox: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    growth: list = field(default_factory=list)
    points_inserted: int = 0
    moving_retained: int = 0
    static_kept: int = 0
    static_total: int = 0
    moving_total: int = 0

    def add(self, keys: np.ndarray, keep: np.ndarray, gt_moving: np.ndarray, gt_static: np.ndarray) -> None:
        k = keys[keep]
        self.occupied = np.union1d(self.occupied, k)
        self.contaminated = np.union1d(self.contaminated, keys[keep & gt_moving])
        self.static_vox = np.union1d(self.static_vox, keys[keep & gt_static])
        self.growth.append(int(self.occupied.size))
        self.points_inserted += int(keep.sum())
        self.moving_retained += int((keep & gt_moving).sum())
        self.static_kept += int((keep & gt_static).sum())
        self.static_total += int(gt_static.sum())
        self.moving_total += int(gt_moving.sum())

    def summary(self, reference_static_voxels: int) -> dict:
        d = lambda a, b: a / b if b else float("nan")
        return {"resolution_m": self.resolution,
                "points_inserted": self.points_inserted,
                "occupied_voxels": int(self.occupied.size),
                "contaminated_voxels": int(self.contaminated.size),
                "contaminated_voxel_fraction": d(self.contaminated.size, self.occupied.size),
                "moving_points_retained": self.moving_retained,
                "moving_points_total": self.moving_total,
                "static_points_kept": self.static_kept,
                "static_point_retention": d(self.static_kept, self.static_total),
                "static_voxels": int(self.static_vox.size),
                "static_coverage": d(self.static_vox.size, reference_static_voxels)}
