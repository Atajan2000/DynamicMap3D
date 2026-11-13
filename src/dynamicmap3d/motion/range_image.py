"""Spherical range images for a rotating multi-beam LiDAR (default: Velodyne HDL-64E in KITTI)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter


@dataclass(frozen=True)
class RangeImageSpec:
    height: int = 64
    width: int = 2048
    fov_up_deg: float = 3.0
    fov_down_deg: float = -25.0

    def project(self, xyz: np.ndarray):
        """Return (row, col, range, in_fov) for each point. Rows/cols are clipped to the image."""
        r = np.linalg.norm(xyz, axis=1)
        safe = np.maximum(r, 1e-9)
        pitch = np.arcsin(np.clip(xyz[:, 2] / safe, -1.0, 1.0))
        yaw = np.arctan2(xyz[:, 1], xyz[:, 0])
        up, down = np.radians(self.fov_up_deg), np.radians(self.fov_down_deg)
        v = (up - pitch) / (up - down) * self.height
        u = 0.5 * (1.0 - yaw / np.pi) * self.width
        row = np.floor(v).astype(np.int64)
        col = np.floor(u).astype(np.int64) % self.width
        in_fov = (row >= 0) & (row < self.height) & (r > 1e-6)
        return np.clip(row, 0, self.height - 1), col, r, in_fov

    def min_range_image(self, xyz: np.ndarray, min_range: float = 0.0) -> np.ndarray:
        """(H, W) image of the closest return per pixel; empty pixels are +inf."""
        row, col, r, ok = self.project(xyz)
        ok &= r >= min_range
        img = np.full((self.height, self.width), np.inf, dtype=np.float32)
        np.minimum.at(img, (row[ok], col[ok]), r[ok].astype(np.float32))
        return img


def conservative_min(img: np.ndarray, half_rows: int, half_cols: int) -> np.ndarray:
    """Minimum over a (2*half_rows+1) x (2*half_cols+1) neighbourhood (columns wrap around).

    A point is only declared "seen through" if every return around its projection is farther,
    which suppresses false free-space evidence at depth discontinuities and from discretisation.
    """
    return minimum_filter(img, size=(2 * half_rows + 1, 2 * half_cols + 1), mode=("nearest", "wrap"))
