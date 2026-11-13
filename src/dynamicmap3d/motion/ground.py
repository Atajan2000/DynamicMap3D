"""Simple label-free ground segmentation: height above the local minimum of a 2-D grid."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import minimum_filter


def ground_mask(xyz: np.ndarray, cell: float = 1.0, height: float = 0.25, neighbourhood: int = 1,
                extent: float = 100.0) -> np.ndarray:
    """True for points within ``height`` metres of the lowest point in the surrounding cells.

    The neighbourhood minimum ((2n+1)^2 cells) lets cells fully covered by an object (e.g. the
    roof of a car) inherit the ground height from adjacent cells.
    """
    n = int(np.ceil(2 * extent / cell))
    ij = np.floor((xyz[:, :2] + extent) / cell).astype(np.int64)
    inside = np.all((ij >= 0) & (ij < n), axis=1)
    grid = np.full((n, n), np.inf)
    np.minimum.at(grid, (ij[inside, 0], ij[inside, 1]), xyz[inside, 2])
    if neighbourhood > 0:
        grid = minimum_filter(grid, size=2 * neighbourhood + 1, mode="nearest")
    out = np.zeros(len(xyz), dtype=bool)
    out[inside] = xyz[inside, 2] - grid[ij[inside, 0], ij[inside, 1]] < height
    return out
