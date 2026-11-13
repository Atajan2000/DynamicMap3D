"""KITTI Odometry readers: Velodyne scans, calib.txt, poses, times.

Pose convention (KITTI odometry): ``P_i`` maps camera-0 coordinates of frame i into camera-0
coordinates of frame 0. ``Tr`` maps Velodyne -> rectified camera 0. Velodyne poses in the
Velodyne frame of scan 0 are therefore ``T_i = Tr^-1 @ P_i @ Tr``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_scan(path: str | Path) -> np.ndarray:
    """Return an (N, 4) float32 array: x, y, z, reflectance (Velodyne frame)."""
    raw = np.fromfile(path, dtype=np.float32)
    if raw.size % 4:
        raise ValueError(f"{path}: size {raw.size * 4} B is not a multiple of 16")
    return raw.reshape(-1, 4)


def scan_point_count(path: str | Path) -> int:
    size = Path(path).stat().st_size
    if size % 16:
        raise ValueError(f"{path}: size {size} B is not a multiple of 16")
    return size // 16


def _to4x4(v: np.ndarray) -> np.ndarray:
    T = np.eye(4)
    T[:3, :4] = np.asarray(v, dtype=np.float64).reshape(3, 4)
    return T


def read_calib(path: str | Path) -> dict[str, np.ndarray]:
    """Parse an odometry calib.txt. Returns P0..P3 as 3x4 and Tr as 4x4 (velo -> cam0)."""
    out: dict[str, np.ndarray] = {}
    for line in Path(path).read_text().splitlines():
        if ":" not in line:
            continue
        key, vals = line.split(":", 1)
        arr = np.array([float(x) for x in vals.split()])
        out[key.strip()] = _to4x4(arr) if key.strip() == "Tr" else arr.reshape(3, 4)
    if "Tr" not in out:
        raise ValueError(f"{path}: no Tr entry")
    return out


def parse_poses(text: str) -> np.ndarray:
    rows = [np.array(l.split(), dtype=np.float64) for l in text.strip().splitlines() if l.strip()]
    if any(r.size != 12 for r in rows):
        raise ValueError("pose rows must have 12 values")
    return np.stack([_to4x4(r) for r in rows])


def read_poses(path: str | Path) -> np.ndarray:
    """(N, 4, 4) camera-0 poses in KITTI 12-value-per-line format."""
    return parse_poses(Path(path).read_text())


def read_times(path: str | Path) -> np.ndarray:
    return np.array(Path(path).read_text().split(), dtype=np.float64)


def cam_to_velo_poses(poses_cam: np.ndarray, Tr: np.ndarray) -> np.ndarray:
    """Convert camera-0 poses to Velodyne poses (world = Velodyne frame of scan 0)."""
    return np.linalg.inv(Tr) @ poses_cam @ Tr


def transform(points_xyz: np.ndarray, T: np.ndarray) -> np.ndarray:
    return points_xyz @ T[:3, :3].T + T[:3, 3]
