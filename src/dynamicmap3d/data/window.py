"""Load a contiguous SemanticKITTI window (scans, labels, Velodyne poses)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dynamicmap3d.data.kitti import cam_to_velo_poses, read_calib, read_poses, read_scan, read_times
from dynamicmap3d.data.semantickitti import read_labels

POSE_SOURCES = ("semantickitti", "kitti_gt")


@dataclass
class Window:
    sequence: str
    frames: list[int]
    scans: dict[int, np.ndarray]
    sem: dict[int, np.ndarray]
    inst: dict[int, np.ndarray]
    poses: dict[int, np.ndarray]   # 4x4 Velodyne -> world (world = Velodyne frame of scan 0)
    Tr: np.ndarray
    times: np.ndarray


def pose_file(root: Path, seq: str, source: str) -> Path:
    if source == "semantickitti":
        return root / "sequences" / seq / "poses.txt"
    if source == "kitti_gt":
        return root / "poses" / f"{seq}.txt"
    raise ValueError(f"pose source must be one of {POSE_SOURCES}")


def load_window(root: str | Path, seq: str, first: int, last: int,
                pose_source: str = "semantickitti", with_labels: bool = True) -> Window:
    root = Path(root)
    sdir = root / "sequences" / seq
    Tr = read_calib(sdir / "calib.txt")["Tr"]
    poses_all = cam_to_velo_poses(read_poses(pose_file(root, seq, pose_source)), Tr)
    frames = list(range(first, last + 1))
    scans, sem, inst = {}, {}, {}
    for i in frames:
        scans[i] = read_scan(sdir / "velodyne" / f"{i:06d}.bin")
        if with_labels:
            sem[i], inst[i] = read_labels(sdir / "labels" / f"{i:06d}.label")
    return Window(seq, frames, scans, sem, inst, {i: poses_all[i] for i in frames}, Tr,
                  read_times(sdir / "times.txt"))
