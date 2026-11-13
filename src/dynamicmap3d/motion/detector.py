"""Temporal geometric moving-point detection by free-space (visibility) consistency.

For a target scan t and a neighbouring scan j (both registered with known poses):

  1. Transform the points of t into the sensor frame of j and project them into j's range image.
  2. If j measured a return *farther* than the point along the same ray (beyond a range-dependent
     tolerance, using the minimum over a pixel neighbourhood), the laser of j passed through the
     point's location: the space was empty at time j  ->  a *violation* vote (evidence of motion).
  3. If j's return is at about the same range, the point is consistent (static evidence).
     If j's return is closer, the point was occluded: no information.

A point is a motion candidate if it has >= ``min_votes`` violations and the fraction of
violations among informative observations is >= ``min_ratio``. Candidates are restricted to
non-ground points in the valid range, clustered (isolated responses are removed), and grown into
neighbouring points that carry some (but not majority) free-space evidence.

No labels are used anywhere in this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from dynamicmap3d.data.kitti import transform
from dynamicmap3d.motion.ground import ground_mask
from dynamicmap3d.motion.range_image import RangeImageSpec, conservative_min


@dataclass
class DetectorConfig:
    neighbor_offsets: tuple[int, ...] = (-10, -6, -3, 3, 6, 10)
    min_range: float = 3.0           # ignore returns from the ego vehicle / very near field
    max_range: float = 50.0
    tau_abs: float = 0.5             # free-space tolerance: tau = tau_abs + tau_rel * range
    tau_rel: float = 0.02
    window_half_rows: int = 1
    window_half_cols: int = 2
    min_votes: int = 2
    min_ratio: float = 0.5
    ground_cell: float = 1.0
    ground_height: float = 0.25
    cluster_radius: float = 0.5
    min_cluster_points: int = 10
    grow_radius: float = 0.5
    grow_min_ratio: float = 0.2      # grow only into points whose violation ratio >= this
    grow_iterations: int = 0          # growth disabled by default (dev window: +7x FP vs TP)
    grow_require_vote: bool = True   # True: never grow into points without any free-space violation
    spec: RangeImageSpec = field(default_factory=RangeImageSpec)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["neighbor_offsets"] = list(self.neighbor_offsets)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "DetectorConfig":
        d = dict(d)
        spec = RangeImageSpec(**d.pop("spec", {}))
        if "neighbor_offsets" in d:
            d["neighbor_offsets"] = tuple(d["neighbor_offsets"])
        return cls(spec=spec, **d)


@dataclass
class Detection:
    dynamic: np.ndarray      # final per-point decision
    candidate: np.ndarray    # raw voting decision (before clustering / growing)
    votes: np.ndarray        # number of free-space violations
    informative: np.ndarray  # number of non-occluded observations
    ground: np.ndarray
    valid: np.ndarray        # inside [min_range, max_range]
    neighbors_used: list[int]


class TemporalMotionDetector:
    def __init__(self, scans: dict[int, np.ndarray], poses: dict[int, np.ndarray],
                 config: DetectorConfig | None = None):
        """scans: frame -> (N, >=3) points in the sensor frame; poses: frame -> 4x4 sensor-to-world."""
        self.cfg = config or DetectorConfig()
        self.scans, self.poses = scans, poses
        self._rimg: dict[int, np.ndarray] = {}

    def _range_image(self, j: int) -> np.ndarray:
        if j not in self._rimg:
            c = self.cfg
            img = c.spec.min_range_image(self.scans[j][:, :3], min_range=c.min_range)
            self._rimg[j] = conservative_min(img, c.window_half_rows, c.window_half_cols)
        return self._rimg[j]

    def vote(self, t: int):
        c = self.cfg
        xyz = self.scans[t][:, :3].astype(np.float64)
        r = np.linalg.norm(xyz, axis=1)
        valid = (r >= c.min_range) & (r <= c.max_range)
        votes = np.zeros(len(xyz), np.int32)
        informative = np.zeros(len(xyz), np.int32)
        used = []
        for o in c.neighbor_offsets:
            j = t + o
            if j not in self.scans or j not in self.poses:
                continue
            used.append(j)
            T_jt = np.linalg.inv(self.poses[j]) @ self.poses[t]
            row, col, rp, in_fov = c.spec.project(transform(xyz, T_jt))
            R = self._range_image(j)[row, col]
            obs = valid & in_fov & (rp >= c.min_range) & (rp <= c.max_range) & np.isfinite(R)
            tau = c.tau_abs + c.tau_rel * rp
            votes += obs & (R > rp + tau)
            informative += obs & (R >= rp - tau)
        return votes, informative, valid, used

    def detect(self, t: int) -> Detection:
        c = self.cfg
        xyz = self.scans[t][:, :3].astype(np.float64)
        votes, informative, valid, used = self.vote(t)
        ratio = votes / np.maximum(informative, 1)
        ground = ground_mask(xyz, cell=c.ground_cell, height=c.ground_height)
        eligible = valid & ~ground
        candidate = eligible & (votes >= c.min_votes) & (ratio >= c.min_ratio)

        dynamic = np.zeros(len(xyz), bool)
        cand_idx = np.flatnonzero(candidate)
        if len(cand_idx) >= c.min_cluster_points:
            pairs = cKDTree(xyz[cand_idx]).query_pairs(c.cluster_radius, output_type="ndarray")
            n = len(cand_idx)
            g = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
            _, lab = connected_components(g, directed=False)
            sizes = np.bincount(lab)
            dynamic[cand_idx[sizes[lab] >= c.min_cluster_points]] = True

            if c.grow_require_vote:
                growable = eligible & (votes >= 1) & (ratio >= c.grow_min_ratio)
            else:
                growable = eligible & ((informative == 0) | (ratio >= c.grow_min_ratio))
            grow_idx = np.flatnonzero(growable)
            if len(grow_idx) and dynamic.any():
                tree = cKDTree(xyz[grow_idx])
                frontier = np.flatnonzero(dynamic)
                for _ in range(c.grow_iterations):
                    hits = tree.query_ball_point(xyz[frontier], c.grow_radius, return_sorted=False)
                    new = np.unique(np.concatenate([np.asarray(h, dtype=np.int64) for h in hits])) \
                        if len(hits) else np.array([], np.int64)
                    new = grow_idx[new] if len(new) else new
                    new = new[~dynamic[new]] if len(new) else new
                    if not len(new):
                        break
                    dynamic[new] = True
                    frontier = new
        return Detection(dynamic, candidate, votes, informative, ground, valid, used)
