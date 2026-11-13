"""Analytic ray-cast LiDAR simulator for tests: ground plane + axis-aligned boxes."""

from __future__ import annotations

import numpy as np


def beam_directions(n_rows=64, n_cols=1024, up_deg=2.0, down_deg=-24.0):
    el = np.radians(np.linspace(up_deg, down_deg, n_rows))
    az = np.linspace(-np.pi, np.pi, n_cols, endpoint=False)
    E, A = np.meshgrid(el, az, indexing="ij")
    return np.stack([np.cos(E) * np.cos(A), np.cos(E) * np.sin(A), np.sin(E)], -1).reshape(-1, 3)


def raycast(origin, dirs, boxes, ground_z=-1.73, max_range=80.0):
    """Return (points_world, hit_id): hit_id -1 = ground, k = boxes[k]; misses dropped."""
    best = np.full(len(dirs), np.inf)
    hit = np.full(len(dirs), -2)
    with np.errstate(divide="ignore", invalid="ignore"):
        tg = (ground_z - origin[2]) / dirs[:, 2]
    g = (dirs[:, 2] < 0) & (tg > 0)
    best[g], hit[g] = tg[g], -1
    for k, (lo, hi) in enumerate(boxes):
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (np.asarray(lo) - origin) / dirs
            t2 = (np.asarray(hi) - origin) / dirs
        tmin = np.nanmax(np.minimum(t1, t2), axis=1)
        tmax = np.nanmin(np.maximum(t1, t2), axis=1)
        h = (tmax >= np.maximum(tmin, 0)) & (tmin > 0) & (tmin < best)
        best[h], hit[h] = tmin[h], k
    ok = best < max_range
    return origin + dirs[ok] * best[ok, None], hit[ok]


def translation(p):
    T = np.eye(4)
    T[:3, 3] = p
    return T


def make_sequence(n_frames=21, ego_speed=1.0, mover_speed=1.0, seed=0):
    """Ego drives along +x; a 'car' box crosses along +y; a parked box and a wall are static.

    Returns scans (sensor frame, with x,y,z,0), poses (sensor->world), and per-frame hit ids.
    Box ids: 0 = wall, 1 = parked car, 2 = moving car.
    """
    rng = np.random.default_rng(seed)
    dirs = beam_directions()
    scans, poses, ids = {}, {}, {}
    for f in range(n_frames):
        origin = np.array([f * ego_speed, 0.0, 0.0])
        my = -12.0 + mover_speed * f
        boxes = [((35.0, -40.0, -1.73), (36.0, 40.0, 4.0)),          # wall
                 ((14.0, 5.0, -1.73), (18.5, 6.8, -0.2)),             # parked car (left)
                 ((22.0, my - 0.9, -1.73), (26.5, my + 0.9, -0.2))]   # moving car crossing
        pts, hid = raycast(origin, dirs, boxes)
        pts_s = pts - origin + rng.normal(0, 0.01, pts.shape)
        scans[f] = np.c_[pts_s, np.zeros(len(pts_s))].astype(np.float32)
        poses[f] = translation(origin)
        ids[f] = hid
    return scans, poses, ids
