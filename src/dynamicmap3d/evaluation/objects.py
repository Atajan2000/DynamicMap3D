"""Object- and class-level analysis using SemanticKITTI instance ids (evaluation only)."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from dynamicmap3d.data.kitti import transform
from dynamicmap3d.data.semantickitti import MOVING_CLASSES, STATIC_MOVABLE
from dynamicmap3d.motion.range_image import RangeImageSpec

PARKED_VEHICLES = {10: "car", 13: "bus", 15: "motorcycle", 18: "truck", 20: "other-vehicle"}
STATIC_CLASSES = {50: "building", 51: "fence", 52: "other-structure", 70: "vegetation", 71: "trunk",
                  72: "terrain", 80: "pole", 81: "traffic-sign", 40: "road", 44: "parking",
                  48: "sidewalk", 49: "other-ground"}
SPEED_BINS = [(0.0, 1.5, "slow <1.5 m/s"), (1.5, 5.0, "1.5-5 m/s"), (5.0, np.inf, "fast >=5 m/s")]


def depth_boundary_mask(xyz: np.ndarray, spec: RangeImageSpec, jump: float = 1.0, rel: float = 0.05) -> np.ndarray:
    """Label-free occlusion-boundary mask: the point's pixel has a 4-neighbour whose closest
    return differs by more than max(jump, rel * range)."""
    row, col, r, ok = spec.project(xyz)
    img = spec.min_range_image(xyz)
    H, W = img.shape
    out = np.zeros(len(xyz), bool)
    thr = np.maximum(jump, rel * r)
    for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
        rr, cc = np.clip(row + dr, 0, H - 1), (col + dc) % W
        nb = img[rr, cc]
        out |= ok & np.isfinite(nb) & (np.abs(nb - r) > thr)
    return out


def object_records(frames, sem, inst, xyz, poses, pred, times, valid, min_points=10):
    """One record per (frame, object) for moving objects and parked vehicles."""
    recs = []
    centroids = defaultdict(dict)
    for f in frames:
        s, i = sem[f], inst[f]
        key = (s.astype(np.uint32) << 16) | i
        interesting = np.isin(s, list(MOVING_CLASSES) + list(PARKED_VEHICLES))
        for k in np.unique(key[interesting]):
            m = key == k
            n = int(m.sum())
            if n < min_points:
                continue
            cls, iid = int(k >> 16), int(k & 0xFFFF)
            c_world = transform(xyz[f][m].mean(0, keepdims=True), poses[f])[0]
            centroids[(cls, iid)][f] = c_world
            recs.append({"frame": f, "class_id": cls,
                         "class": MOVING_CLASSES.get(cls) or "parked-" + PARKED_VEHICLES[cls],
                         "moving": cls in MOVING_CLASSES, "instance": iid, "n_points": n,
                         "flagged_fraction": float(pred[f][m].mean()),
                         "median_range_m": float(np.median(np.linalg.norm(xyz[f][m], axis=1))),
                         "in_range_fraction": float(valid[f][m].mean())})
    speed = {}
    for obj, cs in centroids.items():
        fs = sorted(cs)
        v = [np.linalg.norm(cs[b] - cs[a]) / (times[b] - times[a])
             for a in fs for b in fs if 3 <= b - a <= 10]
        speed[obj] = float(np.median(v)) if v else float("nan")
    for r in recs:
        r["instance_speed_mps"] = speed[(r["class_id"], r["instance"])]
    return recs


def summarize_objects(recs, detect_threshold=0.5) -> dict:
    """Coverage of moving objects (flagged >= threshold) and false removal of parked vehicles."""
    out: dict = {}
    mov = [r for r in recs if r["moving"] and r["in_range_fraction"] >= 0.5]
    by_cls = defaultdict(list)
    for r in mov:
        by_cls[r["class"]].append(r)
    def cover(rs):
        if not rs:
            return {"object_frames": 0}
        det = np.array([r["flagged_fraction"] >= detect_threshold for r in rs])
        inst = defaultdict(list)
        for r, d in zip(rs, det):
            inst[(r["class_id"], r["instance"])].append(d)
        return {"object_frames": len(rs), "object_frames_detected": int(det.sum()),
                "object_frame_coverage": float(det.mean()),
                "mean_point_recall": float(np.mean([r["flagged_fraction"] for r in rs])),
                "instances": len(inst),
                "instances_detected_in_majority_of_frames": int(sum(np.mean(v) >= 0.5 for v in inst.values()))}
    out["moving_all"] = cover(mov)
    out["moving_by_class"] = {c: cover(rs) for c, rs in sorted(by_cls.items())}
    out["moving_by_speed"] = {lab: cover([r for r in mov if lo <= r["instance_speed_mps"] < hi])
                              for lo, hi, lab in SPEED_BINS}
    out["moving_unknown_speed_object_frames"] = sum(1 for r in mov if not np.isfinite(r["instance_speed_mps"]))
    parked = [r for r in recs if not r["moving"]]
    if parked:
        ff = np.array([r["flagged_fraction"] for r in parked])
        out["parked_vehicles"] = {
            "object_frames": len(parked),
            "instances": len({(r["class_id"], r["instance"]) for r in parked}),
            "mean_flagged_fraction": float(ff.mean()),
            "object_frames_falsely_removed_ge50pct": int((ff >= 0.5).sum()),
            "object_frames_touched_ge10pct": int((ff >= 0.1).sum()),
            "object_frames_untouched": int((ff == 0).sum())}
    return out
