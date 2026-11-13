"""Phase 2: evaluate the FROZEN Phase 1 detector on one window (run once per window, fresh process).

Order of operations enforces ground-truth isolation:
  1. verify frozen checksums  2. load scans + poses ONLY  3. run detection on every frame
  4. only then read label files and score.
Neighbours are restricted to the window's own frames (identical boundary handling everywhere).
The detector is offline/non-causal: it uses scans up to +10 frames (1.0 s) in the future.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from dynamicmap3d.data.kitti import transform
from dynamicmap3d.data.semantickitti import is_ignored, is_moving, read_labels
from dynamicmap3d.data.splits import get_window
from dynamicmap3d.data.window import load_window
from dynamicmap3d.evaluation.mapping import MapAccumulator, voxel_keys
from dynamicmap3d.evaluation.metrics import Confusion
from dynamicmap3d.evaluation.objects import (PARKED_VEHICLES, STATIC_CLASSES, depth_boundary_mask,
                                             object_records, summarize_objects)
from dynamicmap3d.motion.detector import DetectorConfig, TemporalMotionDetector
from dynamicmap3d.resources import peak_rss_bytes

METHODS = ("raw_unfiltered", "temporal_detector", "votes_only", "oracle_labels")
VOXEL = 0.2


def verify_frozen(path: Path) -> dict:
    frozen = json.loads(path.read_text())
    bad = {f: h for f, h in frozen["sha256"].items()
           if hashlib.sha256(Path(f).read_bytes()).hexdigest() != h}
    if bad:
        raise SystemExit(f"FROZEN CHECKSUM MISMATCH, refusing to evaluate: {sorted(bad)}")
    return frozen


def pred_hash(preds: dict[int, np.ndarray]) -> str:
    h = hashlib.sha256()
    for f in sorted(preds):
        h.update(np.packbits(preds[f]).tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--root", type=Path, default=Path("data/semantickitti/dataset"))
    ap.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    ap.add_argument("--frozen", type=Path, default=Path("configs/FROZEN_phase1.json"))
    ap.add_argument("--out", type=Path, default=Path("results/phase2/windows"))
    a = ap.parse_args()
    frozen = verify_frozen(a.frozen)
    w = get_window(a.window)
    cfg_all = yaml.safe_load(a.config.read_text())
    cfg = DetectorConfig.from_dict(cfg_all["detector"])

    # ---- 2. scans + poses only ---------------------------------------------------------------
    t0 = time.perf_counter()
    W = load_window(a.root, w.sequence, w.first, w.last, cfg_all["pose_source"], with_labels=False)
    load_s = time.perf_counter() - t0
    mem_after_load = peak_rss_bytes()

    # ---- 3. detection (labels not loaded yet) -----------------------------------------------
    det = TemporalMotionDetector(W.scans, W.poses, cfg)
    final, cand, valid, nbrs, rt = {}, {}, {}, {}, {}
    for f in W.frames:
        t = time.perf_counter()
        d = det.detect(f)
        rt[f] = time.perf_counter() - t
        final[f], cand[f], valid[f], nbrs[f] = d.dynamic, d.candidate, d.valid, d.neighbors_used
    mem_after_detect = peak_rss_bytes()
    mid = W.frames[len(W.frames) // 2]
    d2 = TemporalMotionDetector(W.scans, W.poses, cfg).detect(mid)   # fresh instance
    deterministic = bool(np.array_equal(d2.dynamic, final[mid]) and np.array_equal(d2.candidate, cand[mid]))
    hashes = {"temporal_detector": pred_hash(final), "votes_only": pred_hash(cand)}

    # ---- 4. labels, read only now ------------------------------------------------------------
    sem, inst = {}, {}
    sdir = a.root / "sequences" / w.sequence
    for f in W.frames:
        sem[f], inst[f] = read_labels(sdir / "labels" / f"{f:06d}.label")
        if len(sem[f]) != len(W.scans[f]) or len(final[f]) != len(sem[f]):
            raise SystemExit(f"point/label/prediction length mismatch at frame {f}")
    gt = {f: is_moving(sem[f]) for f in W.frames}
    keep = {f: ~is_ignored(sem[f]) for f in W.frames}
    preds = {"raw_unfiltered": {f: np.zeros(len(sem[f]), bool) for f in W.frames},
             "temporal_detector": final, "votes_only": cand, "oracle_labels": gt}
    n_off = len(cfg.neighbor_offsets)
    interior = [f for f in W.frames if len(nbrs[f]) == n_off]

    R = {"window": w.key, "role": w.role, "sequence": w.sequence, "frames": [w.first, w.last],
         "n_frames": len(W.frames), "n_interior_frames": len(interior),
         "interior_frame_range": [interior[0], interior[-1]] if interior else None,
         "points_total": int(sum(len(s) for s in sem.values())),
         "points_ignored_label": int(sum((~k).sum() for k in keep.values())),
         "points_evaluated": int(sum(k.sum() for k in keep.values())),
         "points_in_detector_range": int(sum((valid[f] & keep[f]).sum() for f in W.frames)),
         "frozen_config_sha256": frozen["sha256"]["configs/phase1.yaml"],
         "causality": "offline, non-causal: uses scans up to +10 frames (1.0 s) in the future",
         "boundary_handling": "neighbours restricted to window frames; edge frames use fewer neighbours",
         "neighbors_used_histogram": {str(k): int(v) for k, v in zip(*np.unique([len(v) for v in nbrs.values()], return_counts=True))},
         "deterministic_rerun_identical": deterministic, "prediction_sha256": hashes}

    # point-level metrics
    pm = {}
    for m in METHODS:
        conf = {"all_frames": Confusion(), "interior_frames": Confusion(), "in_range_all_frames": Confusion()}
        for f in W.frames:
            conf["all_frames"].add(preds[m][f], gt[f], keep[f])
            conf["in_range_all_frames"].add(preds[m][f], gt[f], keep[f] & valid[f])
            if f in interior:
                conf["interior_frames"].add(preds[m][f], gt[f], keep[f])
        pm[m] = {k: c.summary() for k, c in conf.items()}
    R["point_metrics"] = pm

    # per-class false removal + occlusion boundaries (detector variants only)
    spec = cfg.spec
    bnd = {f: depth_boundary_mask(W.scans[f][:, :3], spec) for f in W.frames}   # label-free
    classes = {**STATIC_CLASSES, **{k: "parked-" + v for k, v in PARKED_VEHICLES.items()}}
    R["class_false_removal"] = {}
    R["occlusion_boundaries"] = {}
    for m in ("temporal_detector", "votes_only"):
        cr = {}
        for cid, name in classes.items():
            tot = sum(int((sem[f] == cid).sum()) for f in W.frames)
            if tot:
                fl = sum(int((preds[m][f] & (sem[f] == cid)).sum()) for f in W.frames)
                cr[name] = {"points": tot, "removed": fl, "false_removal_rate": fl / tot}
        R["class_false_removal"][m] = cr
        ob = {}
        for name, sel in (("static_on_depth_boundary", True), ("static_off_boundary", False)):
            tot = fl = 0
            for f in W.frames:
                s = keep[f] & ~gt[f] & (bnd[f] == sel)
                tot += int(s.sum()); fl += int((preds[m][f] & s).sum())
            ob[name] = {"points": tot, "removed": fl, "false_removal_rate": fl / tot if tot else float("nan")}
        tot = sum(int((gt[f] & bnd[f]).sum()) for f in W.frames)
        hit = sum(int((gt[f] & bnd[f] & preds[m][f]).sum()) for f in W.frames)
        ob["moving_on_depth_boundary"] = {"points": tot, "recall": hit / tot if tot else float("nan")}
        R["occlusion_boundaries"][m] = ob

    # object level
    xyz = {f: W.scans[f][:, :3] for f in W.frames}
    R["objects"] = {}
    obj_rows = []
    for m in ("temporal_detector", "votes_only"):
        recs = object_records(W.frames, sem, inst, xyz, W.poses, preds[m], W.times, valid)
        R["objects"][m] = summarize_objects(recs)
        if m == "temporal_detector":
            obj_rows = recs

    # mapping: same range mask (detector range, label-free), poses and voxel size for all methods
    keys = {f: voxel_keys(transform(xyz[f].astype(np.float64), W.poses[f]), VOXEL) for f in W.frames}
    maps, map_time = {}, {}
    for m in METHODS:
        acc = MapAccumulator(VOXEL)
        t = time.perf_counter()
        for f in W.frames:
            acc.add(keys[f], valid[f] & ~preds[m][f], gt[f] & valid[f], keep[f] & ~gt[f] & valid[f])
        map_time[m] = time.perf_counter() - t
        maps[m] = acc
    ref = maps["raw_unfiltered"].static_vox.size
    R["mapping"] = {m: maps[m].summary(ref) | {"map_build_s": map_time[m]} for m in METHODS}
    R["mapping_note"] = ("points 3-50 m from the sensor, 0.2 m voxels, SemanticKITTI poses; methods differ "
                         "only in removed points. Fewer points is not by itself better map quality.")

    r = np.array([rt[f] for f in W.frames])
    R["runtime"] = {"detect_s_per_frame_mean": float(r.mean()), "detect_s_per_frame_median": float(np.median(r)),
                    "detect_s_per_frame_p95": float(np.percentile(r, 95)), "detect_s_per_frame_max": float(r.max()),
                    "window_load_s": load_s}
    R["memory"] = {"peak_rss_after_load_MB": mem_after_load / 1e6, "peak_rss_after_detection_MB": mem_after_detect / 1e6,
                   "peak_rss_end_MB": peak_rss_bytes() / 1e6,
                   "scans_in_memory_MB": sum(s.nbytes for s in W.scans.values()) / 1e6}

    out = a.out / w.key
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(R, indent=2))
    with open(out / "per_frame.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame", "n_points", "gt_moving", "pred_moving", "tp", "neighbors", "detect_s"])
        for f in W.frames:
            wr.writerow([f, len(sem[f]), int(gt[f].sum()), int(final[f].sum()), int((final[f] & gt[f]).sum()),
                         len(nbrs[f]), round(rt[f], 4)])
    with open(out / "objects.csv", "w", newline="") as fh:
        if obj_rows:
            wr = csv.DictWriter(fh, fieldnames=list(obj_rows[0]))
            wr.writeheader(); wr.writerows(obj_rows)
    with open(out / "map_growth.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["frame"] + list(METHODS))
        for i, f in enumerate(W.frames):
            wr.writerow([f] + [maps[m].growth[i] for m in METHODS])
    s = pm["temporal_detector"]["all_frames"]
    print(f"{w.key} seq {w.sequence} {w.first}-{w.last}: IoU {s['iou_moving']:.3f} P {s['precision']:.3f} "
          f"R {s['recall']:.3f} static-FP {100 * s['static_fp_rate']:.3f}% | det {r.mean() * 1000:.0f} ms/frame "
          f"| peak {R['memory']['peak_rss_end_MB']:.0f} MB | deterministic {deterministic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
