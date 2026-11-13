"""Real-data smoke test of the temporal motion detector on the DEVELOPMENT window only.

Labels are used exclusively for scoring, after detection. Reports point-level MOS metrics
(all points and points within the detector's range), object-level recall, false positives by
semantic class, the raw-voting ablation (no clustering / growing), and runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import yaml

from dynamicmap3d.data.semantickitti import GROUND, IGNORE_CLASSES, MOVING_CLASSES, STATIC_MOVABLE, STRUCTURE
from dynamicmap3d.data.window import load_window
from dynamicmap3d.evaluation.metrics import Evaluator
from dynamicmap3d.motion.detector import DetectorConfig, TemporalMotionDetector

NAMES = {**MOVING_CLASSES, **IGNORE_CLASSES, **STATIC_MOVABLE, **STRUCTURE, **GROUND}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=Path("configs/splits.yaml"))
    ap.add_argument("--config", type=Path, default=Path("configs/phase1.yaml"))
    ap.add_argument("--root", type=Path, default=Path("data/semantickitti/dataset"))
    ap.add_argument("--out", type=Path, default=Path("results/phase1/dev_smoke"))
    ap.add_argument("--frames", type=str, default=None, help="optional sub-range 'a-b' inside the window")
    a = ap.parse_args()

    dev = yaml.safe_load(a.splits.read_text())["development"]
    assert dev["role"].startswith("development"), "smoke test runs on the development window only"
    seq, first, last = str(dev["sequence"]).zfill(2), dev["first_frame"], dev["last_frame"]
    cfg_all = yaml.safe_load(a.config.read_text())
    cfg = DetectorConfig.from_dict(cfg_all["detector"])

    W = load_window(a.root, seq, first, last, cfg_all["pose_source"])
    det = TemporalMotionDetector(W.scans, W.poses, cfg)
    targets = W.frames
    if a.frames:
        lo, hi = map(int, a.frames.split("-"))
        targets = [f for f in targets if lo <= f <= hi]
    reach = max(abs(o) for o in cfg.neighbor_offsets)
    interior = {f for f in targets if f - reach >= first and f + reach <= last}

    ev = {k: Evaluator() for k in ("final_all", "final_interior", "votes_only_all")}
    rows, times = [], []
    for f in targets:
        t0 = time.perf_counter()
        d = det.detect(f)
        times.append(time.perf_counter() - t0)
        sem, inst = W.sem[f], W.inst[f]
        ev["final_all"].add(f, d.dynamic, sem, inst, d.valid)
        ev["votes_only_all"].add(f, d.candidate, sem, inst, d.valid)
        if f in interior:
            ev["final_interior"].add(f, d.dynamic, sem, inst, d.valid)
        gt = (sem >= 252) & (sem <= 259)
        rows.append({"frame": f, "n_points": len(sem), "gt_moving": int(gt.sum()),
                     "pred_moving": int(d.dynamic.sum()), "tp": int((d.dynamic & gt).sum()),
                     "neighbors": len(d.neighbors_used), "runtime_s": round(times[-1], 4)})

    out = {"sequence": seq, "window": [first, last], "frames_evaluated": len(targets),
           "interior_frames": len(interior), "pose_source": cfg_all["pose_source"],
           "config": cfg.to_dict(), "runtime_s_per_frame": {"mean": float(np.mean(times)),
                                                           "max": float(np.max(times))}}
    for k, e in ev.items():
        out[k] = {"points_all": e.all_points.summary(), "points_in_range": e.in_range.summary(),
                  "objects": e.object_summary()}
    fp = ev["final_all"].fp_by_class
    tot = sum(fp.values()) or 1
    out["final_all"]["false_positives_by_class"] = {
        NAMES.get(c, str(c)): {"points": n, "share": round(n / tot, 4)}
        for c, n in sorted(fp.items(), key=lambda kv: -kv[1])[:12]}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "metrics.json").write_text(json.dumps(out, indent=2))
    with open(a.out / "per_frame.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    def line(name, s):
        p = s["points_all"]
        return (f"{name:16s} IoU {p['iou_moving']:.3f}  P {p['precision']:.3f}  R {p['recall']:.3f}  "
                f"static-FP-rate {100 * p['static_fp_rate']:.3f}%  | in-range IoU {s['points_in_range']['iou_moving']:.3f}")
    print(f"seq {seq} frames {targets[0]}-{targets[-1]} ({len(targets)}; interior {len(interior)})")
    for k in ev:
        print(line(k, out[k]))
    print("objects (final_all):", out["final_all"]["objects"])
    print("top FP classes:", {k: v["share"] for k, v in list(out["final_all"]["false_positives_by_class"].items())[:6]})
    print(f"runtime {out['runtime_s_per_frame']['mean'] * 1000:.0f} ms/frame mean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
