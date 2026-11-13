"""Markdown panels for the viewer. Every number is read from saved results (no hard-coded metrics)."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from dynamicmap3d.viz.data import DETECTOR_CLASSES, EVAL_CLASSES, ViewData

ROLE_NOTE = {
    "eval2": "**Example A: successful fast-motion filtering.** Held-out window; moving cars and a truck at 8-11 m/s.",
    "eval0": ("**Example B: near-stationary objects.** Held-out window. The largest annotated *moving* cars move "
              "< 0.5 m/s during these 100 frames; a ±1 s free-space test cannot separate them from parked cars, "
              "so they stay in the filtered map (blue in the evaluation layers)."),
    "eval1": "**Example C: slow pedestrians and cyclists.** Held-out window with mostly 0.2-1.6 m/s movers.",
    "development": "**Development window** (parameters were tuned here, so its numbers are optimistic).",
}


def pct(x: float, d: int = 1) -> str:
    return "n/a" if x != x else f"{100 * x:.{d}f}%"


def frame_markdown(d: ViewData, diag: dict) -> str:
    w = d.window
    lines = [f"### Frame {diag['frame']}  ·  seq {w.sequence}, frames {d.frames[0]}-{d.frames[-1]}",
             "**Per-frame values (this scan):**", "",
             "| | |", "|---|---|",
             f"| points in scan | {diag['points']:,} |",
             f"| inside detector range (3-50 m) | {diag['points_in_range']:,} |",
             f"| removed (dynamic) | {diag['removed']:,} |",
             f"| retained | {diag['retained']:,} |",
             f"| dynamic fraction | {100 * diag['dynamic_fraction']:.2f}% |",
             f"| voted but rejected by cluster filter | {diag['rejected_candidates']:,} |",
             f"| neighbour scans used | {diag['neighbors_used']} of 6{'  (window edge)' if diag['neighbors_used'] < 6 else ''} |",
             f"| detection time (measured in Phase 2 run) | {diag['detect_ms_phase2']:.0f} ms |"]
    if "gt_moving" in diag:
        lines += ["", "**Evaluation only (ground truth, this scan):**", "", "| | |", "|---|---|",
                  f"| annotated moving points | {diag['gt_moving']:,} |",
                  f"| moving removed / missed | {diag['tp']:,} / {diag['fn']:,} |",
                  f"| static points removed | {diag['fp']:,} |"]
    return "\n".join(lines)


def window_markdown(d: ViewData) -> str:
    m, S, k = d.metrics, d.summary, d.window.key
    pm = m["point_metrics"]["temporal_detector"]["all_frames"]
    mp = m["mapping"]
    pv = m["objects"]["temporal_detector"].get("parked_vehicles", {})
    cr = m["class_false_removal"]["temporal_detector"]
    ob = m["occlusion_boundaries"]["temporal_detector"]
    shown = (d.frames[0], d.frames[-1])
    full = tuple(m["frames"])
    lines = [ROLE_NOTE.get(k, ""), "",
             f"#### Window-level results: seq {m['sequence']} frames {full[0]}-{full[1]} (whole window, NOT per frame)",
             f"Custom held-out SemanticKITTI window experiment, not an official benchmark score."
             if k != "development" else "Development window (tuned).", ""]
    if shown != full:
        lines += [f"*Displayed sub-range {shown[0]}-{shown[1]}; metrics below cover the whole window.*", ""]
    lines += ["| metric | value |", "|---|---|",
              f"| moving IoU | {pm['iou_moving']:.3f} |",
              f"| precision / recall | {pm['precision']:.3f} / {pm['recall']:.3f} |",
              f"| static points falsely removed | {pct(pm['static_fp_rate'], 3)} |",
              f"| annotated moving points | {pm['n_moving_gt']:,} |",
              f"| static-point retention (map, 3-50 m) | {pct(mp['temporal_detector']['static_point_retention'], 2)} |",
              f"| static voxel coverage retained | {pct(mp['temporal_detector']['static_coverage'], 2)} |",
              f"| voxels with moving points: raw → filtered | {pct(mp['raw_unfiltered']['contaminated_voxel_fraction'], 2)} → "
              f"{pct(mp['temporal_detector']['contaminated_voxel_fraction'], 2)} |"]
    if pv:
        lines.append(f"| parked-vehicle object-frames ≥50% removed | {pv['object_frames_falsely_removed_ge50pct']} of {pv['object_frames']} |")
    for name in ("pole", "traffic-sign", "parked-car", "building", "road"):
        if name in cr:
            lines.append(f"| false removal: {name} | {pct(cr[name]['false_removal_rate'], 3)} |")
    lines.append(f"| false removal on / off depth boundaries | {pct(ob['static_on_depth_boundary']['false_removal_rate'], 3)} / "
                 f"{pct(ob['static_off_boundary']['false_removal_rate'], 3)} |")
    lines += ["", "#### All windows (point-level, temporal detector)", "",
              "| window | moving IoU | precision | recall |", "|---|---|---|---|"]
    names = {"development": "development seq 09 1100-1199 (tuned)", "eval0": "held-out seq 08 0-99",
             "eval1": "held-out seq 08 1850-1949", "eval2": "held-out seq 07 600-699",
             "heldout_pooled": "held-out pooled (summed counts)"}
    for key, label in names.items():
        s = S["point_metrics"]["temporal_detector"][key]["all_frames"]
        lines.append(f"| {label} | {s['iou_moving']:.3f} | {s['precision']:.3f} | {s['recall']:.3f} |")
    lines += ["", "#### Moving objects in this window (evaluation only)", "",
              "| object | measured speed | mean removed | frames |", "|---|---|---|---|"]
    inst = defaultdict(list)
    for r in d.objects:
        if r["moving"] == "True":
            inst[(r["class"], r["instance"])].append(r)
    rows = sorted(inst.items(), key=lambda kv: -sum(int(r["n_points"]) for r in kv[1]))
    for (cls, iid), rs in rows[:10]:
        sp = float(rs[0]["instance_speed_mps"])
        fl = np.mean([float(r["flagged_fraction"]) for r in rs])
        lines.append(f"| {cls} #{iid} | {'n/a' if sp != sp else f'{sp:.1f} m/s'} | {100 * fl:.0f}% | {len(rs)} |")
    return "\n".join(lines)


def legend_markdown() -> str:
    det = "\n".join(f"| detector | {lab} |" for lab, _ in DETECTOR_CLASSES.values())
    ev = "\n".join(f"| evaluation | {lab} |" for lab, _ in EVAL_CLASSES.values())
    return f"""#### Method
Geometry-based, pose-assisted free-space test over neighbour scans at ±3, ±6, ±10 frames.
**Offline and non-causal:** it uses scans up to 1.0 s in the future. Semantic labels are never used for inference.
Not online detection, not SLAM (poses are given), and not perfect moving-object removal.

#### Views
* **A Raw map**: all points 3-50 m, 0.2 m voxels, grey by height.
* **B Motion-aware static map**: same voxels minus detector-removed points; removed points in orange.
* **C Dynamic evidence**: current scan coloured by detector state, plus the detector's removals from the previous 10 frames (faint orange).
* **D Evaluation overlay**: current scan coloured by comparison with ground truth. Evaluation only: computed after detection and never fed back.
* Tabs **Failure analysis** and **Static preservation** accumulate evaluation layers over the window.

| layer | meaning (colour shown in the 3D views) |
|---|---|
{det}
{ev}
| maps | grey brightness = height (same scale in A and B) |
| evaluation maps | blue = annotated moving geometry that stays in the filtered map · white = static removed · aqua = parked vehicles kept |

Map voxels appear at the frame that first observed them. Rerun 0.41 does not link cameras between
views; all 3D views start from the same fixed viewpoint. Frame ranges given as `a:b` exclude `b`."""
