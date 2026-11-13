"""Phase 2 driver: evaluate each held-out window ONCE in a fresh process, then aggregate.

Aggregation:
  pooled point metrics  = metrics recomputed from SUMMED confusion counts over the three held-out
                          windows (micro-average; every evaluated point weighs equally)
  macro mean            = unweighted mean of per-window IoU/F1, reported only as a secondary number
  pooled objects        = summed object-frame counts; pooled map counts = summed voxel/point counts
                          (windows are spatially disjoint, so voxel counts add)
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from dynamicmap3d.evaluation.metrics import Confusion

EVAL = ["eval0", "eval1", "eval2"]
METHODS = ("raw_unfiltered", "temporal_detector", "votes_only", "oracle_labels")
OUT = Path("results/phase2")


def run(key: str, force: bool) -> dict:
    p = OUT / "windows" / key / "metrics.json"
    if force or not p.exists():
        subprocess.run([sys.executable, "-I", "scripts/evaluate_window.py", "--window", key], check=True)
    return json.loads(p.read_text())


def pooled_conf(ws, method, scope="all_frames") -> Confusion:
    c = Confusion()
    for w in ws:
        s = w["point_metrics"][method][scope]
        c = c + Confusion(s["tp"], s["fp"], s["fn"], s["tn"])
    return c


def pool_cover(items):
    of = sum(i.get("object_frames", 0) for i in items)
    od = sum(i.get("object_frames_detected", 0) for i in items)
    return {"object_frames": of, "object_frames_detected": od,
            "object_frame_coverage": od / of if of else float("nan"),
            "instances": sum(i.get("instances", 0) for i in items),
            "instances_detected_in_majority_of_frames": sum(i.get("instances_detected_in_majority_of_frames", 0) for i in items)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dev = json.loads((OUT / "windows" / "development" / "metrics.json").read_text())
    ws = [run(k, a.force) for k in EVAL]
    S: dict = {"aggregation": __doc__.split("Aggregation:")[1].strip(),
               "frozen_config_sha256": {w["window"]: w["frozen_config_sha256"] for w in [dev] + ws},
               "deterministic": {w["window"]: w["deterministic_rerun_identical"] for w in [dev] + ws}}
    assert len(set(S["frozen_config_sha256"].values())) == 1, "windows evaluated with different configs"

    S["windows"] = {w["window"]: {"role": w["role"], "sequence": w["sequence"], "frames": w["frames"],
                                  "points_total": w["points_total"], "points_evaluated": w["points_evaluated"],
                                  "points_ignored_label": w["points_ignored_label"],
                                  "moving_points": w["point_metrics"]["oracle_labels"]["all_frames"]["n_moving_gt"],
                                  "interior_frames": w["n_interior_frames"]} for w in [dev] + ws}
    S["point_metrics"] = {}
    for m in METHODS:
        S["point_metrics"][m] = {w["window"]: w["point_metrics"][m] for w in [dev] + ws}
        S["point_metrics"][m]["heldout_pooled"] = {sc: pooled_conf(ws, m, sc).summary()
                                                   for sc in ("all_frames", "interior_frames", "in_range_all_frames")}
        S["point_metrics"][m]["heldout_macro_mean"] = {
            k: float(np.mean([w["point_metrics"][m]["all_frames"][k] for w in ws]))
            for k in ("iou_moving", "iou_static", "f1", "static_fp_rate", "recall")}

    S["objects"] = {}
    for m in ("temporal_detector", "votes_only"):
        o = {w["window"]: w["objects"][m] for w in [dev] + ws}
        classes = sorted({c for w in ws for c in w["objects"][m]["moving_by_class"]})
        speeds = list(ws[0]["objects"][m]["moving_by_speed"])
        o["heldout_pooled"] = {
            "moving_all": pool_cover([w["objects"][m]["moving_all"] for w in ws]),
            "moving_by_class": {c: pool_cover([w["objects"][m]["moving_by_class"].get(c, {}) for w in ws]) for c in classes},
            "moving_by_speed": {s: pool_cover([w["objects"][m]["moving_by_speed"][s] for w in ws]) for s in speeds},
            "parked_vehicles": {k: sum(w["objects"][m]["parked_vehicles"][k] for w in ws)
                                for k in ("object_frames", "instances", "object_frames_falsely_removed_ge50pct",
                                          "object_frames_touched_ge10pct", "object_frames_untouched")}}
        S["objects"][m] = o

    S["class_false_removal_heldout"] = {}
    for m in ("temporal_detector", "votes_only"):
        names = sorted({n for w in ws for n in w["class_false_removal"][m]})
        S["class_false_removal_heldout"][m] = {}
        for n in names:
            pts = sum(w["class_false_removal"][m].get(n, {}).get("points", 0) for w in ws)
            rem = sum(w["class_false_removal"][m].get(n, {}).get("removed", 0) for w in ws)
            S["class_false_removal_heldout"][m][n] = {"points": pts, "removed": rem, "false_removal_rate": rem / pts}
    S["occlusion_boundaries_heldout"] = {}
    for m in ("temporal_detector", "votes_only"):
        d = {}
        for k in ("static_on_depth_boundary", "static_off_boundary"):
            pts = sum(w["occlusion_boundaries"][m][k]["points"] for w in ws)
            rem = sum(w["occlusion_boundaries"][m][k]["removed"] for w in ws)
            d[k] = {"points": pts, "removed": rem, "false_removal_rate": rem / pts}
        S["occlusion_boundaries_heldout"][m] = d

    S["mapping"] = {}
    for m in METHODS:
        per = {w["window"]: w["mapping"][m] for w in [dev] + ws}
        tot = lambda k: sum(w["mapping"][m][k] for w in ws)
        ref = sum(w["mapping"]["raw_unfiltered"]["static_voxels"] for w in ws)
        per["heldout_pooled"] = {
            "points_inserted": tot("points_inserted"), "occupied_voxels": tot("occupied_voxels"),
            "contaminated_voxels": tot("contaminated_voxels"),
            "contaminated_voxel_fraction": tot("contaminated_voxels") / tot("occupied_voxels"),
            "moving_points_retained": tot("moving_points_retained"), "moving_points_total": tot("moving_points_total"),
            "static_point_retention": tot("static_points_kept") / sum(
                w["mapping"]["raw_unfiltered"]["static_points_kept"] for w in ws),
            "static_coverage": tot("static_voxels") / ref, "map_build_s": tot("map_build_s")}
        S["mapping"][m] = per

    S["runtime_memory"] = {w["window"]: {**w["runtime"], **w["memory"]} for w in [dev] + ws}
    frames = sum(w["n_frames"] for w in ws)
    S["runtime_memory"]["heldout_pooled"] = {
        "detect_s_per_frame_mean": sum(w["runtime"]["detect_s_per_frame_mean"] * w["n_frames"] for w in ws) / frames,
        "detect_s_per_frame_max": max(w["runtime"]["detect_s_per_frame_max"] for w in ws),
        "peak_rss_max_MB": max(w["memory"]["peak_rss_end_MB"] for w in ws)}
    (OUT / "summary.json").write_text(json.dumps(S, indent=2))

    f = lambda x: "n/a" if x != x else f"{x:.3f}"
    print(f"{'window':16s} {'mov pts':>8s} {'IoU_mov':>8s} {'IoU_sta':>8s} {'P':>6s} {'R':>6s} {'F1':>6s} {'staticFP%':>9s}")
    for k in ["development"] + EVAL + ["heldout_pooled"]:
        s = S["point_metrics"]["temporal_detector"][k]["all_frames"]
        print(f"{k:16s} {s['n_moving_gt']:8d} {f(s['iou_moving']):>8s} {f(s['iou_static']):>8s} {f(s['precision']):>6s} "
              f"{f(s['recall']):>6s} {f(s['f1']):>6s} {100 * s['static_fp_rate']:9.3f}")
    print("macro mean IoU (secondary):", f(S["point_metrics"]["temporal_detector"]["heldout_macro_mean"]["iou_moving"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
