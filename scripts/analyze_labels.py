"""Per-frame annotation statistics for SemanticKITTI sequences 00-10, streamed from the label zip.

Outputs (``results/label_stats``):
  frame_stats.csv        one row per (seq, frame): point counts per class group, ego speed
  moving_instances.jsonl one line per frame with moving objects: [[instance_id, class, n_points], ...]
  sequence_summary.csv   per-sequence totals and distinct moving objects per class
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

from dynamicmap3d.data.kitti import parse_poses
from dynamicmap3d.data.semantickitti import (GROUND, IGNORE_CLASSES, MOVING_CLASSES,
                                             STATIC_MOVABLE, STRUCTURE, labels_from_bytes)

GROUPS = {"ignore": IGNORE_CLASSES, "structure": STRUCTURE, "ground": GROUND,
          "static_movable": STATIC_MOVABLE}
MOVING_COLS = {c: n.replace("-", "_") for c, n in MOVING_CLASSES.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=Path("data/downloads/data_odometry_labels.zip"))
    ap.add_argument("--out", type=Path, default=Path("results/label_stats"))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(a.zip)
    names = defaultdict(list)
    for n in zf.namelist():
        m = re.search(r"sequences/(\d\d)/labels/(\d{6})\.label$", n)
        if m:
            names[m[1]].append((int(m[2]), n))

    lut = {g: np.zeros(65536, bool) for g in GROUPS}
    for g, classes in GROUPS.items():
        lut[g][list(classes)] = True

    fields = (["seq", "frame", "n_points", "ego_speed_mps"] + [f"n_{g}" for g in GROUPS]
              + ["n_moving"] + [f"n_{v}" for v in MOVING_COLS.values()] + ["n_moving_objects"])
    fs = open(a.out / "frame_stats.csv", "w", newline="")
    w = csv.DictWriter(fs, fieldnames=fields)
    w.writeheader()
    mi = open(a.out / "moving_instances.jsonl", "w")
    summary = []
    for seq in sorted(names):
        poses = parse_poses(zf.read(f"dataset/sequences/{seq}/poses.txt").decode())
        speed = np.r_[0.0, np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1) * 10.0]
        objs = defaultdict(set)  # class -> instance ids
        tot = defaultdict(int)
        for frame, name in sorted(names[seq]):
            sem, inst = labels_from_bytes(zf.read(name))
            row = {"seq": seq, "frame": frame, "n_points": sem.size,
                   "ego_speed_mps": round(float(speed[frame]), 3)}
            for g in GROUPS:
                row[f"n_{g}"] = int(lut[g][sem].sum())
            mv = sem >= 252
            row["n_moving"] = int(mv.sum())
            frame_objs = []
            if row["n_moving"]:
                pairs, counts = np.unique(
                    (sem[mv].astype(np.uint32) << 16) | inst[mv], return_counts=True)
                for p, c in zip(pairs.tolist(), counts.tolist()):
                    cls, iid = p >> 16, p & 0xFFFF
                    frame_objs.append([iid, cls, c])
                    objs[cls].add(iid)
            for c, col in MOVING_COLS.items():
                row[f"n_{col}"] = int((sem == c).sum())
            row["n_moving_objects"] = len(frame_objs)
            w.writerow(row)
            if frame_objs:
                mi.write(json.dumps({"seq": seq, "frame": frame, "objects": frame_objs}) + "\n")
            for k in ("n_points", "n_moving"):
                tot[k] += row[k]
            tot["frames_with_moving"] += bool(row["n_moving"])
        summary.append({"seq": seq, "frames": len(names[seq]), **tot,
                        "moving_frac": round(tot["n_moving"] / tot["n_points"], 5),
                        **{f"objects_{MOVING_COLS[c]}": len(objs[c]) for c in MOVING_COLS},
                        "mean_ego_speed_mps": round(float(speed.mean()), 2)})
        print(summary[-1], flush=True)
    fs.close()
    mi.close()
    with open(a.out / "sequence_summary.csv", "w", newline="") as f:
        sw = csv.DictWriter(f, fieldnames=list(summary[0]))
        sw.writeheader()
        sw.writerows(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
