"""Select the development window and freeze evaluation windows from annotation statistics.

Selection uses aggregate annotation statistics only (no algorithm output), and is written to
configs/splits.yaml BEFORE any detector is run. Evaluation windows are never used for tuning.

Pools (disjoint sequences, following the LiDAR-MOS convention that seq 08 is validation):
  dev  : sequences 00-07, 09, 10
  eval : seq 08 (two non-overlapping windows) + one window from another sequence != dev seq

Window score deliberately does NOT use the raw moving-point count. It rewards:
  persistent moving objects (>= MIN_OBJ_PTS points in >= MIN_OBJ_FRAMES frames), capped at 5 per group
  class diversity (vehicles AND pedestrians/cyclists), temporal coverage of motion,
  parked/static movable objects (hard negatives), static structure (buildings, poles, trunks...)
Hard constraints: coverage >= 0.6, >= 2 persistent movers, moving fraction <= 5 %,
  ego speed 1-12 m/s, structure fraction >= 15 %.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml

from dynamicmap3d.data.semantickitti import MOVING_CLASSES

VEHICLE = {252, 255, 256, 257, 258, 259}  # 255 moving-motorcyclist counted with vehicles
VRU = {253, 254}                           # moving-bicyclist, moving-person
MIN_OBJ_PTS, MIN_OBJ_FRAMES = 50, 10
# Geometric usefulness: above ~12 m/s consecutive scans are >1.2 m apart, which reduces viewpoint
# overlap and makes free-space (visibility) evidence asymmetric. Highway-speed windows are excluded.
MAX_EGO_SPEED = 12.0
DEV_POOL = ["00", "01", "02", "03", "04", "05", "06", "07", "09", "10"]


def load(stats_dir: Path):
    frames = defaultdict(dict)
    with open(stats_dir / "frame_stats.csv") as f:
        for r in csv.DictReader(f):
            frames[r["seq"]][int(r["frame"])] = {k: float(v) for k, v in r.items() if k != "seq"}
    objs = defaultdict(dict)
    for line in open(stats_dir / "moving_instances.jsonl"):
        d = json.loads(line)
        objs[d["seq"]][d["frame"]] = d["objects"]
    return frames, objs


def window_features(fr: dict, ob: dict, start: int, length: int) -> dict:
    idx = range(start, start + length)
    tot = lambda k: sum(fr[i][k] for i in idx)
    seen = defaultdict(int)          # (cls, iid) -> frames with >= MIN_OBJ_PTS
    any_obj = defaultdict(int)
    covered = 0
    for i in idx:
        big = [(c, iid) for iid, c, n in ob.get(i, []) if n >= MIN_OBJ_PTS]
        covered += bool(big)
        for key in big:
            seen[key] += 1
        for iid, c, n in ob.get(i, []):
            any_obj[c] += 1
    pers = [k for k, n in seen.items() if n >= MIN_OBJ_FRAMES]
    veh = sorted({k for k in pers if k[0] in VEHICLE})
    vru = sorted({k for k in pers if k[0] in VRU})
    npts = tot("n_points")
    feat = {
        "start": start, "end": start + length - 1,
        "coverage": covered / length,
        "persistent_vehicles": len(veh), "persistent_vru": len(vru),
        "persistent_by_class": {MOVING_CLASSES[c]: sum(1 for k in pers if k[0] == c)
                                for c in sorted({k[0] for k in pers})},
        "moving_frac": tot("n_moving") / npts,
        "structure_frac": tot("n_structure") / npts,
        "static_movable_frac": tot("n_static_movable") / npts,
        "ignore_frac": tot("n_ignore") / npts,
        "ego_speed_mean": tot("ego_speed_mps") / length,
        "moving_points": int(tot("n_moving")),
    }
    feat["eligible"] = (feat["coverage"] >= 0.6 and len(pers) >= 2 and feat["moving_frac"] <= 0.05
                        and 1.0 <= feat["ego_speed_mean"] <= MAX_EGO_SPEED and feat["structure_frac"] >= 0.15)
    feat["score"] = (min(len(veh), 5) + min(len(vru), 5) + 2.0 * (bool(veh) and bool(vru))
                     + 2.0 * feat["coverage"] + 2.0 * min(feat["static_movable_frac"] / 0.03, 1.0)
                     + min(feat["structure_frac"] / 0.3, 1.0))
    return feat


def best(cands, exclude=()):
    for c in cands:
        if all(not (c["seq"] == e["seq"] and c["start"] <= e["end"] and e["start"] <= c["end"])
               for e in exclude):
            return c
    raise RuntimeError("no eligible window")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stats", type=Path, default=Path("results/label_stats"))
    ap.add_argument("--length", type=int, default=100)
    ap.add_argument("--step", type=int, default=10)
    ap.add_argument("--velodyne-cd", type=Path, default=Path("data/manifests/velodyne_central_directory.json"))
    ap.add_argument("--out", type=Path, default=Path("configs/splits.yaml"))
    a = ap.parse_args()
    frames, objs = load(a.stats)

    cands = []
    for seq in sorted(frames):
        n = len(frames[seq])
        for s in range(0, n - a.length + 1, a.step):
            f = window_features(frames[seq], objs[seq], s, a.length)
            f["seq"] = seq
            cands.append(f)
    elig = sorted([c for c in cands if c["eligible"]], key=lambda c: -c["score"])
    print(f"{len(cands)} windows, {len(elig)} eligible")

    dev = best([c for c in elig if c["seq"] in DEV_POOL])
    ev_a = best([c for c in elig if c["seq"] == "08"])
    ev_b = best(sorted([c for c in elig if c["seq"] == "08"],
                       key=lambda c: (-c["persistent_vru"], -c["score"])), exclude=[ev_a])
    ev_c = best([c for c in elig if c["seq"] in DEV_POOL and c["seq"] != dev["seq"]])

    cd = {n: (c, o) for n, c, u, crc, o in json.load(open(a.velodyne_cd))["members"]}
    dev_bytes = sum(cd[f"dataset/sequences/{dev['seq']}/velodyne/{i:06d}.bin"][0]
                    for i in range(dev["start"], dev["end"] + 1))

    keys = ["seq", "start", "end", "score", "coverage", "persistent_vehicles", "persistent_vru",
            "persistent_by_class", "moving_frac", "structure_frac", "static_movable_frac",
            "ego_speed_mean", "moving_points"]
    with open(a.stats / "window_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for c in elig[:40]:
            w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})

    def entry(c, role):
        return {"role": role, "sequence": c["seq"], "first_frame": c["start"], "last_frame": c["end"],
                "stats": {k: (round(c[k], 4) if isinstance(c[k], float) else c[k]) for k in keys[3:]}}

    splits = {
        "frozen_before_algorithm_runs": True,
        "policy": ("Parameters may be tuned ONLY on the development window. Evaluation windows are "
                   "selected from annotation statistics before any detector run and are reported once."),
        "selection_criteria": {"window_length": a.length, "min_object_points": MIN_OBJ_PTS,
                               "min_object_frames": MIN_OBJ_FRAMES,
                               "constraints": "coverage>=0.6, persistent movers>=2, moving_frac<=0.05, "
                                              "ego speed 1-12 m/s, structure_frac>=0.15",
                               "score": "min(veh,5)+min(vru,5)+2*diversity+2*coverage"
                                        "+2*min(static_movable/0.03,1)+min(structure/0.3,1)"},
        "development": entry(dev, "development (tuning allowed)") | {"velodyne_bytes": dev_bytes},
        "evaluation": [entry(ev_a, "held-out, seq 08 best overall"),
                       entry(ev_b, "held-out, seq 08 most moving pedestrians/cyclists"),
                       entry(ev_c, "held-out, different sequence from development")],
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(yaml.safe_dump(splits, sort_keys=False))
    print(yaml.safe_dump(splits, sort_keys=False))
    print("Top 8 eligible windows:")
    for c in elig[:8]:
        print(f"  seq {c['seq']} {c['start']:4d}-{c['end']:4d} score {c['score']:.2f} cov {c['coverage']:.2f} "
              f"veh {c['persistent_vehicles']} vru {c['persistent_vru']} mov {100*c['moving_frac']:.2f}% "
              f"parked {100*c['static_movable_frac']:.1f}% ego {c['ego_speed_mean']:.1f} m/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
