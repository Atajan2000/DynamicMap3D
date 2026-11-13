"""Data preparation for the Phase 3 viewer (no Rerun dependency, fully testable).

Everything shown comes from saved files:
  per-point detector output  results/phase3/predictions/<window>.npz  (frozen detector, SHA-256 = Phase 2)
  window/pooled metrics      results/phase2/windows/<window>/metrics.json, results/phase2/summary.json
  per-frame runtime          results/phase2/windows/<window>/per_frame.csv  (measured in the Phase 2 run)
  object speeds              results/phase2/windows/<window>/objects.csv
Ground-truth labels are used ONLY for the evaluation layers; detector layers never read them.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from dynamicmap3d.data.kitti import transform
from dynamicmap3d.data.semantickitti import MOVING_CLASSES, is_ignored, is_moving, read_labels
from dynamicmap3d.data.splits import WindowSpec, all_windows
from dynamicmap3d.data.window import load_window
from dynamicmap3d.evaluation.mapping import voxel_keys
from dynamicmap3d.evaluation.objects import PARKED_VEHICLES

VOXEL = 0.2  # identical to the Phase 2 map evaluation

# Colour semantics are global: orange = removed by detector, blue = annotated moving but kept (miss),
# white = annotated static but removed (false removal), aqua = parked vehicle kept, grey = kept geometry.
ORANGE, BLUE, AQUA, WHITE = (217, 89, 38), (57, 135, 229), (25, 158, 112), (250, 250, 248)
GREY, LIGHT, DIM = (122, 121, 117), (216, 215, 208), (68, 68, 63)

DETECTOR_CLASSES = {  # class id -> (label, colour); computed from detector output only
    0: ("kept: no motion evidence", GREY),
    1: ("removed: dynamic (detector)", ORANGE),
    2: ("kept: voted but rejected by cluster-size filter", LIGHT),
    3: ("outside detector range (3-50 m), never removed", DIM),
}
EVAL_CLASSES = {  # class id -> (label, colour); evaluation only (uses ground-truth labels)
    0: ("static, kept (correct)", GREY),
    1: ("moving, removed (correct)", ORANGE),
    2: ("moving, KEPT (missed)", BLUE),
    3: ("static, REMOVED (false removal)", WHITE),
    4: ("unlabeled/outlier (ignored in metrics)", DIM),
}


@dataclass
class ViewData:
    window: WindowSpec
    frames: list[int]
    scans: dict[int, np.ndarray]
    poses_seq: dict[int, np.ndarray]       # sensor -> sequence world (SemanticKITTI poses)
    poses_display: dict[int, np.ndarray]   # sensor -> display world (= sensor frame of first shown frame)
    pred: dict[str, dict[int, np.ndarray]]  # dynamic, candidate, valid, ground, votes, informative
    neighbors_used: dict[int, int]
    sem: dict[int, np.ndarray] = field(default_factory=dict)   # evaluation only
    inst: dict[int, np.ndarray] = field(default_factory=dict)  # evaluation only
    per_frame: dict[int, dict] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    objects: list[dict] = field(default_factory=list)


def parse_frames(spec: str) -> tuple[int, int]:
    """'600:700' -> (600, 699). The end is EXCLUSIVE, like a Python slice."""
    a, b = spec.split(":")
    first, end = int(a), int(b)
    if end <= first:
        raise ValueError(f"empty frame range '{spec}' (end is exclusive)")
    return first, end - 1


def resolve_window(seq: str, first: int, last: int, splits="configs/splits.yaml") -> WindowSpec:
    """Return the frozen window that contains [first, last] for this sequence."""
    seq = str(seq).zfill(2)
    for w in all_windows(splits).values():
        if w.sequence == seq and w.first <= first and last <= w.last:
            return w
    known = ", ".join(f"--seq {w.sequence} --frames {w.first}:{w.last + 1}" for w in all_windows(splits).values())
    raise ValueError(f"seq {seq} frames {first}-{last} is not inside an evaluated window. Available: {known}")


def load_predictions(path: Path, frames: list[int], scans: dict[int, np.ndarray]) -> tuple[dict, dict]:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python scripts/export_predictions.py --window <key>` once")
    z = np.load(path)
    all_frames = z["frames"].tolist()
    offs = np.r_[0, np.cumsum(z["counts"])]
    pred = {k: {} for k in ("dynamic", "candidate", "valid", "ground", "votes", "informative")}
    nbrs = {}
    for f in frames:
        if f not in all_frames:
            raise ValueError(f"frame {f} not in {path}")
        i = all_frames.index(f)
        n = int(z["counts"][i])
        if n != len(scans[f]):
            raise ValueError(f"frame {f}: {n} predictions for {len(scans[f])} scan points")
        for k in pred:
            pred[k][f] = z[k][offs[i]:offs[i + 1]]
        nbrs[f] = int(z["neighbors_used"][i])
    return pred, nbrs


def load_view_data(window: WindowSpec, first: int | None = None, last: int | None = None,
                   root=Path("data/semantickitti/dataset"), results=Path("results"),
                   with_labels: bool = True) -> ViewData:
    first = window.first if first is None else first
    last = window.last if last is None else last
    W = load_window(root, window.sequence, first, last, "semantickitti", with_labels=False)
    pred, nbrs = load_predictions(results / "phase3/predictions" / f"{window.key}.npz", W.frames, W.scans)
    T0inv = np.linalg.inv(W.poses[first])
    d = ViewData(window, W.frames, W.scans, W.poses, {f: T0inv @ W.poses[f] for f in W.frames}, pred, nbrs)
    if with_labels:
        sdir = Path(root) / "sequences" / window.sequence / "labels"
        for f in W.frames:
            d.sem[f], d.inst[f] = read_labels(sdir / f"{f:06d}.label")
            if len(d.sem[f]) != len(W.scans[f]):
                raise ValueError(f"frame {f}: label/scan length mismatch")
    wdir = results / "phase2/windows" / window.key
    d.metrics = json.loads((wdir / "metrics.json").read_text())
    d.summary = json.loads((results / "phase2/summary.json").read_text())
    with open(wdir / "per_frame.csv") as fh:
        d.per_frame = {int(r["frame"]): r for r in csv.DictReader(fh)}
    with open(wdir / "objects.csv") as fh:
        d.objects = list(csv.DictReader(fh))
    return d


# ---- per-point classes ---------------------------------------------------------------------
def detector_classes(dynamic: np.ndarray, candidate: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Detector-only point states (no labels involved)."""
    c = np.zeros(len(dynamic), np.uint16)
    c[candidate & ~dynamic] = 2
    c[dynamic] = 1
    c[~valid] = 3
    return c


def evaluation_classes(dynamic: np.ndarray, sem: np.ndarray) -> np.ndarray:
    gt = is_moving(sem)
    c = np.zeros(len(dynamic), np.uint16)
    c[gt & dynamic] = 1
    c[gt & ~dynamic] = 2
    c[~gt & dynamic] = 3
    c[is_ignored(sem)] = 4
    return c


# ---- maps ------------------------------------------------------------------------------------
def voxel_centers(keys: np.ndarray, resolution: float) -> np.ndarray:
    m = (1 << 21) - 1
    off = 1 << 20
    ijk = np.stack([(keys >> 42) & m, (keys >> 21) & m, keys & m], axis=1) - off
    return (ijk + 0.5) * resolution


def map_increments(d: ViewData, masks: dict[str, dict[int, np.ndarray]], resolution: float = VOXEL):
    """Per layer and frame, the voxels occupied for the first time (display coordinates).

    Voxelised in the sequence world frame exactly as in the Phase 2 map evaluation, so final voxel
    counts are directly comparable with results/phase2 mapping numbers.
    """
    T0inv = np.linalg.inv(d.poses_seq[d.frames[0]])
    out = {name: {} for name in masks}
    totals = {}
    for name, mk in masks.items():
        seen = np.empty(0, np.int64)
        for f in d.frames:
            pts = d.scans[f][:, :3].astype(np.float64)
            k = np.unique(voxel_keys(transform(pts[mk[f]], d.poses_seq[f]), resolution))
            new = np.setdiff1d(k, seen, assume_unique=True)
            seen = np.union1d(seen, new)
            out[name][f] = transform(voxel_centers(new, resolution), T0inv).astype(np.float32)
        totals[name] = int(seen.size)
    return out, totals


def map_masks(d: ViewData) -> dict[str, dict[int, np.ndarray]]:
    v, dyn = d.pred["valid"], d.pred["dynamic"]
    return {"raw_map": {f: v[f] for f in d.frames},                     # Phase 2 "raw_unfiltered"
            "filtered_map": {f: v[f] & ~dyn[f] for f in d.frames},      # Phase 2 "temporal_detector"
            "removed": {f: v[f] & dyn[f] for f in d.frames}}


def evaluation_masks(d: ViewData) -> dict[str, dict[int, np.ndarray]]:
    v, dyn = d.pred["valid"], d.pred["dynamic"]
    parked = {f: np.isin(d.sem[f], list(PARKED_VEHICLES)) for f in d.frames}
    gt = {f: is_moving(d.sem[f]) for f in d.frames}
    static = {f: ~gt[f] & ~is_ignored(d.sem[f]) for f in d.frames}
    return {"tp_removed": {f: v[f] & dyn[f] & gt[f] for f in d.frames},
            "fn_kept": {f: v[f] & ~dyn[f] & gt[f] for f in d.frames},
            "fp_removed": {f: v[f] & dyn[f] & static[f] for f in d.frames},
            "parked_kept": {f: v[f] & ~dyn[f] & parked[f] for f in d.frames}}


# ---- annotations / diagnostics ------------------------------------------------------------------
def object_annotations(d: ViewData, f: int, min_points: int = 10):
    """Evaluation-only labels for annotated moving objects in frame f (centroid in display frame)."""
    speeds = {(int(r["class_id"]), int(r["instance"])): float(r["instance_speed_mps"])
              for r in d.objects if r["moving"] == "True"}
    sem, inst, dyn = d.sem[f], d.inst[f], d.pred["dynamic"][f]
    gt = is_moving(sem)
    out = []
    key = (sem.astype(np.uint32) << 16) | inst
    for k in np.unique(key[gt]):
        m = key == k
        if m.sum() < min_points:
            continue
        cls, iid = int(k >> 16), int(k & 0xFFFF)
        frac = float(dyn[m].mean())
        sp = speeds.get((cls, iid), float("nan"))
        c = transform(d.scans[f][m, :3].mean(0, keepdims=True).astype(np.float64), d.poses_display[f])[0]
        state = "removed" if frac >= 0.5 else ("MISSED, near-stationary" if sp < 0.5 else "MISSED")
        sp_txt = "n/a" if sp != sp else f"{sp:.1f} m/s"
        out.append({"position": c, "class": MOVING_CLASSES[cls], "instance": iid, "flagged": frac,
                    "speed": sp, "missed": frac < 0.5,
                    "label": f"GT {MOVING_CLASSES[cls]} #{iid} | {sp_txt} | {100 * frac:.0f}% removed | {state}"})
    return out


def frame_diagnostics(d: ViewData, f: int) -> dict:
    dyn, valid, cand = d.pred["dynamic"][f], d.pred["valid"][f], d.pred["candidate"][f]
    n = len(dyn)
    out = {"frame": f, "points": n, "points_in_range": int(valid.sum()), "removed": int(dyn.sum()),
           "retained": int(n - dyn.sum()), "dynamic_fraction": float(dyn.sum() / n),
           "rejected_candidates": int((cand & ~dyn).sum()), "neighbors_used": d.neighbors_used[f],
           "detect_ms_phase2": 1000 * float(d.per_frame[f]["detect_s"])}
    if d.sem:
        gt = is_moving(d.sem[f])
        keep = ~is_ignored(d.sem[f])
        out.update({"gt_moving": int(gt.sum()), "tp": int((dyn & gt).sum()), "fn": int((~dyn & gt).sum()),
                    "fp": int((dyn & ~gt & keep).sum())})
    return out
