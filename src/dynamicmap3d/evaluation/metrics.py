"""Moving-object segmentation metrics (LiDAR-MOS convention: unlabeled/outlier points ignored)."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from dynamicmap3d.data.semantickitti import is_ignored, is_moving


@dataclass
class Confusion:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def add(self, pred: np.ndarray, gt: np.ndarray, mask: np.ndarray | None = None) -> None:
        if mask is not None:
            pred, gt = pred[mask], gt[mask]
        self.tp += int(np.sum(pred & gt))
        self.fp += int(np.sum(pred & ~gt))
        self.fn += int(np.sum(~pred & gt))
        self.tn += int(np.sum(~pred & ~gt))

    def __add__(self, other: "Confusion") -> "Confusion":
        """Pooling = summing counts over windows (micro-average), not averaging window metrics."""
        return Confusion(self.tp + other.tp, self.fp + other.fp, self.fn + other.fn, self.tn + other.tn)

    def summary(self) -> dict:
        """Positive class = moving. 'Removal' = predicted moving (the point is removed from the map).

        precision is NaN when nothing is predicted moving (e.g. the raw, unfiltered baseline).
        """
        d = lambda a, b: a / b if b else float("nan")
        p, r = d(self.tp, self.tp + self.fp), d(self.tp, self.tp + self.fn)
        return {"tp": self.tp, "fp": self.fp, "fn": self.fn, "tn": self.tn,
                "n_moving_gt": self.tp + self.fn, "n_static_gt": self.fp + self.tn,
                "iou_moving": d(self.tp, self.tp + self.fp + self.fn),
                "iou_static": d(self.tn, self.tn + self.fp + self.fn),
                "precision": p, "recall": r,
                "f1": d(2 * p * r, p + r) if p == p and r == r else float("nan"),
                "static_fp_rate": d(self.fp, self.fp + self.tn),        # static-point false-removal rate
                "moving_removal_rate": r}                               # identical to recall by definition


@dataclass
class Evaluator:
    """Accumulates point-level and object-level results over frames."""
    detect_threshold: float = 0.5
    min_object_points: int = 10
    all_points: Confusion = field(default_factory=Confusion)
    in_range: Confusion = field(default_factory=Confusion)
    objects: list = field(default_factory=list)                 # (frame, instance, class, n, recall)
    fp_by_class: dict = field(default_factory=lambda: defaultdict(int))

    def add(self, frame: int, pred: np.ndarray, sem: np.ndarray, inst: np.ndarray,
            valid_range: np.ndarray) -> None:
        keep = ~is_ignored(sem)
        gt = is_moving(sem)
        self.all_points.add(pred, gt, keep)
        self.in_range.add(pred, gt, keep & valid_range)
        fp = pred & ~gt & keep
        for cls, n in zip(*np.unique(sem[fp], return_counts=True)):
            self.fp_by_class[int(cls)] += int(n)
        if gt.any():
            key = (sem[gt].astype(np.uint32) << 16) | inst[gt]
            for k in np.unique(key):
                m = gt.copy()
                m[gt] = key == k
                n = int(m.sum())
                if n >= self.min_object_points:
                    self.objects.append((frame, int(k & 0xFFFF), int(k >> 16), n,
                                         float(pred[m].mean()), bool(valid_range[m].mean() >= 0.5)))

    def object_summary(self) -> dict:
        objs = [o for o in self.objects if o[5]]  # objects mostly inside the detector's range
        if not objs:
            return {"object_frames": 0}
        rec = np.array([o[4] for o in objs])
        per_inst = defaultdict(list)
        for f, iid, cls, n, r, _ in objs:
            per_inst[(cls, iid)].append(r >= self.detect_threshold)
        return {"object_frames": len(objs),
                "object_frames_detected": int((rec >= self.detect_threshold).sum()),
                "object_frame_detection_rate": float((rec >= self.detect_threshold).mean()),
                "mean_object_point_recall": float(rec.mean()),
                "instances": len(per_inst),
                "instances_detected_in_majority_of_frames": int(sum(np.mean(v) >= 0.5 for v in per_inst.values()))}
