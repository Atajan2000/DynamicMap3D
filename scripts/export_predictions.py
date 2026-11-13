"""Export per-point outputs of the FROZEN detector for visualization (run once per window).

Phase 2 saved only aggregate results, so the per-point masks are regenerated here with the frozen
configuration. Each export must reproduce the Phase 2 prediction SHA-256 exactly, or it is rejected.
Labels are never loaded. Output: results/phase3/predictions/<window>.npz (git-ignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import yaml

from dynamicmap3d.data.splits import get_window
from dynamicmap3d.data.window import load_window
from dynamicmap3d.motion.detector import DetectorConfig, TemporalMotionDetector


def pred_hash(masks: list[np.ndarray]) -> str:  # identical to scripts/evaluate_window.py
    h = hashlib.sha256()
    for m in masks:
        h.update(np.packbits(m).tobytes())
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", required=True)
    ap.add_argument("--root", type=Path, default=Path("data/semantickitti/dataset"))
    a = ap.parse_args()
    frozen = json.loads(Path("configs/FROZEN_phase1.json").read_text())
    for f, h in frozen["sha256"].items():
        if hashlib.sha256(Path(f).read_bytes()).hexdigest() != h:
            raise SystemExit(f"frozen file changed: {f}")
    w = get_window(a.window)
    cfg_all = yaml.safe_load(Path("configs/phase1.yaml").read_text())
    W = load_window(a.root, w.sequence, w.first, w.last, cfg_all["pose_source"], with_labels=False)
    det = TemporalMotionDetector(W.scans, W.poses, DetectorConfig.from_dict(cfg_all["detector"]))
    out = {k: [] for k in ("dynamic", "candidate", "valid", "ground", "votes", "informative")}
    nbrs, t0 = [], time.perf_counter()
    for f in W.frames:
        d = det.detect(f)
        for k in out:
            out[k].append(getattr(d, k))
        nbrs.append(len(d.neighbors_used))
    elapsed = time.perf_counter() - t0

    saved = json.loads((Path("results/phase2/windows") / w.key / "metrics.json").read_text())["prediction_sha256"]
    got = {"temporal_detector": pred_hash(out["dynamic"]), "votes_only": pred_hash(out["candidate"])}
    if got != saved:
        raise SystemExit(f"export does not reproduce Phase 2 predictions: {got} != {saved}")

    dst = Path("results/phase3/predictions")
    dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        dst / f"{w.key}.npz", frames=np.array(W.frames), counts=np.array([len(m) for m in out["dynamic"]]),
        **{k: np.concatenate(out[k]) for k in ("dynamic", "candidate", "valid", "ground")},
        votes=np.concatenate(out["votes"]).astype(np.uint8),
        informative=np.concatenate(out["informative"]).astype(np.uint8),
        neighbors_used=np.array(nbrs), sequence=w.sequence,
        prediction_sha256=json.dumps(got), frozen_config_sha256=frozen["sha256"]["configs/phase1.yaml"])
    print(f"{w.key}: {len(W.frames)} frames exported, sha256 matches Phase 2, detector {elapsed:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
