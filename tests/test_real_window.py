"""Real-data checks on the development window (skipped when the data is not present)."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from dynamicmap3d.data.kitti import scan_point_count
from dynamicmap3d.data.semantickitti import is_ignored, is_moving
from dynamicmap3d.data.window import load_window
from dynamicmap3d.motion.detector import DetectorConfig, TemporalMotionDetector

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data/semantickitti/dataset"
DEV = yaml.safe_load((ROOT / "configs/splits.yaml").read_text())["development"]
SEQ, FIRST, LAST = str(DEV["sequence"]).zfill(2), DEV["first_frame"], DEV["last_frame"]
pytestmark = [pytest.mark.realdata,
              pytest.mark.skipif(not (DATA / "sequences" / SEQ / "velodyne" / f"{LAST:06d}.bin").exists(),
                                 reason="development window not downloaded")]


def test_every_label_matches_its_scan():
    s = DATA / "sequences" / SEQ
    for i in range(FIRST, LAST + 1):
        assert (s / "labels" / f"{i:06d}.label").stat().st_size // 4 == scan_point_count(s / "velodyne" / f"{i:06d}.bin")


@pytest.mark.parametrize("key", ["development", "eval0", "eval1", "eval2"])
def test_every_window_label_matches_scan_and_archive(key):
    """Original point-to-label correspondence + archive CRC for every downloaded window."""
    import json, zlib
    from dynamicmap3d.data.splits import get_window
    w = get_window(key, ROOT / "configs/splits.yaml")
    s = DATA / "sequences" / w.sequence
    if not (s / "velodyne" / f"{w.last:06d}.bin").exists():
        pytest.skip(f"{key} not downloaded")
    man = json.loads((ROOT / "data/manifests" / f"window_{w.tag}.json").read_text())["velodyne_members"]
    for i in w.frames:
        b = (s / "velodyne" / f"{i:06d}.bin").read_bytes()
        assert len(b) == man[f"{i:06d}"]["size"] and zlib.crc32(b) & 0xFFFFFFFF == man[f"{i:06d}"]["crc32"]
        assert (s / "labels" / f"{i:06d}.label").stat().st_size // 4 == len(b) // 16


def test_detector_beats_chance_on_a_real_interior_frame():
    """Sanity, not a benchmark: precision must be far above the moving-point base rate."""
    cfg = DetectorConfig.from_dict(yaml.safe_load((ROOT / "configs/phase1.yaml").read_text())["detector"])
    mid = (FIRST + LAST) // 2
    W = load_window(DATA, SEQ, mid - 10, mid + 10)
    d = TemporalMotionDetector(W.scans, W.poses, cfg).detect(mid)
    keep = ~is_ignored(W.sem[mid])
    gt = is_moving(W.sem[mid])[keep]
    pred = d.dynamic[keep]
    assert gt.sum() > 0 and pred.sum() > 0
    precision = (pred & gt).sum() / pred.sum()
    assert precision > 10 * gt.mean()
    assert (pred & gt).sum() / gt.sum() > 0.2
