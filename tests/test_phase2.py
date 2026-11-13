"""Phase 2 checks: metrics, map metrics, GT isolation, split integrity, determinism, boundaries."""

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from dynamicmap3d.data.splits import all_windows
from dynamicmap3d.evaluation.mapping import MapAccumulator, voxel_keys
from dynamicmap3d.evaluation.metrics import Confusion
from dynamicmap3d.evaluation.objects import depth_boundary_mask, summarize_objects
from dynamicmap3d.motion import detector as detector_module
from dynamicmap3d.motion.detector import TemporalMotionDetector
from dynamicmap3d.motion.range_image import RangeImageSpec
from synthetic import make_sequence
from test_detector import CFG

ROOT = Path(__file__).parents[1]


# ---- metrics ------------------------------------------------------------------------------
def test_static_iou_f1_and_rates_from_known_counts():
    s = Confusion(tp=6, fp=2, fn=4, tn=88).summary()
    assert s["iou_moving"] == pytest.approx(6 / 12) and s["iou_static"] == pytest.approx(88 / 94)
    assert s["precision"] == pytest.approx(0.75) and s["recall"] == pytest.approx(0.6)
    assert s["f1"] == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert s["static_fp_rate"] == pytest.approx(2 / 90) and s["moving_removal_rate"] == s["recall"]
    assert (s["n_moving_gt"], s["n_static_gt"]) == (10, 90)


def test_pooling_is_count_summation_not_metric_averaging():
    rng = np.random.default_rng(0)
    parts, whole = [], Confusion()
    for n in (50, 500, 5000):
        p, g = rng.random(n) < 0.1, rng.random(n) < 0.05
        c = Confusion(); c.add(p, g); parts.append(c); whole.add(p, g)
    pooled = parts[0] + parts[1] + parts[2]
    assert pooled == whole
    assert pooled.summary()["iou_moving"] != pytest.approx(np.mean([c.summary()["iou_moving"] for c in parts]))


def test_raw_baseline_precision_is_undefined_not_zero():
    c = Confusion(); c.add(np.zeros(4, bool), np.array([1, 0, 0, 1], bool))
    s = c.summary()
    assert np.isnan(s["precision"]) and s["recall"] == 0 and s["static_fp_rate"] == 0


# ---- map metrics --------------------------------------------------------------------------
def test_voxel_keys_unique_per_voxel_including_negative_coordinates():
    k = voxel_keys(np.array([[0.01, 0.01, 0.01], [0.19, 0.0, 0.1], [-0.01, 0, 0], [0, -0.01, 0], [0, 0, 0.2]]), 0.2)
    assert k[0] == k[1] and len({k[0], k[2], k[3], k[4]}) == 4


def test_map_metrics_and_static_point_preservation():
    xyz = np.array([[0.05, 0, 0], [0.06, 0, 0], [1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0]])
    keys = voxel_keys(xyz, 0.2)
    gt_mov = np.array([0, 1, 1, 0, 0], bool)        # voxel 0 mixes static + moving
    gt_sta = np.array([1, 0, 0, 1, 1], bool)
    raw, oracle, partial = MapAccumulator(0.2), MapAccumulator(0.2), MapAccumulator(0.2)
    raw.add(keys, np.ones(5, bool), gt_mov, gt_sta)
    oracle.add(keys, ~gt_mov, gt_mov, gt_sta)
    partial.add(keys, np.array([1, 1, 0, 0, 1], bool), gt_mov, gt_sta)   # removes one moving + one static
    ref = raw.static_vox.size
    r, o, p = raw.summary(ref), oracle.summary(ref), partial.summary(ref)
    assert (r["occupied_voxels"], r["contaminated_voxels"], r["static_point_retention"]) == (4, 2, 1.0)
    assert (o["contaminated_voxels"], o["static_point_retention"], o["static_coverage"]) == (0, 1.0, 1.0)
    assert o["occupied_voxels"] == 3
    assert p["moving_points_retained"] == 1 and p["static_point_retention"] == pytest.approx(2 / 3)
    assert p["static_coverage"] == pytest.approx(2 / 3) and p["contaminated_voxels"] == 1


def test_map_growth_is_cumulative_and_monotone():
    acc = MapAccumulator(0.5)
    for f in range(4):
        xyz = np.c_[np.arange(10) + f * 5.0, np.zeros(10), np.zeros(10)]
        acc.add(voxel_keys(xyz, 0.5), np.ones(10, bool), np.zeros(10, bool), np.ones(10, bool))
    assert acc.growth == sorted(acc.growth) and acc.growth[-1] == len(acc.occupied) == 25


# ---- object analysis ----------------------------------------------------------------------
def test_object_summary_speed_bins_and_parked_vehicles():
    base = dict(class_id=252, moving=True, n_points=50, median_range_m=10.0, in_range_fraction=1.0)
    recs = [dict(base, frame=0, cls="", instance=1, flagged_fraction=0.9, instance_speed_mps=8.0, **{"class": "moving-car"}),
            dict(base, frame=0, instance=2, flagged_fraction=0.1, instance_speed_mps=0.8, class_id=254, **{"class": "moving-person"}),
            dict(base, frame=0, instance=3, flagged_fraction=0.6, instance_speed_mps=np.nan, moving=False,
                 class_id=10, **{"class": "parked-car"})]
    s = summarize_objects(recs)
    assert s["moving_all"]["object_frames"] == 2 and s["moving_all"]["object_frames_detected"] == 1
    assert s["moving_by_speed"]["slow <1.5 m/s"]["object_frames_detected"] == 0
    assert s["moving_by_speed"]["fast >=5 m/s"]["object_frames_detected"] == 1
    assert s["parked_vehicles"]["object_frames_falsely_removed_ge50pct"] == 1


def test_depth_boundary_mask_marks_both_sides_of_a_range_jump():
    spec = RangeImageSpec(height=4, width=64, fov_up_deg=5, fov_down_deg=-5)
    az = np.radians(np.arange(64) * (360 / 64) - 180 + 2.8)
    r = np.where(np.arange(64) < 32, 10.0, 20.0)
    xyz = np.c_[r * np.cos(az), r * np.sin(az), np.zeros(64)]
    m = depth_boundary_mask(xyz, spec)
    assert m[31] and m[32] and not m[10] and not m[50]


# ---- ground-truth isolation, determinism, boundaries -----------------------------------------
def test_detector_has_no_access_to_labels():
    params = set(inspect.signature(TemporalMotionDetector.__init__).parameters)
    assert params == {"self", "scans", "poses", "config"}
    src = inspect.getsource(detector_module)
    assert "semantickitti" not in src and "label" not in src.lower().replace("label-free", "").replace("no labels", "")


def test_evaluator_reads_labels_only_after_all_detections():
    src = (ROOT / "scripts/evaluate_window.py").read_text()
    assert "with_labels=False" in src
    assert src.index("det.detect(f)") < src.index("read_labels(")


@pytest.fixture(scope="module")
def scene():
    return make_sequence(n_frames=21)


def test_detection_is_deterministic(scene):
    scans, poses, _ = scene
    a = TemporalMotionDetector(scans, poses, CFG).detect(10)
    b = TemporalMotionDetector(scans, poses, CFG).detect(10)
    assert np.array_equal(a.dynamic, b.dynamic) and np.array_equal(a.votes, b.votes)


def test_window_boundaries_use_only_frames_inside_the_window(scene):
    scans, poses, _ = scene
    win = {f: scans[f] for f in range(5, 16)}
    det = TemporalMotionDetector(win, {f: poses[f] for f in win}, CFG)
    assert det.detect(5).neighbors_used == [8, 11]          # offsets -6,-3 fall outside
    assert det.detect(15).neighbors_used == [9, 12]
    assert det.detect(10).neighbors_used == [7, 13]           # 4 and 16 fall outside
    assert all(5 <= j <= 15 for f in win for j in det.detect(f).neighbors_used)


# ---- split integrity / freeze ---------------------------------------------------------------
def test_evaluation_windows_are_disjoint_from_development_and_each_other():
    w = all_windows(ROOT / "configs/splits.yaml")
    dev = w.pop("development")
    assert len(w) == 3
    assert all(e.sequence != dev.sequence for e in w.values())
    ev = list(w.values())
    for i in range(len(ev)):
        for j in range(i + 1, len(ev)):
            if ev[i].sequence == ev[j].sequence:
                assert ev[i].last < ev[j].first or ev[j].last < ev[i].first


def test_frozen_files_unchanged():
    frozen = json.loads((ROOT / "configs/FROZEN_phase1.json").read_text())
    for f, h in frozen["sha256"].items():
        assert hashlib.sha256((ROOT / f).read_bytes()).hexdigest() == h, f
