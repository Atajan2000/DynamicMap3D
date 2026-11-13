"""Phase 3 visualization correctness: indexing, colouring, GT separation, map consistency,
annotations, launcher and recording validity."""

import inspect
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from dynamicmap3d.data.semantickitti import is_moving
from dynamicmap3d.data.splits import get_window
from dynamicmap3d.evaluation.mapping import voxel_keys
from dynamicmap3d.viz import data as vd
from dynamicmap3d.viz.text import window_markdown

ROOT = Path(__file__).parents[1]
RERUN = Path(sys.executable).parent / ("rerun.exe" if sys.platform == "win32" else "rerun")


def have(key):
    w = get_window(key, ROOT / "configs/splits.yaml")
    return (ROOT / "results/phase3/predictions" / f"{key}.npz").exists() and \
        (ROOT / "data/semantickitti/dataset/sequences" / w.sequence / "velodyne" / f"{w.last:06d}.bin").exists()


needs = lambda key: pytest.mark.skipif(not have(key), reason=f"{key} data/predictions missing")


@pytest.fixture(autouse=True)
def _cwd(monkeypatch):
    monkeypatch.chdir(ROOT)


# ---- pure logic ------------------------------------------------------------------------------
def test_frame_range_end_is_exclusive_and_resolves_to_frozen_window():
    assert vd.parse_frames("600:700") == (600, 699)
    with pytest.raises(ValueError):
        vd.parse_frames("5:5")
    assert vd.resolve_window("07", 600, 699).key == "eval2"
    assert vd.resolve_window("8", 1850, 1860).key == "eval1"          # sub-range allowed
    with pytest.raises(ValueError):
        vd.resolve_window("08", 95, 110)                             # crosses the window edge
    with pytest.raises(ValueError):
        vd.resolve_window("05", 0, 10)                               # never evaluated


def test_detector_classes_partition_and_never_see_labels():
    assert list(inspect.signature(vd.detector_classes).parameters) == ["dynamic", "candidate", "valid"]
    dyn = np.array([1, 0, 0, 0, 0], bool)
    cand = np.array([1, 1, 0, 0, 0], bool)
    valid = np.array([1, 1, 1, 0, 0], bool)
    assert vd.detector_classes(dyn, cand, valid).tolist() == [1, 2, 0, 3, 3]
    assert set(vd.DETECTOR_CLASSES) == {0, 1, 2, 3}


def test_evaluation_classes_and_colour_semantics():
    sem = np.array([252, 252, 10, 50, 0, 254], np.uint16)
    dyn = np.array([1, 0, 1, 0, 1, 1], bool)
    assert vd.evaluation_classes(dyn, sem).tolist() == [1, 2, 3, 0, 4, 1]
    assert vd.EVAL_CLASSES[1][1] == vd.DETECTOR_CLASSES[1][1] == vd.ORANGE   # removed = orange everywhere
    assert vd.EVAL_CLASSES[2][1] == vd.BLUE and vd.EVAL_CLASSES[3][1] == vd.WHITE


def test_voxel_centers_invert_voxel_keys():
    p = np.random.default_rng(0).uniform(-300, 300, (1000, 3))
    k = voxel_keys(p, 0.2)
    assert np.array_equal(voxel_keys(vd.voxel_centers(k, 0.2), 0.2), k)
    assert np.abs(vd.voxel_centers(k, 0.2) - p).max() <= 0.2 * np.sqrt(3) / 2 + 1e-9


def test_prediction_count_mismatch_is_rejected(tmp_path):
    np.savez(tmp_path / "w.npz", frames=np.array([5]), counts=np.array([3]),
             **{k: np.zeros(3, bool) for k in ("dynamic", "candidate", "valid", "ground")},
             votes=np.zeros(3, np.uint8), informative=np.zeros(3, np.uint8), neighbors_used=np.array([6]))
    with pytest.raises(ValueError):
        vd.load_predictions(tmp_path / "w.npz", [5], {5: np.zeros((4, 4), np.float32)})
    with pytest.raises(FileNotFoundError):
        vd.load_predictions(tmp_path / "missing.npz", [5], {5: np.zeros((3, 4), np.float32)})


# ---- real data -----------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def eval2():
    if not have("eval2"):
        pytest.skip("eval2 missing")
    import os
    old = os.getcwd(); os.chdir(ROOT)
    try:
        return vd.load_view_data(get_window("eval2"))
    finally:
        os.chdir(old)


@needs("eval2")
def test_scan_prediction_alignment_reproduces_phase2_hash(eval2):
    import hashlib
    h = hashlib.sha256()
    for f in eval2.frames:
        assert len(eval2.pred["dynamic"][f]) == len(eval2.scans[f]) == len(eval2.sem[f])
        h.update(np.packbits(eval2.pred["dynamic"][f]).tobytes())
    saved = json.loads((ROOT / "results/phase2/windows/eval2/metrics.json").read_text())["prediction_sha256"]
    assert h.hexdigest() == saved["temporal_detector"]


@needs("eval2")
def test_display_poses_are_relative_to_first_shown_frame(eval2):
    f0, f = eval2.frames[0], eval2.frames[50]
    assert np.allclose(eval2.poses_display[f0], np.eye(4))
    p = eval2.scans[f][:20, :3].astype(np.float64)
    seq = (eval2.poses_seq[f] @ np.c_[p, np.ones(20)].T).T[:, :3]
    disp = (eval2.poses_display[f] @ np.c_[p, np.ones(20)].T).T[:, :3]
    back = (eval2.poses_seq[f0] @ np.c_[disp, np.ones(20)].T).T[:, :3]
    assert np.allclose(back, seq, atol=1e-6)


@needs("eval2")
def test_per_frame_diagnostics_sum_to_window_confusion(eval2):
    s = eval2.metrics["point_metrics"]["temporal_detector"]["all_frames"]
    tot = {k: 0 for k in ("tp", "fn", "fp", "removed", "retained", "points")}
    for f in eval2.frames:
        d = vd.frame_diagnostics(eval2, f)
        assert d["removed"] + d["retained"] == d["points"]
        for k in tot:
            tot[k] += d[k]
    assert (tot["tp"], tot["fn"], tot["fp"]) == (s["tp"], s["fn"], s["fp"])


@needs("eval2")
def test_raw_and_filtered_maps_match_phase2_and_have_no_duplicates(eval2):
    inc, totals = vd.map_increments(eval2, vd.map_masks(eval2))
    mp = eval2.metrics["mapping"]
    assert totals["raw_map"] == mp["raw_unfiltered"]["occupied_voxels"]
    assert totals["filtered_map"] == mp["temporal_detector"]["occupied_voxels"]
    assert totals["filtered_map"] <= totals["raw_map"]
    T0 = eval2.poses_seq[eval2.frames[0]]
    for layer in ("raw_map", "filtered_map"):
        pts = np.concatenate([p for p in inc[layer].values()]).astype(np.float64)
        keys = voxel_keys((T0 @ np.c_[pts, np.ones(len(pts))].T).T[:, :3], 0.2)
        assert len(np.unique(keys)) == len(keys) == totals[layer]   # each voxel logged exactly once


@needs("eval2")
def test_evaluation_layers_never_change_detector_output(eval2):
    f = eval2.frames[10]
    before = vd.detector_classes(eval2.pred["dynamic"][f], eval2.pred["candidate"][f], eval2.pred["valid"][f]).copy()
    sem_saved = eval2.sem[f].copy()
    eval2.sem[f][:] = 252                                  # pretend every point is moving
    vd.evaluation_classes(eval2.pred["dynamic"][f], eval2.sem[f])
    after = vd.detector_classes(eval2.pred["dynamic"][f], eval2.pred["candidate"][f], eval2.pred["valid"][f])
    eval2.sem[f][:] = sem_saved
    assert np.array_equal(before, after)


@needs("eval0")
def test_object_annotations_match_phase2_objects_and_flag_near_stationary_misses():
    d = vd.load_view_data(get_window("eval0"), 40, 41)
    rows = {(int(r["frame"]), int(r["class_id"]), int(r["instance"])): r for r in d.objects if r["moving"] == "True"}
    ann = vd.object_annotations(d, 40)
    assert ann
    inv = {v: k for k, v in __import__("dynamicmap3d.data.semantickitti", fromlist=["x"]).MOVING_CLASSES.items()}
    for a in ann:
        r = rows[(40, inv[a["class"]], a["instance"])]
        assert a["flagged"] == pytest.approx(float(r["flagged_fraction"]))
    assert any(a["missed"] and a["speed"] < 0.5 and "near-stationary" in a["label"] for a in ann)


@needs("eval2")
def test_displayed_metrics_equal_saved_results(eval2):
    md = window_markdown(eval2)
    S = json.loads((ROOT / "results/phase2/summary.json").read_text())["point_metrics"]["temporal_detector"]
    expect = {"development": 0.658, "heldout_pooled": 0.503, "eval2": 0.916, "eval0": 0.241, "eval1": 0.282}
    for k, v in expect.items():
        assert round(S[k]["all_frames"]["iou_moving"], 3) == v
        assert f"| {v:.3f} |" in md
    assert "not an official benchmark" in md and "NOT per frame" in md


# ---- launcher + recording -------------------------------------------------------------------------
def test_launcher_help_and_bad_range():
    run = lambda *a: subprocess.run([sys.executable, "-I", "scripts/view.py", *a], cwd=ROOT, capture_output=True, text=True)
    assert run("--help").returncode == 0
    assert run("--seq", "05", "--frames", "0:10").returncode == 2


@needs("eval0")
def test_saved_recording_is_valid_and_complete(tmp_path):
    rrd = tmp_path / "t.rrd"
    r = subprocess.run([sys.executable, "-I", "scripts/view.py", "--seq", "08", "--frames", "0:3", "--save", str(rrd)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert subprocess.run([str(RERUN), "rrd", "verify", str(rrd)], capture_output=True).returncode == 0
    out = subprocess.run([str(RERUN), "rrd", "print", str(rrd)], capture_output=True, text=True).stdout
    paths = set(re.findall(r"/(?:world|diagnostics|info|plots)/[a-z_/0-9]*", out))
    for p in ("/world/sensor", "/world/sensor/scan_detector", "/world/sensor/scan_eval", "/world/raw_map/f000000",
              "/world/filtered_map/f000000", "/world/eval_objects", "/diagnostics/frame", "/info/window"):
        assert p in paths, p


def test_no_eval_mode_contains_no_ground_truth_layers(tmp_path):
    if not have("eval0"):
        pytest.skip("eval0 missing")
    rrd = tmp_path / "n.rrd"
    subprocess.run([sys.executable, "-I", "scripts/view.py", "--seq", "08", "--frames", "0:2", "--save", str(rrd), "--no-eval"],
                   cwd=ROOT, check=True, capture_output=True)
    out = subprocess.run([str(RERUN), "rrd", "print", str(rrd)], capture_output=True, text=True).stdout
    assert "/world/sensor/scan_detector" in out
    assert "scan_eval" not in out and "eval_map" not in out and "eval_objects" not in out
