import numpy as np
import pytest

from dynamicmap3d.motion.detector import DetectorConfig, TemporalMotionDetector
from dynamicmap3d.motion.ground import ground_mask
from dynamicmap3d.motion.range_image import RangeImageSpec, conservative_min
from synthetic import make_sequence

CFG = DetectorConfig(neighbor_offsets=(-6, -3, 3, 6), min_cluster_points=10,
                     spec=RangeImageSpec(height=64, width=1024, fov_up_deg=2.2, fov_down_deg=-24.2))


def test_projection_centre_and_wrap():
    spec = RangeImageSpec(height=64, width=2048)
    row, col, r, ok = spec.project(np.array([[10.0, 0.0, 0.0], [-10.0, 1e-6, 0.0], [-10.0, -1e-6, 0.0]]))
    assert ok.all() and np.allclose(r, 10.0)
    assert col[0] == 1024                       # straight ahead maps to the image centre
    assert {col[1], col[2]} <= {0, 2047}        # behind the sensor wraps at the image border


def test_min_range_image_keeps_closest_and_conservative_min_wraps():
    spec = RangeImageSpec(height=8, width=16, fov_up_deg=10, fov_down_deg=-10)
    pts = np.array([[10.0, 0, 0], [5.0, 0, 0], [20.0, 0, 0]])
    img = spec.min_range_image(pts)
    assert np.isfinite(img).sum() == 1 and img[np.isfinite(img)][0] == pytest.approx(5.0)
    a = np.full((3, 6), 9.0); a[1, 0] = 1.0
    assert conservative_min(a, 0, 1)[1, 5] == 1.0   # column neighbourhood wraps around


def test_ground_mask_separates_ground_from_object():
    g = np.c_[np.random.default_rng(0).uniform(-20, 20, (4000, 2)), np.full(4000, -1.73)]
    car = np.c_[np.random.default_rng(1).uniform([5, 5], [9, 7], (500, 2)),
                np.random.default_rng(2).uniform(-1.2, -0.2, 500)]
    m = ground_mask(np.r_[g, car])
    assert m[:4000].mean() > 0.99 and m[4000:].mean() < 0.01


@pytest.fixture(scope="module")
def scene():
    return make_sequence(n_frames=21, ego_speed=1.0, mover_speed=1.0)


def test_detects_crossing_car_without_flagging_static_scene(scene):
    scans, poses, ids = scene
    det = TemporalMotionDetector(scans, poses, CFG).detect(10)
    hid = ids[10]
    moving, parked, wall, ground = hid == 2, hid == 1, hid == 0, hid == -1
    assert moving.sum() > 100
    assert det.dynamic[moving].mean() > 0.8, "moving car must be detected"
    assert det.dynamic[parked].mean() < 0.02, "parked car must stay static"
    assert det.dynamic[wall].mean() < 0.01
    assert det.dynamic[ground].mean() < 0.001


def test_fully_static_world_yields_no_detections():
    scans, poses, ids = make_sequence(n_frames=21, ego_speed=1.0, mover_speed=0.0)
    det = TemporalMotionDetector(scans, poses, CFG).detect(10)
    assert det.dynamic.mean() < 0.001


def test_wrong_poses_break_static_consistency(scene):
    """Negative control: dropping ego motion makes the static world look dynamic."""
    scans, poses, ids = scene
    ident = {k: np.eye(4) for k in poses}
    det = TemporalMotionDetector(scans, ident, CFG).detect(10)
    assert det.candidate[ids[10] == 0].mean() > 0.3


def test_no_neighbours_means_no_votes(scene):
    scans, poses, _ = scene
    det = TemporalMotionDetector({10: scans[10]}, {10: poses[10]}, CFG).detect(10)
    assert det.neighbors_used == [] and not det.dynamic.any() and det.votes.max() == 0


def test_config_roundtrip():
    c = DetectorConfig.from_dict(CFG.to_dict())
    assert c == CFG
