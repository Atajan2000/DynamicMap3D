import io
import zipfile

import numpy as np
import pytest

from dynamicmap3d.data.kitti import cam_to_velo_poses, parse_poses, read_calib, read_scan, transform
from dynamicmap3d.data.remote_zip import fetch_member
from dynamicmap3d.data.semantickitti import decode_labels, is_ignored, is_moving
from dynamicmap3d.evaluation.metrics import Confusion, Evaluator


def test_label_bit_layout():
    raw = np.array([(7 << 16) | 252, (3 << 16) | 10, 0, 254], dtype=np.uint32)
    sem, inst = decode_labels(raw)
    assert sem.tolist() == [252, 10, 0, 254] and inst.tolist() == [7, 3, 0, 0]
    assert is_moving(sem).tolist() == [True, False, False, True]
    assert is_ignored(sem).tolist() == [False, False, True, False]


def test_read_scan_rejects_truncated_file(tmp_path):
    p = tmp_path / "a.bin"
    np.arange(8, dtype=np.float32).tofile(p)
    assert read_scan(p).shape == (2, 4)
    p.write_bytes(b"\x00" * 20)
    with pytest.raises(ValueError):
        read_scan(p)


def test_calib_and_velo_pose_conversion(tmp_path):
    rng = np.random.default_rng(0)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))
    Tr = np.eye(4); Tr[:3, :3] = q; Tr[:3, 3] = [0.1, -0.2, 0.3]
    (tmp_path / "calib.txt").write_text("P0: " + " ".join(["1"] * 12) + "\nTr: " +
                                        " ".join(map(str, Tr[:3].ravel())) + "\n")
    assert np.allclose(read_calib(tmp_path / "calib.txt")["Tr"], Tr)
    # a velodyne-frame motion expressed in the camera frame must convert back exactly
    Tv = np.eye(4); Tv[:3, 3] = [2.0, 1.0, 0.0]
    Pc = Tr @ Tv @ np.linalg.inv(Tr)
    assert np.allclose(cam_to_velo_poses(Pc[None], Tr)[0], Tv)
    poses = parse_poses(" ".join(map(str, Pc[:3].ravel())))
    assert np.allclose(poses[0], Pc)
    assert np.allclose(transform(np.zeros((1, 3)), Tv), [[2, 1, 0]])


class _Seekable(io.BytesIO):
    pass


@pytest.mark.parametrize("method", [zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED])
def test_fetch_member_by_cached_offsets(method):
    buf = io.BytesIO()
    payload = np.arange(1000, dtype=np.float32).tobytes()
    with zipfile.ZipFile(buf, "w", compression=method) as zf:
        zf.writestr("a/other.txt", b"hello")
        zf.writestr("a/000001.bin", payload)
    info = zipfile.ZipFile(io.BytesIO(buf.getvalue())).getinfo("a/000001.bin")
    rf = _Seekable(buf.getvalue())
    assert fetch_member(rf, info.header_offset, info.compress_size, info.file_size, info.CRC) == payload
    with pytest.raises(zipfile.BadZipFile):
        fetch_member(rf, info.header_offset, info.compress_size, info.file_size, info.CRC ^ 1)


def test_fetch_member_transfers_no_read_ahead():
    """Regression: bytes transferred must equal header + payload (no 1 MiB block read-ahead)."""
    from dynamicmap3d.data.remote_zip import HttpRangeFile

    buf = io.BytesIO()
    payload = np.random.default_rng(0).bytes(3_000_000)
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("x/000000.bin", payload)
        zf.writestr("x/000001.bin", payload[::-1])
    raw = buf.getvalue()
    info = zipfile.ZipFile(io.BytesIO(raw)).getinfo("x/000001.bin")

    rf = HttpRangeFile.__new__(HttpRangeFile)  # no network: serve ranges from memory
    rf.url, rf.block, rf.max_blocks, rf.pos, rf.size = "mem://", 1 << 20, 16, 0, len(raw)
    rf.bytes_fetched = rf.requests = 0
    rf._cache = {}

    def fake_get(start, end, retries=8):
        rf.requests += 1
        rf.bytes_fetched += end - start + 1
        return raw[start:end + 1]

    rf._get = fake_get
    assert fetch_member(rf, info.header_offset, info.compress_size, info.file_size, info.CRC) == payload[::-1]
    assert rf.bytes_fetched == 30 + info.compress_size   # name/extra fields are skipped, not fetched
    assert rf.requests == 2


def test_confusion_and_object_metrics():
    c = Confusion()
    c.add(np.array([1, 1, 0, 0], bool), np.array([1, 0, 1, 0], bool))
    s = c.summary()
    assert (s["tp"], s["fp"], s["fn"], s["tn"]) == (1, 1, 1, 1) and s["iou_moving"] == pytest.approx(1 / 3)
    ev = Evaluator(min_object_points=2)
    sem = np.array([252] * 4 + [254] * 2 + [10, 0], np.uint16)
    inst = np.array([5] * 4 + [6] * 2 + [0, 0], np.uint16)
    pred = np.array([1, 1, 1, 0, 0, 0, 1, 1], bool)
    ev.add(0, pred, sem, inst, np.ones(8, bool))
    s = ev.all_points.summary()
    assert (s["tp"], s["fp"], s["fn"]) == (3, 1, 3)      # unlabeled point (class 0) ignored
    o = ev.object_summary()
    assert o["object_frames"] == 2 and o["object_frames_detected"] == 1
    assert dict(ev.fp_by_class) == {10: 1}
