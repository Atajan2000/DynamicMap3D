"""Validate the development window: integrity, frame ordering, scan/label correspondence,
calibration and pose alignment. Writes results/validation/window_validation.json.

Pose alignment test: static structure points (building, pole, trunk, sign, fence) of frame k are
transformed into frame k+1 with each pose source; the median nearest-neighbour distance to frame
k+1's structure points must be small. Identity poses serve as a negative control, and moving
points serve as a semantic control (they must NOT align as well as static points).
"""

from __future__ import annotations

import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial import cKDTree

from dynamicmap3d.data.kitti import read_calib, read_poses, read_times, scan_point_count, transform
from dynamicmap3d.data.semantickitti import STRUCTURE, is_moving
from dynamicmap3d.data.splits import all_windows, get_window
from dynamicmap3d.data.window import load_window, pose_file

# KITTI odometry devkit mapping: sequence -> (raw drive, first raw frame)
DEVKIT_MAP = {"00": ("2011_10_03_drive_0027", 0), "01": ("2011_10_03_drive_0042", 0),
              "02": ("2011_10_03_drive_0034", 0), "03": ("2011_09_26_drive_0067", 0),
              "04": ("2011_09_30_drive_0016", 0), "05": ("2011_09_30_drive_0018", 0),
              "06": ("2011_09_30_drive_0020", 0), "07": ("2011_09_30_drive_0027", 0),
              "08": ("2011_09_30_drive_0028", 1100), "09": ("2011_09_30_drive_0033", 0),
              "10": ("2011_09_30_drive_0034", 0)}
STRUCT_IDS = np.array(sorted(STRUCTURE), dtype=np.uint16)
# Compact vertical structures (trunk, pole, traffic-sign). Used for the negative control because
# planar structures parallel to the driving direction (walls, fences) alias under a wrong
# translation: a point slid along a wall still has a near neighbour on the same wall.
COMPACT_IDS = np.array([71, 80, 81], dtype=np.uint16)


def parse_kv(path: Path) -> dict[str, np.ndarray]:
    out = {}
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = np.array(v.split(), dtype=np.float64)
            except ValueError:
                pass
    return out


def check(results: dict, name: str, ok: bool, **info) -> None:
    results[name] = {"pass": bool(ok), **info}
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=Path("configs/splits.yaml"))
    ap.add_argument("--root", type=Path, default=Path("data/semantickitti/dataset"))
    ap.add_argument("--raw-calib-root", type=Path, default=Path("../LiDAR-MapLab/data/kitti/raw"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--window", default="development", help="development | eval0 | eval1 | eval2")
    a = ap.parse_args()
    w = get_window(a.window, a.splits)
    seq, first, last = w.sequence, w.first, w.last
    if a.out is None:
        a.out = Path("results/validation") / ("window_validation.json" if w.key == "development"
                                              else f"window_validation_{w.key}_{w.tag}.json")
    sdir = a.root / "sequences" / seq
    manifest = json.loads((Path("data/manifests") / f"window_{seq}_{first:06d}_{last:06d}.json").read_text())
    R: dict = {"window": w.key, "role": w.role, "sequence": seq, "first_frame": first, "last_frame": last}
    frames = list(range(first, last + 1))

    # 1. integrity: every scan matches the archive's size + CRC-32
    bad = [f for f, m in manifest["velodyne_members"].items()
           if (sdir / "velodyne" / f"{f}.bin").stat().st_size != m["size"]
           or (zlib.crc32((sdir / "velodyne" / f"{f}.bin").read_bytes()) & 0xFFFFFFFF) != m["crc32"]]
    check(R, "scan_crc32_matches_odometry_archive", not bad, scans=len(manifest["velodyne_members"]), bad=bad[:5])

    # 2. frame ordering / completeness
    # A sequence folder may hold several declared windows (e.g. two seq-08 eval windows): the window
    # must be complete, and every other file must belong to another declared window of that sequence.
    bins = sorted(int(p.stem) for p in (sdir / "velodyne").glob("*.bin"))
    labs = sorted(int(p.stem) for p in (sdir / "labels").glob("*.label"))
    declared = {f for ws in all_windows(a.splits).values() if ws.sequence == seq for f in ws.frames}
    in_win = lambda xs: [x for x in xs if first <= x <= last]
    stray = sorted((set(bins) | set(labs)) - declared)
    check(R, "frames_contiguous_and_complete",
          in_win(bins) == frames and in_win(labs) == frames and set(bins) == set(labs) and not stray,
          window_scans=len(in_win(bins)), window_labels=len(in_win(labs)), expected=f"{first}..{last}",
          files_of_other_declared_windows=len(set(bins) - set(frames)), undeclared_files=stray[:5])

    # 3. point-count correspondence
    mism = [i for i in frames if (sdir / "labels" / f"{i:06d}.label").stat().st_size % 4
            or (sdir / "labels" / f"{i:06d}.label").stat().st_size // 4 != scan_point_count(sdir / "velodyne" / f"{i:06d}.bin")]
    counts = [scan_point_count(sdir / "velodyne" / f"{i:06d}.bin") for i in frames]
    check(R, "label_point_count_equals_scan_point_count", not mism, mismatched=mism[:5],
          points_min=min(counts), points_max=max(counts), points_total=sum(counts))

    # 4. timestamps and pose files cover the whole sequence; window is uniformly sampled
    times = read_times(sdir / "times.txt")
    sk_poses = read_poses(pose_file(a.root, seq, "semantickitti"))
    gt_poses = read_poses(pose_file(a.root, seq, "kitti_gt"))
    n_seq = len(times)
    dt = np.diff(times[first:last + 1])
    check(R, "times_and_pose_lengths_consistent",
          len(sk_poses) == n_seq == len(gt_poses) and last < n_seq,
          times=n_seq, semantickitti_poses=len(sk_poses), kitti_gt_poses=len(gt_poses))
    check(R, "window_timestamps_monotonic_10hz", bool(np.all(dt > 0.08) and np.all(dt < 0.12)),
          dt_min=float(dt.min()), dt_max=float(dt.max()), dt_mean=float(dt.mean()))

    # 5. calibration: Tr is a rigid transform and equals R_rect_00 * [R|T]_velo->cam from raw calib
    calib = read_calib(sdir / "calib.txt")
    Tr = calib["Tr"]
    Rm = Tr[:3, :3]
    rigid = np.allclose(Rm @ Rm.T, np.eye(3), atol=1e-5) and abs(np.linalg.det(Rm) - 1) < 1e-5
    drive, raw_first = DEVKIT_MAP[seq]
    date = drive[:10]
    raw_dir = a.raw_calib_root / date
    if (raw_dir / "calib_velo_to_cam.txt").exists():
        v2c = parse_kv(raw_dir / "calib_velo_to_cam.txt")
        c2c = parse_kv(raw_dir / "calib_cam_to_cam.txt")
        T = np.eye(4); T[:3, :3] = v2c["R"].reshape(3, 3); T[:3, 3] = v2c["T"]
        Rr = np.eye(4); Rr[:3, :3] = c2c["R_rect_00"].reshape(3, 3)
        err = float(np.abs(Rr @ T - Tr).max())
        check(R, "calib_Tr_rigid_and_matches_raw_calibration", rigid and err < 1e-6,
              raw_calibration=f"{raw_dir}/calib_*.txt (calibration only; raw scans NOT used)",
              devkit_drive=drive, max_abs_diff=err)
    else:
        check(R, "calib_Tr_rigid", rigid, note=f"raw calibration for {date} not available locally")

    # 6. pose alignment on static structure, consecutive frames
    def residuals(poses, sel, ref_ids=STRUCT_IDS, gap=1):
        meds = []
        for k in frames[:-gap]:
            a_pts, b_pts = W.scans[k][:, :3], W.scans[k + gap][:, :3]
            ma, mb = sel(W.sem[k]), np.isin(W.sem[k + gap], ref_ids)
            if ma.sum() < 50 or mb.sum() < 50:
                continue
            Tba = np.linalg.inv(poses[k + gap]) @ poses[k]
            d, _ = cKDTree(b_pts[mb]).query(transform(a_pts[ma], Tba), distance_upper_bound=5.0)
            meds.append(np.median(np.minimum(d, 5.0)))
        return np.array(meds)

    W = load_window(a.root, seq, first, last, "semantickitti")
    W_gt = load_window(a.root, seq, first, last, "kitti_gt", with_labels=False)
    static = lambda s: np.isin(s, STRUCT_IDS)
    res = {"semantickitti": residuals(W.poses, static), "kitti_gt": residuals(W_gt.poses, static),
           "identity": residuals({i: np.eye(4) for i in frames}, static)}
    summary = {k: {"median_of_frame_medians_m": float(np.median(v)), "max_frame_median_m": float(v.max())}
               for k, v in res.items()}
    check(R, "pose_alignment_static_structure",
          summary["semantickitti"]["median_of_frame_medians_m"] < 0.10
          and summary["semantickitti"]["max_frame_median_m"] < 0.25, **summary)
    # Negative control strength scales with ego displacement between the compared scans, so frames
    # are compared at a gap giving >= 1 m of mean ego motion (gap 1 at 9 m/s, 3 at 3.5 m/s).
    compact = lambda s: np.isin(s, COMPACT_IDS)
    step = float(np.mean([np.linalg.norm(W.poses[k + 1][:3, 3] - W.poses[k][:3, 3]) for k in frames[:-1]]))
    gap = max(1, int(np.ceil(1.0 / max(step, 1e-3))))
    ctl = {k: float(np.median(residuals(P, compact, COMPACT_IDS, gap)))
           for k, P in [("semantickitti", W.poses), ("identity", {i: np.eye(4) for i in frames})]}
    check(R, "negative_control_identity_poses_misalign_compact_structures",
          ctl["identity"] > 5 * ctl["semantickitti"], median_of_frame_medians_m=ctl,
          frame_gap=gap, mean_ego_step_m=step,
          note=("control on all structure aliased along walls (dev: 4.7x) -> compact structures; "
                "consecutive frames too close at low ego speed (eval2: 4.5x) -> gap for >= 1 m ego motion"))

    # 7. semantic control: moving points must align worse than static ones under correct poses
    mov = residuals(W.poses, is_moving)
    check(R, "moving_points_misalign_under_correct_poses",
          len(mov) > 0 and np.median(mov) > 3 * summary["semantickitti"]["median_of_frame_medians_m"],
          moving_median_of_frame_medians_m=float(np.median(mov)) if len(mov) else None, frames=len(mov))

    # relative-motion agreement between pose sources
    rel = lambda P: np.linalg.norm(np.diff(P[first:last + 1, :3, 3], axis=0), axis=1)
    sk_step, gt_step = rel(sk_poses), rel(gt_poses)
    R["pose_sources_step_length_diff_m"] = {"mean_abs": float(np.abs(sk_step - gt_step).mean()),
                                           "max_abs": float(np.abs(sk_step - gt_step).max()),
                                           "mean_step_m": float(sk_step.mean())}
    print("pose step-length difference SemanticKITTI vs KITTI GT:", R["pose_sources_step_length_diff_m"])

    R["all_pass"] = all(v["pass"] for v in R.values() if isinstance(v, dict) and "pass" in v)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(R, indent=2))
    print("ALL PASS" if R["all_pass"] else "SOME CHECKS FAILED")
    return 0 if R["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
