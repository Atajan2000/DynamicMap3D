"""Assemble the development window in the standard SemanticKITTI layout.

  data/semantickitti/dataset/sequences/SS/velodyne/NNNNNN.bin  <- KITTI Odometry Velodyne zip (HTTP Range)
  data/semantickitti/dataset/sequences/SS/labels/NNNNNN.label  <- local SemanticKITTI label zip
  data/semantickitti/dataset/sequences/SS/poses.txt            <- local SemanticKITTI label zip
  data/semantickitti/dataset/sequences/SS/{calib,times}.txt    <- reused from LiDAR-MapLab
  data/semantickitti/dataset/poses/SS.txt                      <- reused from LiDAR-MapLab (KITTI GT)

Only the Odometry release's own scans are used: KITTI Raw scans differ geometrically (verified:
up to 0.63 m on seq 04 frame 0) and must never be substituted.
Existing files whose size and CRC-32 match the archive's central directory are not re-fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
import zlib
from pathlib import Path

import yaml

from dynamicmap3d.data.remote_zip import HttpRangeFile, fetch_member
from dynamicmap3d.data.splits import get_window

VELO_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_odometry_velodyne.zip"
LIDAR_MAPLAB = Path("../LiDAR-MapLab/data/kitti/odometry/dataset")


def crc_ok(path: Path, size: int, crc: int) -> bool:
    return path.exists() and path.stat().st_size == size and \
        (zlib.crc32(path.read_bytes()) & 0xFFFFFFFF) == crc


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=Path, default=Path("configs/splits.yaml"))
    ap.add_argument("--labels-zip", type=Path, default=Path("data/downloads/data_odometry_labels.zip"))
    ap.add_argument("--velodyne-cd", type=Path, default=Path("data/manifests/velodyne_central_directory.json"))
    ap.add_argument("--reuse-root", type=Path, default=LIDAR_MAPLAB)
    ap.add_argument("--root", type=Path, default=Path("data/semantickitti/dataset"))
    ap.add_argument("--max-bytes", type=int, default=230_000_000, help="abort if the window needs more")
    ap.add_argument("--window", default="development", help="development | eval0 | eval1 | eval2")
    a = ap.parse_args()

    w = get_window(a.window, a.splits)
    seq, first, last = w.sequence, w.first, w.last
    print(f"window {w.key}: seq {seq} frames {first}-{last} ({w.role})")
    sdir = a.root / "sequences" / seq
    (sdir / "velodyne").mkdir(parents=True, exist_ok=True)
    (sdir / "labels").mkdir(parents=True, exist_ok=True)
    (a.root / "poses").mkdir(parents=True, exist_ok=True)

    cd = {n: (c, u, crc, o) for n, c, u, crc, o in json.load(open(a.velodyne_cd))["members"]}
    frames = range(first, last + 1)
    members = [(i, cd[f"dataset/sequences/{seq}/velodyne/{i:06d}.bin"]) for i in frames]
    need = sum(c for _, (c, *_r) in members)
    if need > a.max_bytes:
        raise SystemExit(f"window needs {need:,} B > --max-bytes {a.max_bytes:,}")

    rf = HttpRangeFile(VELO_URL)
    skipped = 0
    hard_cap = need + 30 * len(members) + 4096   # exact header + payload; anything more aborts
    for i, (c, u, crc, off) in members:
        out = sdir / "velodyne" / f"{i:06d}.bin"
        if crc_ok(out, u, crc):
            skipped += 1
            continue
        if rf.bytes_fetched + 30 + c > hard_cap:
            raise SystemExit(f"transfer guard: {rf.bytes_fetched:,} B already fetched, cap {hard_cap:,} B")
        tmp = out.with_suffix(".part")
        tmp.write_bytes(fetch_member(rf, off, c, u, crc))
        tmp.replace(out)
        if (i - first) % 10 == 0:
            print(f"  scan {i:06d}  fetched so far {rf.bytes_fetched / 1e6:.1f} MB", flush=True)

    with zipfile.ZipFile(a.labels_zip) as zf:
        for i in frames:
            name = f"dataset/sequences/{seq}/labels/{i:06d}.label"
            info = zf.getinfo(name)
            out = sdir / "labels" / f"{i:06d}.label"
            if not crc_ok(out, info.file_size, info.CRC):
                out.write_bytes(zf.read(name))  # zipfile verifies CRC-32
        (sdir / "poses.txt").write_bytes(zf.read(f"dataset/sequences/{seq}/poses.txt"))

    reused = {}
    for src, dst in [(a.reuse_root / "sequences" / seq / "calib.txt", sdir / "calib.txt"),
                     (a.reuse_root / "sequences" / seq / "times.txt", sdir / "times.txt"),
                     (a.reuse_root / "poses" / f"{seq}.txt", a.root / "poses" / f"{seq}.txt")]:
        shutil.copyfile(src, dst)
        reused[str(dst)] = {"source": str(src.resolve()), "sha256": sha256(dst)}

    mpath = Path("data/manifests") / f"window_{seq}_{first:06d}_{last:06d}.json"
    history = json.loads(mpath.read_text()).get("transfer_log", []) if mpath.exists() else []
    history.append({"velodyne_bytes_transferred": rf.bytes_fetched, "http_range_requests": rf.requests,
                    "scans_fetched": len(members) - skipped, "scans_already_valid_skipped": skipped})
    manifest = {
        "sequence": seq, "first_frame": first, "last_frame": last,
        "velodyne_source": VELO_URL,
        "velodyne_bytes_required": need,
        "transfer_log": history,          # appended on every run, never overwritten
        "velodyne_bytes_transferred_total": sum(h["velodyne_bytes_transferred"] for h in history),
        "velodyne_members": {f"{i:06d}": {"size": u, "crc32": crc} for i, (c, u, crc, off) in members},
        "labels_source": str(a.labels_zip),
        "semantickitti_poses_source": f"{a.labels_zip}:dataset/sequences/{seq}/poses.txt",
        "reused_files": reused,
    }
    mpath.write_text(json.dumps(manifest, indent=2))
    print(f"done: {len(members)} scans ({skipped} already valid), transferred "
          f"{rf.bytes_fetched:,} B in {rf.requests} range requests -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
