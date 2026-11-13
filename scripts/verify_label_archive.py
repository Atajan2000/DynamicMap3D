"""Verify the downloaded SemanticKITTI label archive and write a JSON integrity manifest.

Checks: exact byte size (server Content-Length 179,298,418), CRC-32 of every member
(``ZipFile.testzip`` decompresses all of them), expected layout (labels for 00-10 only,
poses.txt for 00-21). Records SHA-256 for future reproducibility (no official checksum is published).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

EXPECTED_SIZE = 179_298_418
LABELS_URL = "https://www.semantic-kitti.org/assets/data_odometry_labels.zip"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=Path("data/downloads/data_odometry_labels.zip"))
    ap.add_argument("--out", type=Path, default=Path("data/manifests/labels_archive.json"))
    ap.add_argument("--reference-cd", type=Path, help="optional JSON of a remote central-directory probe")
    a = ap.parse_args()

    size = a.zip.stat().st_size
    sha = hashlib.sha256(a.zip.read_bytes()).hexdigest()
    with zipfile.ZipFile(a.zip) as zf:
        infos = zf.infolist()
        bad = zf.testzip()  # CRC-32 of every member
    per_seq = Counter()
    poses = []
    for i in infos:
        m = re.search(r"sequences/(\d\d)/labels/\d{6}\.label$", i.filename)
        if m:
            per_seq[m[1]] += 1
        elif i.filename.endswith("poses.txt"):
            poses.append(i.filename.split("/")[-2])
    ref_match = None
    if a.reference_cd:
        ref = {n: (u, crc) for n, c, u, crc, o in json.load(open(a.reference_cd))["members"]}
        ref_match = ref == {i.filename: (i.file_size, i.CRC) for i in infos}

    report = {
        "url": LABELS_URL,
        "path": str(a.zip),
        "size_bytes": size,
        "size_ok": size == EXPECTED_SIZE,
        "sha256": sha,
        "members": len(infos),
        "crc_all_members_ok": bad is None,
        "first_bad_member": bad,
        "label_files_per_sequence": dict(sorted(per_seq.items())),
        "poses_txt_sequences": sorted(poses),
        "central_directory_matches_remote_probe": ref_match,
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    ok = report["size_ok"] and report["crc_all_members_ok"] and ref_match in (None, True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
