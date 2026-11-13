# Dataset Report

Audit date 2026-11-06. Every number below was measured on the files on disk.
Machine-readable evidence: `data/manifests/*.json`, `results/label_stats/`, `results/validation/window_validation.json`.

## 1. Sources

| Component | Source | Bytes |
|---|---|---|
| SemanticKITTI labels + SuMa poses | `https://www.semantic-kitti.org/assets/data_odometry_labels.zip` (full archive) | 179,298,418 |
| Velodyne scans, seq 09 frames 1100–1199 | `data_odometry_velodyne.zip` (84.8 GB) via HTTP Range, 100 members only | 198,133,824 needed |
| `calib.txt`, `times.txt`, `poses/09.txt` (KITTI ground truth) | reused from `LiDAR-MapLab` (SHA-256 in the window manifest) | 0 new |
| Raw calibration `2011_09_30` | reused from `LiDAR-MapLab`, **calibration only** (cross-check of `Tr`) | 0 new |

Label archive integrity: exact size, CRC-32 verified for all 23,259 members, central directory identical
to the remote probe, SHA-256 `408ec524636a393bae0288a0b2f48bf5418a1af988e82dee8496f89ddb7e6dda`.

### Network transfer (datasets only)

| Item | Bytes |
|---|---|
| Pre-download probes (2 central directories + 1 seq-04 scan for the Raw/Odometry comparison) | 11,145,906 |
| Label archive | 179,298,418 |
| Development-window scans (**includes 104,857,600 B of overhead**) | 302,991,424 |
| **Total** | **493,435,748** |

The window fetch went over its budget. The first fetcher read each member's 30-byte local header
through a 1 MiB read-ahead cache, so each scan cost one extra MiB. This is fixed in `fetch_member`
(exact-size requests), and a regression test now asserts transfer = 30 B + payload. Without the bug,
the total would have been about 388.6 MB.

## 2. KITTI Raw ≠ KITTI Odometry

The local Raw scans of drive `2011_09_30_drive_0016` were compared with Odometry seq 04, frames 0–270.
Point counts are identical in all 271 frames, but no file is byte-identical. On frame 0, the point
order is the same (reflectance is equal at every index), yet x differs by up to 0.63 m. The
difference is linear in azimuth (0 m at the front, ±0.55 m at the rear seam), which means the two
releases apply different per-sweep motion compensation. SemanticKITTI labels belong to the Odometry
geometry, so Raw scans are never used here.

## 3. Label distributions (seqs 00–10, 23,201 frames)

`results/label_stats/sequence_summary.csv`. Distinct moving instance ids per sequence, for example:
08 has 32 cars, 40 bicyclists and 44 persons; 05 has 10 cars, 6 bicyclists and 10 persons; 04
(highway) has 23 cars and 1 person; 01 (highway) has 316 cars. Moving points are 0.08–0.93 % of all points.

## 4. Splits (frozen in `configs/splits.yaml` before the detector first ran)

Windows were scored on persistent movers (≥ 50 points in ≥ 10 frames), class diversity, temporal
coverage, parked-vehicle hard negatives and static structure. Raw moving-point count is not used.
Ego speed is limited to 1–12 m/s. At about 14.7 m/s (the first pick, seq 09 frames 1360–1459) scans
are 1.5 m apart and visibility evidence degrades. That pick was replaced by the practically tied
seq 09 frames 1100–1199 (9.3 m/s) before any algorithm ran.

| Role | Seq | Frames | Persistent movers | Moving pts | Ego m/s |
|---|---|---|---|---|---|
| **development** | 09 | 1100–1199 | 5 moving-car, 1 moving-person | 0.63 % | 9.3 |
| eval A | 08 | 0–99 | 4 car, 4 person, 2 bicyclist | 3.77 % | 4.7 |
| eval B | 08 | 1850–1949 | 1 car, 6 person, 2 bicyclist | 0.39 % | 6.7 |
| eval C | 07 | 600–699 | 5 car, 1 person, 1 truck | 2.68 % | 3.5 |

At window selection, the held-out scans had yet to be downloaded. They were fetched later for the evaluation described in §6.

## 5. Development-window validation (`scripts/validate_window.py`): 9/9 PASS

| Check | Result |
|---|---|
| Scan CRC-32/size vs Odometry archive | 100/100 match |
| Frames contiguous, scans = labels | 1100..1199, 100 + 100 |
| Label count = scan point count | 100/100 (102,813–127,278 pts; 12,383,364 total) |
| times / SemanticKITTI poses / KITTI GT poses | 1591 / 1591 / 1591 rows |
| Timestamps | monotonic, dt 0.1027–0.1052 s |
| `Tr` rigid and = `R_rect_00·[R|T]` from raw `2011_09_30` calib | max abs diff 4.5e-14 |
| Static-structure alignment, consecutive frames (median NN) | SemanticKITTI 0.044 m (worst 0.051), KITTI GT 0.045 m |
| Negative control, identity poses (poles/trunks/signs) | 0.892 m vs 0.044 m (20×) |
| Semantic control, moving points under correct poses | 2.11 m (they do not align, as expected) |

The first version of the negative control used all structure and gave only 4.7× (below its own 5×
threshold), because walls parallel to the driving direction alias under a wrong translation. The control
was moved to compact structures; the absolute alignment criterion was not changed.

## 6. Phase 2: held-out evaluation windows

The detector was frozen before these scans were downloaded (`configs/FROZEN_phase1.json`, SHA-256 of
the config and detector code).

| Window | Bytes transferred (= header + payload exactly) | Validation |
|---|---|---|
| seq 08, frames 0–99 | 196,201,688 | 9/9 PASS |
| seq 08, frames 1850–1949 | 196,543,000 | 9/9 PASS |
| seq 07, frames 600–699 | 196,192,664 | 9/9 PASS |
| **Total** | **588,937,352** (budget 650 MB) | |

Labels came from the local archive, and calibration, timestamps and poses were reused from LiDAR-MapLab.
Two validator flaws surfaced on the first validation run and were fixed without touching the detector:
- The completeness check assumed one window per sequence folder; seq 08 holds two.
- The identity-pose control compared consecutive frames, which are only 0.34 m apart at seq 07's
  ego speed (4.5×). It now uses a frame gap that gives ≥ 1 m of ego motion, keeping the 5× criterion.

All four windows pass all checks. Disk usage under `data/` is 1,118 MiB.
