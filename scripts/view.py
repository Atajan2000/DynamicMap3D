"""Open the DynamicMap3D Rerun viewer on a saved window (the detector is NOT rerun).

  python scripts/view.py --seq 07 --frames 600:700     # Example A: fast-motion filtering (held-out)
  python scripts/view.py --seq 08 --frames 0:100       # Example B: near-stationary objects (held-out)
  python scripts/view.py --seq 08 --frames 1850:1950   # Example C: slow pedestrians/cyclists (held-out)
  python scripts/view.py --seq 09 --frames 1100:1200   # development window (tuned)

--frames a:b is a Python-style range: frame a included, frame b EXCLUDED. A sub-range of a window is
allowed (window-level metrics still refer to the whole window). --save writes a .rrd instead of
spawning the viewer; open it later with `rerun <file>.rrd`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from dynamicmap3d.viz.data import load_view_data, parse_frames, resolve_window
from dynamicmap3d.viz.rerun_app import make_recording


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq", required=True, help="SemanticKITTI sequence, e.g. 07")
    ap.add_argument("--frames", required=True, help="a:b, end exclusive, e.g. 600:700")
    ap.add_argument("--save", type=Path, help="write a .rrd recording instead of spawning the viewer")
    ap.add_argument("--no-eval", action="store_true", help="hide all ground-truth evaluation layers")
    a = ap.parse_args(argv)
    first, last = parse_frames(a.frames)
    try:
        w = resolve_window(a.seq, first, last)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2
    t = time.perf_counter()
    d = load_view_data(w, first, last, with_labels=not a.no_eval)
    if a.save:
        a.save.parent.mkdir(parents=True, exist_ok=True)
    out = make_recording(d, save=str(a.save) if a.save else None)
    print(f"seq {w.sequence} frames {first}-{last} ({out['frames']} frames, {out['scan_points']:,} scan points, "
          f"raw/filtered map voxels {out['map_voxels']['raw_map']:,}/{out['map_voxels']['filtered_map']:,}) "
          f"logged in {time.perf_counter() - t:.0f} s" + (f" -> {a.save}" if a.save else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
