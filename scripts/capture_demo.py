#!/usr/bin/env python
"""Capture the README hero screenshot and the demo video from the real Rerun viewer (Windows).

    python scripts/capture_demo.py build        # .rrd recordings + capture layouts (saved predictions, no detector run)
    python scripts/capture_demo.py screenshot   # assets/demo/dynamicmap3d_overview.png
    python scripts/capture_demo.py video        # assets/demo/DynamicMap3D_Demo.mp4
    python scripts/capture_demo.py verify       # decode check, blank-frame check, contact sheet

Every 3D frame is a capture of the viewer window itself (Win32 PrintWindow via _viewer_capture.py;
other windows and the desktop cannot appear). The recordings are made by the same code as
``scripts/view.py --save`` (saved Phase 3 predictions, the detector is not rerun). Capture layouts only
re-arrange the viewer's existing views. All numbers in captions are read from results/phase2/.
Playback is the viewer's own 10 Hz replay (= the 10 Hz sensor rate); cuts between windows are labelled.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
DEMO = REPO / "results/demo"          # git-ignored work dir: .rrd, layouts, raw segments, verification
SEG = DEMO / "segments"
OUT = REPO / "assets/demo"
APP_ID = "dynamicmap3d"
FPS = 30
WINDOWS = {"eval2": ("07", 600, 700), "eval0": ("08", 0, 100), "eval1": ("08", 1850, 1950)}
PLAY = {"snapshot": dict(fps=1000.0, play_state="playing", loop_mode="off"),
        "paused": dict(fps=10.0, play_state="paused", loop_mode="off")}


# --------------------------------------------------------------------------------------------- build
def _layouts(pos, target) -> dict:
    import rerun.blueprint as rrb
    from dynamicmap3d.viz.rerun_app import BACKGROUND

    def v3(name, contents):
        return rrb.Spatial3DView(origin="world", name=name, contents=contents, background=BACKGROUND,
                                 line_grid=rrb.LineGrid3D(visible=False),
                                 eye_controls=rrb.EyeControls3D(kind="Orbital", position=pos, look_target=target,
                                                                eye_up=[0, 0, 1]))

    common = ["+ world/sensor", "+ world/trajectory/**"]
    A = lambda: v3("A · Raw LiDAR map (all points, 3-50 m, 0.2 m voxels)", ["+ world/raw_map/**"] + common)
    B = lambda: v3("B · Motion-aware static map (detector output; orange = removed)",
                   ["+ world/filtered_map/**", "+ world/removed/**"] + common)
    B_clean = lambda: v3("B · Motion-aware static map (detector output; removed points hidden)",
                         ["+ world/filtered_map/**"] + common)
    C = lambda: v3("C · Dynamic evidence: current scan (detector state)",
                   ["+ world/sensor/scan_detector", "+ world/dynamic_trail"] + common)
    D = lambda: v3("D · Evaluation overlay (ground truth, evaluation only)",
                   ["+ world/sensor/scan_eval", "+ world/eval_objects"] + common)

    def side():
        return rrb.Vertical(
            rrb.TextDocumentView(origin="diagnostics/frame", name="Diagnostics (per frame)"),
            rrb.TimeSeriesView(origin="plots/dynamic_fraction", name="Removed points per scan [%]",
                               plot_legend=rrb.PlotLegend(visible=False)),
            rrb.TimeSeriesView(origin="plots/detect_ms", name="Detection time [ms] (Phase 2 run)",
                               plot_legend=rrb.PlotLegend(visible=False)),
            rrb.Tabs(rrb.TextDocumentView(origin="info/window", name="Results (window-level)"),
                     rrb.TextDocumentView(origin="info/legend", name="Legend & method")),
            row_shares=[2.3, 0.55, 0.55, 2.0])

    def results():
        return rrb.TextDocumentView(origin="info/window", name="Results (window-level)")

    return {
        # identical to the viewer's default first tab
        "main": lambda: rrb.Horizontal(rrb.Grid(A(), B(), C(), D(), grid_columns=2), side(), column_shares=[3.2, 1.0]),
        # same, removed-point layer hidden in B (clean static map)
        "main_clean": lambda: rrb.Horizontal(rrb.Grid(A(), B_clean(), C(), D(), grid_columns=2), side(),
                                             column_shares=[3.2, 1.0]),
        # the two maps side by side, full height
        "compare": lambda: rrb.Horizontal(A(), B()),
        # the viewer's evaluation tabs
        "failure": lambda: rrb.Horizontal(
            v3("Failure analysis: filtered map + annotated moving geometry still in it (blue)",
               ["+ world/filtered_map/**", "+ world/eval_map/fn_kept/**", "+ world/eval_map/tp_removed/**",
                "+ world/eval_objects"] + common), results(), column_shares=[3.0, 1.2]),
        "static": lambda: rrb.Horizontal(
            v3("Static preservation: parked vehicles kept (aqua), static removed (white), moving removed (orange)",
               ["+ world/filtered_map/**", "+ world/eval_map/parked_kept/**", "+ world/eval_map/fp_removed/**",
                "+ world/eval_map/tp_removed/**"] + common), results(), column_shares=[3.0, 1.2]),
    }


def build() -> None:
    import rerun as rr
    import rerun.blueprint as rrb
    from dynamicmap3d.viz.data import load_view_data, resolve_window
    from dynamicmap3d.viz.rerun_app import log_window

    def make_recording(d, save):
        """Same logging as rerun_app.make_recording, but WITHOUT an embedded default blueprint:
        the large .rrd would otherwise load after the capture layout and replace it."""
        rec = rr.RecordingStream(application_id=APP_ID, recording_id=f"capture-{d.window.key}")
        rec.save(save)
        out = log_window(d, rec)
        rec.flush()
        return out

    DEMO.mkdir(parents=True, exist_ok=True)
    cams = {}
    for key, (seq, a, b) in WINDOWS.items():
        rrd = DEMO / f"{key}.rrd"
        w = resolve_window(seq, a, b - 1)
        t = time.perf_counter()
        d = load_view_data(w, a, b - 1, with_labels=True)
        out = make_recording(d, save=str(rrd))  # same call as scripts/view.py --save
        pos, target = out["camera"]
        cams[key] = {"position": pos, "target": target, "frames": out["frames"]}
        for name, make in _layouts(pos, target).items():
            for mode, play in PLAY.items():
                bp = rrb.Blueprint(make(), rrb.TimePanel(state="collapsed", timeline="frame", **play),
                                   rrb.BlueprintPanel(state="collapsed"), rrb.SelectionPanel(state="collapsed"),
                                   auto_layout=False, auto_views=False)
                p = DEMO / "layouts" / key / mode / f"{name}.rbl"
                p.parent.mkdir(parents=True, exist_ok=True)
                bp.save(APP_ID, str(p))
        print(f"{key}: seq {seq} {a}-{b - 1}, {out['frames']} frames -> {rrd} "
              f"({rrd.stat().st_size / 2**20:.0f} MB, {time.perf_counter() - t:.0f} s)")
    (DEMO / "cameras.json").write_text(json.dumps(cams, indent=2))


# ------------------------------------------------------------------------------------------- capture
class _session:
    """ViewerSession with the user's persisted app layout (APPDATA/rerun/.../dynamicmap3d.rbl) set aside.

    The native viewer restores that file and it overrides the layout given on the command line, so
    it is moved next to itself for the duration of one session and always moved back afterwards."""

    def __init__(self, key: str, layout: str, mode: str):
        import os
        self.args = (DEMO / f"{key}.rrd", DEMO / "layouts" / key / mode / f"{layout}.rbl")
        self.state = Path(os.environ["APPDATA"]) / "rerun/data/blueprints" / f"{APP_ID}.rbl"
        self.aside = self.state.with_name(f"{APP_ID}.rbl.capture-aside")

    def __enter__(self):
        from _viewer_capture import ViewerSession
        if self.aside.exists():
            raise RuntimeError(f"{self.aside} exists: restore it to {self.state.name} first")
        if self.state.exists():
            self.state.rename(self.aside)
        try:
            self.s = ViewerSession(*self.args)
        except BaseException:
            self._restore()
            raise
        return self.s

    def _restore(self):
        if self.aside.exists():
            self.state.unlink(missing_ok=True)  # anything the capture viewer wrote
            self.aside.rename(self.state)

    def __exit__(self, *exc):
        try:
            self.s.close()
        finally:
            self._restore()


def _wait_rendered(s, timeout: float = 60.0, settle: float = 2.0, min_wait: float = 9.0) -> None:
    """Wait until the window shows content and stops changing (recording fully loaded).
    ``min_wait``: the viewer shows an update-notification toast ~2.5-5.5 s after start (measured)."""
    from _viewer_capture import is_blank
    t0, prev, still = time.time(), None, None
    while time.time() - t0 < timeout:
        f = s.grab()[::4, ::4].astype(np.int16)
        if time.time() - t0 > min_wait and not is_blank(f) and prev is not None and np.abs(f - prev).mean() < 0.05:
            still = still or time.time()
            if time.time() - still >= settle:
                return
        else:
            still = None
        prev = f
        time.sleep(0.25)
    raise RuntimeError("viewer did not settle")


def _record(s, path: Path, seconds: float) -> int:
    """Wall-clock capture at FPS (repeat the latest frame if a grab is late), near-lossless intermediate."""
    from _ffmpeg_io import ffmpeg_exe
    g = s._grabber
    cmd = [ffmpeg_exe(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgra",
           "-s", f"{g.w}x{g.h}", "-framerate", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "ultrafast",
           "-crf", "10", "-pix_fmt", "yuv444p", str(path)]
    ff = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    n, written, t0 = int(seconds * FPS), 0, time.perf_counter()
    frame = g.grab_bgra().tobytes()
    while written < n:
        due = int((time.perf_counter() - t0) * FPS)
        if written <= due:
            frame = g.grab_bgra().tobytes()
        while written <= min(due, n - 1):
            ff.stdin.write(frame)
            written += 1
        time.sleep(max(0.0, written / FPS - (time.perf_counter() - t0)))
    ff.stdin.close()
    if ff.wait():
        raise RuntimeError("ffmpeg failed")
    return written


def _frames(path: Path):
    from _ffmpeg_io import read_frames
    yield from read_frames(path)


def capture_playback(key: str, layout: str, name: str, max_s: float = 25.0) -> Path:
    """One full 10 Hz playback pass from the first frame of the window."""
    path = SEG / f"{name}.mp4"
    with _session(key, layout, "paused") as s:
        _wait_rendered(s)
        threading.Timer(1.0, s.press_space).start()
        _record(s, path, max_s)
    return path


def capture_still(key: str, layout: str, name: str, seconds: float) -> Path:
    """Completed window (viewer ran to the last frame and stopped)."""
    path = SEG / f"{name}.mp4"
    with _session(key, layout, "snapshot") as s:
        _wait_rendered(s)
        _record(s, path, seconds)
    return path


TOP_BAR, TIME_STRIP = 36, 30  # viewer chrome (menu/search bar, collapsed time panel), px at 1920x1080


def screenshot() -> None:
    """Hero: completed seq 07 window in the viewer's default layout, cropped to the panels
    (menu bar and time strip removed; the strip shows the 1000 FPS jump-to-end setting)."""
    from PIL import Image
    from _viewer_capture import is_blank
    OUT.mkdir(parents=True, exist_ok=True)
    with _session("eval2", "main", "snapshot") as s:
        _wait_rendered(s)
        img = s.grab()
    if is_blank(img):
        raise RuntimeError("blank capture")
    p = OUT / "dynamicmap3d_overview.png"
    Image.fromarray(img[TOP_BAR:-TIME_STRIP]).save(p, optimize=True)
    print(f"hero -> {p} ({p.stat().st_size / 2**20:.2f} MB)")


# ------------------------------------------------------------------------------------------ assembly
def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    return ImageFont.truetype("segoeuib.ttf" if bold else "segoeui.ttf", size)


def _caption(lines: list[str], w: int = 1920, h: int = 1080) -> np.ndarray:
    """RGBA full-width caption band at the bottom. It covers the viewer's time strip (UI chrome)
    and the lowest part of the bottom views; it never covers the 3D content centre."""
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    f0, f1 = _font(26, True), _font(21)
    bh = 20 + 36 + 30 * (len(lines) - 1)
    y0 = h - bh - 4
    d.rectangle([0, y0, w, h], fill=(0, 0, 0, 225))
    d.line([0, y0, w, y0], fill=(240, 150, 60, 255), width=2)
    d.text((32, y0 + 10), lines[0], font=f0, fill=(255, 255, 255, 255))
    for i, t in enumerate(lines[1:]):
        d.text((32, y0 + 48 + 30 * i), t, font=f1, fill=(225, 225, 220, 255))
    return np.asarray(img)


def _title(title: str, sub: str, note: str, w: int = 1920, h: int = 1080) -> np.ndarray:
    from PIL import Image, ImageDraw
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    ft, fs, fn = _font(72, True), _font(34), _font(22)
    bw = max(d.textlength(title, font=ft), d.textlength(sub, font=fs), d.textlength(note, font=fn)) + 120
    x0, y0, bh = (w - bw) / 2, h / 2 - 130, 250
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=16, fill=(8, 8, 8, 225))
    d.text((w / 2, y0 + 70), title, font=ft, fill=(255, 255, 255, 255), anchor="mm")
    d.text((w / 2, y0 + 145), sub, font=fs, fill=(240, 150, 60, 255), anchor="mm")
    d.text((w / 2, y0 + 205), note, font=fn, fill=(200, 200, 195, 255), anchor="mm")
    return np.asarray(img)


def _blend(frame: np.ndarray, ov: np.ndarray, alpha_scale: float = 1.0) -> np.ndarray:
    a = ov[:, :, 3:4].astype(np.float32) / 255.0 * alpha_scale
    return (frame.astype(np.float32) * (1 - a) + ov[:, :, :3].astype(np.float32) * a).astype(np.uint8)


def _active_span(path: Path) -> tuple[int, int, int]:
    """(first moving frame, last moving frame, n) of a playback capture: playback start/end."""
    prev, diffs = None, []
    for f in _frames(path):
        g = f[::4, ::4].astype(np.int16)
        diffs.append(0.0 if prev is None else float(np.abs(g - prev).mean()))
        prev = g
    moving = [i for i, v in enumerate(diffs) if v > 0.02]
    if not moving or moving[-1] >= len(diffs) - FPS // 2:
        raise RuntimeError(f"{path.name}: playback did not finish inside the capture")
    return moving[0], moving[-1], len(diffs)


def summary_card(seconds: float) -> np.ndarray:
    """Evaluation summary, every number read from results/phase2/summary.json; real Phase 2 figure."""
    from PIL import Image, ImageDraw
    S = json.loads((REPO / "results/phase2/summary.json").read_text().replace("NaN", "null"))
    pm = S["point_metrics"]["temporal_detector"]
    pool = pm["heldout_pooled"]["all_frames"]
    ret = S["mapping"]["temporal_detector"]["heldout_pooled"]["static_point_retention"]
    img = Image.new("RGB", (1920, 1080), (13, 13, 13))
    d = ImageDraw.Draw(img)
    d.text((80, 60), "Evaluation summary", font=_font(54, True), fill=(255, 255, 255))
    d.text((80, 135), "3 held-out SemanticKITTI windows (custom windows, not an official benchmark) · frozen detector",
           font=_font(26), fill=(200, 200, 195))
    rows = [("Pooled held-out moving IoU", f"{pool['iou_moving']:.3f}"),
            ("Precision (moving)", f"{100 * pool['precision']:.1f}%"),
            ("Recall (moving)", f"{100 * pool['recall']:.1f}%"),
            ("Static-point retention (map, 3-50 m)", f"{100 * ret:.2f}%")]
    y = 215
    for k, v in rows:
        d.text((80, y), k, font=_font(30), fill=(225, 225, 220))
        d.text((700, y), v, font=_font(30, True), fill=(240, 150, 60))
        y += 52
    y += 14
    d.text((80, y), "Per window (moving IoU)", font=_font(26, True), fill=(225, 225, 220))
    y += 44
    for key, label in (("eval2", "seq 07, 600-699: fast vehicles"), ("eval0", "seq 08, 0-99: near-stationary vehicles"),
                       ("eval1", "seq 08, 1850-1949: slow pedestrians / cyclists")):
        d.text((100, y), label, font=_font(24), fill=(200, 200, 195))
        d.text((700, y), f"{pm[key]['all_frames']['iou_moving']:.3f}", font=_font(24, True), fill=(225, 225, 220))
        y += 38
    fig = Image.open(REPO / "results/phase2/figures/point_metrics_by_window.png").convert("RGB")
    fw = 960
    fig = fig.resize((fw, int(fig.height * fw / fig.width)), Image.LANCZOS)
    img.paste(fig, (1920 - fw - 60, 215))
    d.text((1920 - fw - 60, 215 + fig.height + 10), "results/phase2/figures/point_metrics_by_window.png",
           font=_font(20), fill=(150, 150, 145))
    y = max(y + 26, 215 + fig.height + 60)
    d.text((80, y), "Method and limits", font=_font(26, True), fill=(225, 225, 220))
    y += 44
    for t in ("Temporal geometric consistency: free-space test against scans at ±3, ±6, ±10 frames.",
              "Uses supplied KITTI poses and future scans (up to 1.0 s): offline and non-causal.",
              "Not SLAM and not a real-time online detector. Semantic labels never used for inference.",
              "Struggles with nearly stationary objects (< 0.5 m/s): they stay in the static map."):
        d.text((100, y), "· " + t, font=_font(23), fill=(200, 200, 195))
        y += 36
    return np.asarray(img)


def video() -> None:
    from _ffmpeg_io import Writer
    SEG.mkdir(parents=True, exist_ok=True)
    S = json.loads((REPO / "results/phase2/summary.json").read_text().replace("NaN", "null"))
    iou = {k: S["point_metrics"]["temporal_detector"][k]["all_frames"]["iou_moving"] for k in WINDOWS}
    M = {k: json.loads((REPO / f"results/phase2/windows/{k}/metrics.json").read_text().replace("NaN", "null"))
         for k in WINDOWS}
    pv = M["eval2"]["objects"]["temporal_detector"]["parked_vehicles"]
    mp = M["eval2"]["mapping"]
    raw_c, det_c = mp["raw_unfiltered"]["contaminated_voxel_fraction"], mp["temporal_detector"]["contaminated_voxel_fraction"]

    # (segment, kind, seconds or None for playback, caption lines)
    scenes = [
        (("eval2", "main"), "still", 7.0, None),
        (("eval2", "main"), "play", None,
         [f"1  Moving-object removal · held-out SemanticKITTI seq 07, frames 600-699 (window moving IoU {iou['eval2']:.3f})",
          "Viewer replay of saved detector outputs: {replay}. Maps accumulate over time.",
          "A raw map · B filtered static map, orange = removed · C current scan, detector state · D ground truth (evaluation only)"]),
        (("eval2", "compare"), "still", 8.0,
         ["2  Completed maps after 100 scans: raw (left) vs motion-filtered static map (right)",
          f"Map voxels holding annotated moving points: {100 * raw_c:.1f}% raw -> {100 * det_c:.1f}% filtered (seq 07 window)"]),
        (("eval2", "static"), "still", 8.0,
         ["3  Static preservation (evaluation layers, ground truth used only for scoring)",
          f"Parked vehicles kept (aqua): {pv['object_frames_falsely_removed_ge50pct']} of {pv['object_frames']} parked-vehicle "
          f"object-frames lost >= 50% of points · static removed shown in white (not zero)"]),
        (("eval0", "main"), "play", None,
         [f"4  Failure case · held-out seq 08, frames 0-99 (window moving IoU {iou['eval0']:.3f})",
          "The largest annotated moving cars move < 0.5 m/s: a ±1 s free-space test cannot separate them from parked cars ({replay}).",
          "Compare C (detector) with D (ground truth, evaluation only): most moving points are not detected."]),
        (("eval0", "failure"), "still", 7.0,
         ["5  Missed moving geometry left in the filtered map (blue = false negatives, evaluation only)",
          "seq 08, frames 0-99, completed window. Orange = moving points correctly removed."]),
        (("eval1", "failure"), "still", 6.0,
         [f"6  Failure case · held-out seq 08, frames 1850-1949 (window moving IoU {iou['eval1']:.3f})",
          "Slow pedestrians and cyclists (mostly 0.2-1.6 m/s) are largely missed (blue, evaluation only)."]),
    ]
    raw = []
    for k, ((key, layout), kind, secs, cap) in enumerate(scenes):
        name = f"{k}_{key}_{layout}_{kind}"
        p = SEG / f"{name}.mp4"
        if not p.is_file():
            print("capturing", name)
            p = capture_playback(key, layout, name) if kind == "play" else capture_still(key, layout, name, secs + 1.0)
        raw.append(p)

    out = OUT / "DynamicMap3D_Demo.mp4"
    OUT.mkdir(parents=True, exist_ok=True)
    w = Writer(out, 1920, 1080, fps=FPS, crf=20)
    log = []
    try:
        for (seg, kind, secs, cap), p in zip(scenes, raw):
            if kind == "play":
                a, b, n = _active_span(p)
                lo, hi = max(0, a - FPS // 2), min(n - 1, b + int(1.5 * FPS))  # 0.5 s before start, 1.5 s hold at end
                dur = (b - a) / FPS  # measured: the viewer is rendering-bound while captured, slower than 10 Hz
                replay = f"100 scans (10 s of driving) shown in {dur:.0f} s, about {10.0 / dur:.1f}x real time"
                cap = [c.replace("{replay}", replay) for c in cap]
            else:
                lo, hi = FPS // 2, FPS // 2 + int(secs * FPS) - 1
            ov = _caption(cap) if cap else None
            title = None
            if cap is None:
                title = _title("DynamicMap3D", "Motion-Aware LiDAR Mapping in Dynamic Environments",
                               "Real Rerun viewer · SemanticKITTI seq 07, frames 600-699 · saved detector outputs")
            count = 0
            for i, f in enumerate(_frames(p)):
                if i < lo:
                    continue
                if i > hi:
                    break
                if title is not None:
                    f = f.copy()
                    f[-TIME_STRIP:] = 13  # hide the time strip (UI chrome showing the 1000 FPS jump-to-end setting)
                    t = (i - lo) / FPS
                    f = _blend(f, title, min(1.0, t / 0.6, max(0.0, (secs - t) / 0.6)) if secs > 1.2 else 1.0)
                if ov is not None:
                    f = _blend(f, ov)
                w.append(f)
                count += 1
            log.append({"segment": p.name, "frames": count, "seconds": round(count / FPS, 2), "caption": cap})
        card = summary_card(10.0)
        for _ in range(int(11 * FPS)):
            w.append(card)
        log.append({"segment": "summary card (results/phase2)", "frames": int(11 * FPS), "seconds": 11.0})
    finally:
        w.close()
    (DEMO / "video_segments.json").write_text(json.dumps(log, indent=2))
    print(json.dumps(log, indent=2))
    print(f"video -> {out} ({out.stat().st_size / 2**20:.1f} MB, {sum(x['seconds'] for x in log):.1f} s)")


def verify() -> None:
    from PIL import Image
    from _viewer_capture import is_blank
    from _ffmpeg_io import probe
    out = OUT / "DynamicMap3D_Demo.mp4"
    meta = probe(out)
    thumbs, blanks, n = [], [], 0
    for i, f in enumerate(_frames(out)):
        n += 1
        if f.mean() < 3:
            blanks.append(i)
        if i % (2 * FPS) == 0:
            thumbs.append(Image.fromarray(f).resize((384, 216)))
            if is_blank(f):
                blanks.append(i)
    sheet = Image.new("RGB", (384 * 6, 216 * ((len(thumbs) + 5) // 6)))
    for i, t in enumerate(thumbs):
        sheet.paste(t, (384 * (i % 6), 216 * (i // 6)))
    (DEMO / "verify").mkdir(parents=True, exist_ok=True)
    sheet.save(DEMO / "verify/video_contact_sheet.png")
    rep = {"file": str(out), "decoded_frames": n, "fps": meta["fps"], "size": [meta["width"], meta["height"]],
           "duration_s": round(n / meta["fps"], 2), "container_duration_s": meta["duration_s"],
           "size_mb": round(out.stat().st_size / 2**20, 1), "codec": meta["codec"],
           "black_or_blank_frames": blanks[:50]}
    (DEMO / "verify/verification.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "screenshot", "video", "verify"])
    {"build": build, "screenshot": screenshot, "video": video, "verify": verify}[ap.parse_args().step]()
