"""Rerun recording + reusable blueprint for DynamicMap3D (rerun-sdk 0.41).

Entity layout (display world = sensor frame of the first shown scan, z up):
  world/sensor                     sensor pose per frame (Transform3D) + axes
  world/sensor/scan_detector       current scan, class ids = detector state     (labels never used)
  world/sensor/scan_eval           current scan, class ids = outcome vs. GT     (evaluation only)
  world/raw_map/fNNNNNN            raw map voxels first seen at frame N
  world/filtered_map/fNNNNNN       detector-filtered map voxels first seen at frame N
  world/removed/fNNNNNN            voxels of points removed by the detector
  world/eval_map/<layer>/fNNNNNN   evaluation-only accumulations (tp_removed, fn_kept, fp_removed, parked_kept)
  world/dynamic_trail              detector removals of the previous 10 frames
  world/eval_objects               evaluation-only labels on annotated moving objects
  world/trajectory/{full,travelled}
  diagnostics/frame, info/*, plots/*
"""

from __future__ import annotations

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from dynamicmap3d.data.kitti import transform
from dynamicmap3d.viz.data import (BLUE, DETECTOR_CLASSES, EVAL_CLASSES, ORANGE, AQUA, WHITE, ViewData,
                                   detector_classes, evaluation_classes, evaluation_masks, frame_diagnostics,
                                   map_increments, map_masks, object_annotations)
from dynamicmap3d.viz.text import frame_markdown, legend_markdown, window_markdown

BACKGROUND = [13, 13, 13]
TRAIL = 10
EVAL_LAYER_COLORS = {"tp_removed": ORANGE, "fn_kept": BLUE, "fp_removed": WHITE, "parked_kept": AQUA}


def height_colors(z: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Neutral grey ramp by height, identical scale for raw and filtered maps."""
    t = np.clip((z - lo) / max(hi - lo, 1e-6), 0, 1)
    g = (70 + 165 * t).astype(np.uint8)
    return np.stack([g, g, (g * 0.97).astype(np.uint8)], 1)


def camera(d: ViewData, raw_inc: dict) -> tuple[list, list]:
    """One fixed oblique viewpoint, framed on the sensor trajectory (shared by every 3D view)."""
    traj = np.array([d.poses_display[f][:3, 3] for f in d.frames])
    target = np.r_[(traj.min(0)[:2] + traj.max(0)[:2]) / 2, traj[:, 2].mean() - 1.5]
    span = float(np.max(traj.max(0)[:2] - traj.min(0)[:2]))
    direction = np.array([-0.55, -0.65, 0.75])
    pos = target + direction / np.linalg.norm(direction) * (0.6 * span + 32.0)
    return pos.tolist(), target.tolist()


def blueprint(eye_pos, eye_target, has_eval: bool) -> rrb.Blueprint:
    def eye():
        return rrb.EyeControls3D(kind="Orbital", position=eye_pos, look_target=eye_target, eye_up=[0, 0, 1])

    def v3(name, contents):
        return rrb.Spatial3DView(origin="world", name=name, contents=contents, background=BACKGROUND,
                                 line_grid=rrb.LineGrid3D(visible=False), eye_controls=eye())

    common = ["+ world/sensor", "+ world/trajectory/**"]
    A = v3("A · Raw LiDAR map (all points, 3-50 m, 0.2 m voxels)", ["+ world/raw_map/**"] + common)
    B = v3("B · Motion-aware static map (detector output; orange = removed)",
           ["+ world/filtered_map/**", "+ world/removed/**"] + common)
    C = v3("C · Dynamic evidence: current scan (detector state)",
           ["+ world/sensor/scan_detector", "+ world/dynamic_trail"] + common)
    D = v3("D · Evaluation overlay (ground truth, evaluation only)",
           ["+ world/sensor/scan_eval", "+ world/eval_objects"] + common)
    side = rrb.Vertical(
        rrb.TextDocumentView(origin="diagnostics/frame", name="Diagnostics (per frame)"),
        rrb.TimeSeriesView(origin="plots/dynamic_fraction", name="Removed points per scan [%]",
                           plot_legend=rrb.PlotLegend(visible=False)),
        rrb.TimeSeriesView(origin="plots/detect_ms", name="Detection time [ms] (Phase 2 run)",
                           plot_legend=rrb.PlotLegend(visible=False)),
        rrb.Tabs(rrb.TextDocumentView(origin="info/window", name="Results (window-level)"),
                 rrb.TextDocumentView(origin="info/legend", name="Legend & method")),
        row_shares=[2.3, 0.55, 0.55, 2.0])
    main = rrb.Horizontal(rrb.Grid(A, B, C, D if has_eval else C, grid_columns=2), side,
                          column_shares=[3.2, 1.0], name="Raw vs filtered map")
    tabs = [main]
    if has_eval:
        tabs.append(rrb.Horizontal(
            v3("Failure analysis: filtered map + annotated moving geometry still in it (blue)",
               ["+ world/filtered_map/**", "+ world/eval_map/fn_kept/**", "+ world/eval_map/tp_removed/**",
                "+ world/eval_objects"] + common),
            rrb.TextDocumentView(origin="info/window", name="Results (window-level)"),
            column_shares=[3.0, 1.2], name="Failure analysis (evaluation)"))
        tabs.append(rrb.Horizontal(
            v3("Static preservation: parked vehicles kept (aqua), static removed (white), moving removed (orange)",
               ["+ world/filtered_map/**", "+ world/eval_map/parked_kept/**", "+ world/eval_map/fp_removed/**",
                "+ world/eval_map/tp_removed/**"] + common),
            rrb.TextDocumentView(origin="info/window", name="Results (window-level)"),
            column_shares=[3.0, 1.2], name="Static preservation (evaluation)"))
    return rrb.Blueprint(rrb.Tabs(*tabs, active_tab=0),
                         rrb.TimePanel(state="collapsed", timeline="frame", fps=10.0, play_state="playing",
                                       loop_mode="all"),
                         rrb.BlueprintPanel(state="collapsed"), rrb.SelectionPanel(state="collapsed"),
                         auto_layout=False, auto_views=False)


def log_window(d: ViewData, rec: rr.RecordingStream) -> dict:
    """Log the whole window. Returns counts used by tests."""
    has_eval = bool(d.sem)
    maps, totals = map_increments(d, map_masks(d))
    ev, ev_totals = map_increments(d, evaluation_masks(d)) if has_eval else ({}, {})
    zs = np.concatenate([p[:, 2] for p in maps["raw_map"].values() if len(p)])
    lo, hi = np.percentile(zs, [2, 98])

    rec.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rec.log("world/sensor", rr.TransformAxes3D(axis_length=3.0), static=True)
    rec.log("world/sensor/scan_detector", rr.AnnotationContext(
        [rr.AnnotationInfo(id=i, label=lab, color=c) for i, (lab, c) in DETECTOR_CLASSES.items()]), static=True)
    if has_eval:
        rec.log("world/sensor/scan_eval", rr.AnnotationContext(
            [rr.AnnotationInfo(id=i, label=lab, color=c) for i, (lab, c) in EVAL_CLASSES.items()]), static=True)
    traj = np.array([d.poses_display[f][:3, 3] for f in d.frames])
    rec.log("world/trajectory/full", rr.LineStrips3D([traj], colors=[[110, 110, 105]], radii=-1.0), static=True)
    rec.log("info/legend", rr.TextDocument(legend_markdown(), media_type=rr.MediaType.MARKDOWN), static=True)
    rec.log("info/window", rr.TextDocument(window_markdown(d), media_type=rr.MediaType.MARKDOWN), static=True)
    rec.log("plots/dynamic_fraction", rr.SeriesLines(colors=[ORANGE], names=["removed %"]), static=True)
    rec.log("plots/detect_ms", rr.SeriesLines(colors=[[200, 200, 195]], names=["ms"]), static=True)

    logged = {"scan_points": 0, "frames": 0}
    for i, f in enumerate(d.frames):
        rec.set_time("frame", sequence=f)
        T = d.poses_display[f]
        rec.log("world/sensor", rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]))
        xyz = d.scans[f][:, :3]
        dyn = d.pred["dynamic"][f]
        rec.log("world/sensor/scan_detector", rr.Points3D(
            xyz, class_ids=detector_classes(dyn, d.pred["candidate"][f], d.pred["valid"][f]), radii=-1.3))
        logged["scan_points"] += len(xyz)
        if has_eval:
            rec.log("world/sensor/scan_eval", rr.Points3D(xyz, class_ids=evaluation_classes(dyn, d.sem[f]), radii=-1.3))
            ann = object_annotations(d, f)
            if ann:
                rec.log("world/eval_objects", rr.Points3D(
                    [a["position"] for a in ann], labels=[a["label"] for a in ann], show_labels=True,
                    colors=[BLUE if a["missed"] else ORANGE for a in ann], radii=-7.0))
            else:
                rec.log("world/eval_objects", rr.Clear(recursive=False))
            for layer, inc in ev.items():
                if len(inc[f]):
                    rec.log(f"world/eval_map/{layer}/f{f:06d}",
                            rr.Points3D(inc[f], colors=[EVAL_LAYER_COLORS[layer]], radii=-2.0))
        for layer in ("raw_map", "filtered_map"):
            p = maps[layer][f]
            if len(p):
                rec.log(f"world/{layer}/f{f:06d}", rr.Points3D(p, colors=height_colors(p[:, 2], lo, hi), radii=-1.2))
        if len(maps["removed"][f]):
            rec.log(f"world/removed/f{f:06d}", rr.Points3D(maps["removed"][f], colors=[ORANGE], radii=-1.8))
        prev = [transform(d.scans[g][d.pred["dynamic"][g], :3].astype(np.float64), d.poses_display[g])
                for g in d.frames[max(0, i - TRAIL):i]]
        trail = np.concatenate(prev) if prev else np.empty((0, 3))
        rec.log("world/dynamic_trail", rr.Points3D(trail, colors=[[*ORANGE, 90]], radii=-1.0))
        rec.log("world/trajectory/travelled", rr.LineStrips3D([traj[: i + 1]], colors=[[235, 235, 230]], radii=-2.0))
        diag = frame_diagnostics(d, f)
        rec.log("diagnostics/frame", rr.TextDocument(frame_markdown(d, diag), media_type=rr.MediaType.MARKDOWN))
        rec.log("plots/dynamic_fraction", rr.Scalars(100 * diag["dynamic_fraction"]))
        rec.log("plots/detect_ms", rr.Scalars(diag["detect_ms_phase2"]))
        logged["frames"] += 1
    logged.update({"map_voxels": totals, "eval_voxels": ev_totals, "camera": camera(d, maps["raw_map"])})
    return logged


def make_recording(d: ViewData, save: str | None = None, spawn: bool = True) -> dict:
    rec = rr.RecordingStream(application_id="dynamicmap3d",
                             recording_id=f"{d.window.key}-{d.frames[0]}-{d.frames[-1]}")
    maps, _ = map_increments(d, {"raw_map": map_masks(d)["raw_map"]})
    pos, target = camera(d, maps["raw_map"])
    bp = blueprint(pos, target, bool(d.sem))
    if save:
        rec.save(save, default_blueprint=bp)
    elif spawn:
        rec.spawn(default_blueprint=bp)
    out = log_window(d, rec)
    rec.flush()
    return out
