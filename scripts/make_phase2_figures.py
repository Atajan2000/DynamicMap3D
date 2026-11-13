"""Three Phase 2 evaluation figures from saved results (no detector runs).

Palette: reference categorical slots 1-4 (validated light mode; aqua/yellow < 3:1 contrast,
so every bar carries a value label and every chart a legend).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
OUT = Path("results/phase2/figures")
plt.rcParams.update({"figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "axes.edgecolor": GRID,
                     "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                     "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8,
                     "axes.axisbelow": True, "legend.frameon": False})


def bars(ax, groups, series, values, fmt="{:.2f}", ymax=1.0):
    n = len(series)
    width = 0.8 / n
    for j, (name, color) in enumerate(zip(series, SLOTS)):
        xs = [i + (j - (n - 1) / 2) * width for i in range(len(groups))]
        vals = [values[g][j] for g in groups]
        drawn = [0 if v is None else v for v in vals]
        b = ax.bar(xs, drawn, width * 0.92, color=color, label=name, linewidth=0)
        for x, v in zip(xs, vals):
            ax.text(x, (0 if v is None else v) + ymax * 0.012, "n/a" if v is None else fmt.format(v),
                    ha="center", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(range(len(groups)))
    ax.set_ylim(0, ymax * 1.12)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    S = json.loads(Path("results/phase2/summary.json").read_text())
    W = S["windows"]

    # 1. point metrics per window ------------------------------------------------------------
    keys = ["development", "eval0", "eval1", "eval2", "heldout_pooled"]
    labels = {k: f"{'DEV ' if k == 'development' else ''}seq {W[k]['sequence']}\n{W[k]['frames'][0]}-{W[k]['frames'][1]}"
              for k in keys[:-1]}
    labels["heldout_pooled"] = "held-out\npooled"
    pm = S["point_metrics"]["temporal_detector"]
    vals = {labels[k]: [pm[k]["all_frames"][m] for m in ("iou_moving", "precision", "recall")] for k in keys}
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bars(ax, list(vals), ["Moving IoU", "Precision", "Recall"], vals)
    ax.set_xticklabels(list(vals))
    ax.axvline(0.5, color=INK2, linewidth=0.8, linestyle=(0, (3, 3)))
    ax.set_ylabel("score (moving class)")
    ax.set_title("Frozen temporal detector: development (tuned) vs held-out windows", loc="left", fontsize=11)
    ax.legend(ncol=3, loc="upper right", bbox_to_anchor=(1, 1.02))
    fig.tight_layout(); fig.savefig(OUT / "point_metrics_by_window.png", dpi=150); plt.close(fig)

    # 2. recall by measured object speed -----------------------------------------------------
    sp = json.loads(Path("results/phase2/recall_by_measured_speed.json").read_text())["by_speed"]
    bins = list(sp["heldout"])
    vals = {}
    for b in bins:
        lab = b.replace(" (near-stationary)", "\n(near-stationary)")
        lab += f"\nheld-out pts: {sp['heldout'][b]['moving_points']:,}"
        vals[lab] = [sp["development"][b]["point_recall"], sp["heldout"][b]["point_recall"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bars(ax, list(vals), ["Development", "Held-out pooled"], vals)
    ax.set_xticklabels(list(vals), fontsize=9)
    ax.set_ylabel("point recall of annotated moving objects")
    ax.set_xlabel("measured object speed (world frame, approximate)")
    ax.set_title("Recall depends on how far objects move within the ±1 s horizon", loc="left", fontsize=11)
    ax.legend(ncol=2, loc="upper left")
    fig.tight_layout(); fig.savefig(OUT / "recall_by_object_speed.png", dpi=150); plt.close(fig)

    # 3. map contamination per method ---------------------------------------------------------
    methods = [("raw_unfiltered", "Raw (no filtering)"), ("votes_only", "Votes only"),
               ("temporal_detector", "Temporal detector"), ("oracle_labels", "Oracle labels")]
    vals = {}
    for k in keys:
        vals[labels[k]] = [100 * S["mapping"][m][k]["contaminated_voxel_fraction"] for m, _ in methods]
    ymax = max(max(v) for v in vals.values())
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    bars(ax, list(vals), [n for _, n in methods], vals, fmt="{:.1f}", ymax=ymax)
    ax.set_xticklabels(list(vals))
    ax.set_ylabel("map voxels holding annotated moving points (%)")
    ax.set_title("Map contamination, 0.2 m voxels (same scans, poses, range mask)", loc="left", fontsize=11)
    ax.legend(ncol=4, loc="upper right", bbox_to_anchor=(1, 1.02), fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "map_contamination.png", dpi=150); plt.close(fig)
    print(sorted(p.name for p in OUT.glob("*.png")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
