"""
plot_model_comparison.py
========================
Generates a model size (params) vs mIoU scatter plot.

Points are styled as:
  - Grey  x  : small baselines  (<21M)
  - Muted orange x : large baselines (>21M)
  - Blue  x  : TerraCrop (our TerraMind model)
  - Blue  x  : Pipeline  (TerraMind + AE)
  - Dashed reference lines for published U-TAE / TSViT

Run after bench.py has produced results, or edit RESULTS directly.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Results — paste from bench.py output, or import EvalResult and load directly
# Format: (label, params_M, mean_iou, category)
# category: "small" | "large" | "ours" | "pipeline" | "published"
# ─────────────────────────────────────────────────────────────────────────────

RESULTS = [
    # ── Small baselines ──────────────────────────────────────────────
    ("MostFrequent", 0.00, None, "small"),  # set None to skip
    ("SegFormer-B0", 3.71, None, "small"),
    ("UNet-ResNet18", 14.32, None, "small"),
    ("UNet++-ResNet18", 15.14, None, "small"),
    ("FPN-ResNet18", 16.23, None, "small"),
    ("DeepLabV3+-ResNet18", 16.41, None, "small"),
    # ── Large baselines ──────────────────────────────────────────────
    ("SegFormer-B2", 25.00, None, "large"),
    ("SegFormer-B3", 45.00, None, "large"),
    ("UNet-ResNet50", 32.00, None, "large"),
    ("UNet-EfficientNetB4", 19.00, None, "large"),
    ("DeepLabV3+-ResNet50", 40.00, None, "large"),
    ("MAnet-ResNet50", 30.00, None, "large"),
    # ── Ours ─────────────────────────────────────────────────────────
    ("TerraCrop", 22.10, None, "ours"),
    ("Pipeline", 23.50, None, "pipeline"),
    # ── Published reference (dashed horizontal lines, not plotted as x) ──
    ("U-TAE†", 1.30, 0.631, "published"),
    ("TSViT†", 1.70, 0.654, "published"),
]

# Output path
OUTPUT_PATH = Path("./plots/model_comparison.png")

# ─────────────────────────────────────────────────────────────────────────────
# Style constants
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    "small": "#9e9e9e",  # grey
    "large": "#e07b39",  # muted orange
    "ours": "#1565C0",  # strong blue
    "pipeline": "#1565C0",  # same blue
    "published": "#888888",  # grey dashes
}

MARKER_SIZE = 120
FONT_LABEL = 8.5
FONT_AXIS = 10
FONT_TITLE = 12

# ─────────────────────────────────────────────────────────────────────────────
# Plot
# ─────────────────────────────────────────────────────────────────────────────


def make_plot(results: list[tuple], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )

    fig, ax = plt.subplots(figsize=(10, 6.5))

    plotted = []  # (x, y, label, category) — for smart label placement
    published = []  # (label, miou) — for horizontal reference lines

    for label, params, miou, cat in results:
        if miou is None:
            continue
        if cat == "published":
            published.append((label, miou))
            continue
        plotted.append((params, miou, label, cat))

    # ── Published reference lines ─────────────────────────────────────────
    for pub_label, pub_miou in published:
        ax.axhline(
            pub_miou,
            color=COLORS["published"],
            linewidth=1.0,
            linestyle="--",
            alpha=0.55,
            zorder=1,
        )
        ax.text(
            ax.get_xlim()[1] if ax.get_xlim()[1] > 0 else 50,
            pub_miou + 0.003,
            pub_label,
            fontsize=FONT_LABEL - 0.5,
            color=COLORS["published"],
            ha="right",
            va="bottom",
            style="italic",
        )

    # ── Scatter points ────────────────────────────────────────────────────
    for params, miou, label, cat in plotted:
        color = COLORS[cat]
        alpha = 1.0 if cat in ("ours", "pipeline") else 0.72
        zorder = 5 if cat in ("ours", "pipeline") else 3
        lw = 2.5 if cat in ("ours", "pipeline") else 1.5

        ax.scatter(
            params,
            miou,
            marker="x",
            s=MARKER_SIZE,
            color=color,
            linewidths=lw,
            alpha=alpha,
            zorder=zorder,
        )

    # ── Labels (offset to avoid overlap) ─────────────────────────────────
    # Simple collision-aware nudge: alternate above/below for dense clusters
    sorted_pts = sorted(plotted, key=lambda p: p[0])  # sort by x

    for i, (params, miou, label, cat) in enumerate(sorted_pts):
        color = COLORS[cat]
        weight = "bold" if cat in ("ours", "pipeline") else "normal"
        alpha = 1.0 if cat in ("ours", "pipeline") else 0.78

        # Default nudge
        dx, dy = 0.4, 0.008

        # Nudge down for every other point to reduce vertical overlap
        if i % 2 == 1:
            dy = -0.018

        # Extra nudge for our models so they stand out
        if cat == "ours":
            dy = 0.012
        if cat == "pipeline":
            dy = -0.022

        txt = ax.text(
            params + dx,
            miou + dy,
            label,
            fontsize=FONT_LABEL,
            color=color,
            fontweight=weight,
            alpha=alpha,
            zorder=6,
            va="center",
        )

        # White halo behind our labels for legibility
        if cat in ("ours", "pipeline"):
            txt.set_path_effects(
                [
                    pe.withStroke(linewidth=2.5, foreground="white"),
                ]
            )

    # ── Legend ────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="x",
            color=COLORS["small"],
            linestyle="None",
            markersize=8,
            markeredgewidth=1.5,
            label="Small baseline  (<21M)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=COLORS["large"],
            linestyle="None",
            markersize=8,
            markeredgewidth=1.5,
            label="Large baseline  (>21M)",
        ),
        Line2D(
            [0],
            [0],
            marker="x",
            color=COLORS["ours"],
            linestyle="None",
            markersize=9,
            markeredgewidth=2.5,
            label="Ours (TerraCrop / Pipeline)",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["published"],
            linestyle="--",
            linewidth=1.0,
            alpha=0.6,
            label="Published PASTIS  (18 cls, reference)",
        ),
    ]
    ax.legend(
        handles=legend_elements,
        loc="lower right",
        fontsize=FONT_LABEL,
        framealpha=0.9,
        edgecolor="#cccccc",
        fancybox=False,
    )

    # ── Axes formatting ───────────────────────────────────────────────────
    ax.set_xlabel("Model size  (M parameters)", fontsize=FONT_AXIS, labelpad=8)
    ax.set_ylabel("mIoU  (val, 3 classes)", fontsize=FONT_AXIS, labelpad=8)
    ax.set_title(
        "Model size vs segmentation quality\nPASTIS  ·  Meadow / Wheat / Corn",
        fontsize=FONT_TITLE,
        fontweight="normal",
        pad=14,
    )

    ax.tick_params(axis="both", labelsize=FONT_AXIS - 1)
    ax.set_xlim(left=-1)
    ax.grid(axis="y", linewidth=0.4, color="#e0e0e0", zorder=0)
    ax.grid(axis="x", linewidth=0.3, color="#eeeeee", zorder=0)

    # ── Second pass: draw published reference line labels now x-lim is set ─
    # (matplotlib autoscales after scatter, so we do this after all points)
    xmax = ax.get_xlim()[1]
    for pub_label, pub_miou in published:
        ax.texts[-len(published)].set_x(xmax - 0.5)  # flush right

    plt.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✓  Plot saved → {output_path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Integration with bench.py EvalResult
# ─────────────────────────────────────────────────────────────────────────────


def results_from_eval(eval_results) -> list[tuple]:
    """
    Convert a list of EvalResult dataclass instances (from bench.py) into
    the (label, params_M, mean_iou, category) tuples this script expects.

    Usage:
        from bench import main as run_bench, EvalResult
        # ... or load from a saved JSON / pickle

        from plot_model_comparison import results_from_eval, make_plot
        tuples = results_from_eval(eval_results)
        make_plot(tuples, Path("plots/model_comparison.png"))
    """
    cat_map = {
        "small_baseline": "small",
        "large_baseline": "large",
        "terramind": "ours",
        "pipeline": "pipeline",
    }

    rows = []
    for r in eval_results:
        cat = cat_map.get(r.model_type, "small")
        label = "TerraCrop" if r.model_type == "terramind" else r.model_name
        # Strip run-name suffix for pipeline rows  e.g. "small_epochs100... + AE"
        if r.model_type == "pipeline":
            label = "Pipeline"
        rows.append((label, r.params_M, r.mean_iou, cat))

    # Add published reference lines
    rows += [
        ("U-TAE†", 1.3, 0.631, "published"),
        ("TSViT†", 1.7, 0.654, "published"),
    ]
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Main — edit RESULTS above with your actual numbers and run directly
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Filter out rows with no mIoU (None = placeholder)
    ready = [(l, p, m, c) for l, p, m, c in RESULTS if m is not None]

    if not ready:
        print("[WARN] All mIoU values are None in RESULTS.")
        print("       Paste your actual numbers from bench.py output into RESULTS,")
        print(
            "       or call make_plot(results_from_eval(eval_results), ...) directly."
        )
    else:
        make_plot(ready, OUTPUT_PATH)
