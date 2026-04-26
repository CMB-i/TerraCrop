"""
visualize_masks.py
==================
Samples 25 random patches from each of train / val / test splits and
produces one figure per split showing:
  - Left  : Sentinel-2 false-colour composite (NIR-Red-Green)
  - Middle : Predicted segmentation mask (from best TerraMind checkpoint)
  - Right  : Ground truth label

Each crop class gets its own distinct colour:
    0 = Background → transparent / dark grey
    1 = Meadow     → green
    2 = Wheat      → gold
    3 = Corn       → orange-red

Output:
    ./plots/train_masks.png
    ./plots/val_masks.png
    ./plots/test_masks.png

Usage:
    python visualize_masks.py
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from pathlib import Path

from terramind import get_model, preprocess
from dataset import (
    PASTISCropDataset,
    compute_dataset_stats,
    filter_patches,
    split_patches,
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

PASTIS_ROOT = "/mnt/new_volume/dhruv/datasets/PASTIS"
CHECKPOINT_DIR = Path("./checkpoints")
PLOTS_DIR = Path("./plots")

TARGET_CLASSES = [1, 2, 3]
NUM_CLASSES = 4
EMBED_DIM = 384
PATCH_GRID = 14
INPUT_SIZE = 128

N_SAMPLES = 25  # samples per split
COLS = 5  # grid columns  (5 × 5 = 25)
ROWS = 5
SEED = 42

# ── Colour palette ────────────────────────────────────────────────────────────
#   Index : class       : colour
#   0     : Background  : near-black
#   1     : Meadow      : sage green
#   2     : Wheat       : warm gold
#   3     : Corn        : burnt orange

PALETTE = {
    0: "#1a1a2e",  # background — very dark navy
    1: "#4caf50",  # meadow     — vivid green
    2: "#ffc107",  # wheat      — amber gold
    3: "#f44336",  # corn       — vivid red
}

LEGEND_LABELS = {
    0: "Background",
    1: "Meadow",
    2: "Wheat",
    3: "Corn",
}

# Build matplotlib colormap from palette
CMAP_COLORS = [PALETTE[i] for i in range(NUM_CLASSES)]
CMAP = ListedColormap(CMAP_COLORS)
CMAP_NORM = BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5], ncolors=NUM_CLASSES)


# ─────────────────────────────────────────────────────────────────────────────
# TerraMind segmenter (must match train.py)
# ─────────────────────────────────────────────────────────────────────────────


class SegmentationHead(nn.Module):
    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        patch_grid=PATCH_GRID,
        output_size=INPUT_SIZE,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.output_size = output_size
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, tokens):
        B, N, D = tokens.shape
        g = self.patch_grid
        x = tokens.permute(0, 2, 1).reshape(B, D, g, g)
        x = self.decoder(x)
        if x.shape[-1] != self.output_size:
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return x


class TerraMindSegmenter(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.seg_head = SegmentationHead()

    def forward(self, images):
        x = preprocess(images)
        tokens = self.backbone({"S2L2A": x})
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        return self.seg_head(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────


def load_best_checkpoint(
    checkpoint_dir: Path, device: torch.device
) -> nn.Module | None:
    """
    Finds the best checkpoint across all run subdirs by reading saved val_stats.
    Falls back to the most recently modified best_model.pt if no val_stats.
    Returns None if no checkpoints found.
    """
    candidates = []
    for subdir in sorted(checkpoint_dir.iterdir()):
        if not subdir.is_dir():
            continue
        ckpt_path = subdir / "best_model.pt"
        if not ckpt_path.exists():
            continue

        ckpt = torch.load(ckpt_path, map_location="cpu")
        miou = ckpt.get("val_stats", {}).get("mean_iou", 0.0)
        candidates.append((miou, subdir.name, ckpt_path, ckpt))

    if not candidates:
        return None

    # Pick checkpoint with highest saved mIoU
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_miou, best_name, best_path, best_ckpt = candidates[0]
    print(f"[INFO] Using checkpoint: {best_name}  (saved mIoU={best_miou:.4f})")

    # Infer variant from run name
    variant = "small"
    for v in ["tiny", "small", "base", "large"]:
        if v in best_name.lower():
            variant = v
            break

    backbone = get_model(variant=variant)
    model = TerraMindSegmenter(backbone)
    model.load_state_dict(best_ckpt.get("model_state_dict", best_ckpt))
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────────────────────────────────────


def s2_to_rgb(s2_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a single S2 patch to a displayable false-colour image.
    Uses NIR (band 6), Red (band 2), Green (band 1) — classic vegetation composite.

    s2_tensor: (T, C, H, W) or (C, H, W), values in [0, 1] after normalisation.
    Returns: (H, W, 3) uint8 numpy array.
    """
    if s2_tensor.ndim == 4:
        s2 = s2_tensor.mean(dim=0)  # collapse time → (C, H, W)
    else:
        s2 = s2_tensor

    # NIR=idx6, Red=idx2, Green=idx1  — undo z-score approximately via clipping
    nir = s2[6].numpy()
    red = s2[2].numpy()
    green = s2[1].numpy()

    rgb = np.stack([nir, red, green], axis=-1)  # (H, W, 3)

    # Percentile stretch for display
    lo, hi = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def label_to_rgb(label: np.ndarray) -> np.ndarray:
    """
    Converts an integer label map (H, W) to an RGB image using PALETTE.
    Returns: (H, W, 3) uint8 array.
    """
    h, w = label.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, hex_col in PALETTE.items():
        r = int(hex_col[1:3], 16)
        g = int(hex_col[3:5], 16)
        b = int(hex_col[5:7], 16)
        mask = label == cls_id
        rgb[mask] = [r, g, b]
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Main plot function
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def plot_split(
    split_name: str,
    dataset: PASTISCropDataset,
    model: nn.Module | None,
    device: torch.device,
    output_path: Path,
    n_samples: int = N_SAMPLES,
    seed: int = SEED,
):
    """
    Draws a grid of n_samples patches showing:
        [S2 false-colour] [Predicted mask] [Ground truth label]

    If model is None, the predicted mask column shows 'No model loaded'.
    """
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), k=min(n_samples, len(dataset)))

    n_cols = COLS
    n_rows = ROWS

    # Each sample = 3 sub-panels: image | pred | gt
    # Layout: ROWS rows × (COLS × 3) columns
    fig_w = n_cols * 3 * 2.2
    fig_h = n_rows * 2.4 + 1.2  # +1.2 for title + legend

    fig, axes = plt.subplots(
        n_rows,
        n_cols * 3,
        figsize=(fig_w, fig_h),
        facecolor="#0d1117",
    )

    fig.suptitle(
        f"{split_name.upper()} SPLIT — Segmentation Masks  ({n_samples} samples)",
        color="white",
        fontsize=16,
        fontweight="bold",
        fontfamily="monospace",
        y=0.98,
    )

    # Column headers (drawn once on first row)
    col_titles = ["S2 False-Colour", "Predicted", "Ground Truth"]

    for sample_idx, ds_idx in enumerate(indices):
        row = sample_idx // n_cols
        col = sample_idx % n_cols

        # ── Load sample ──────────────────────────────────────────────────────
        sample = dataset[ds_idx]
        image = sample["image"]  # (T, C, H, W)
        label = sample["label"]  # (H, W) int64 tensor

        # ── Predict ──────────────────────────────────────────────────────────
        if model is not None:
            inp = image.mean(dim=0).unsqueeze(0).to(device)  # (1, C, H, W)
            logits = model(inp)  # (1, 4, H, W)
            pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()
        else:
            pred = np.zeros_like(label.numpy())

        # ── Convert to display arrays ─────────────────────────────────────
        s2_rgb = s2_to_rgb(image)  # (H, W, 3) uint8
        pred_rgb = label_to_rgb(pred)  # (H, W, 3) uint8
        gt_rgb = label_to_rgb(label.numpy())  # (H, W, 3) uint8

        # ── Plot 3 sub-panels ─────────────────────────────────────────────
        for panel, (img_arr, title) in enumerate(
            zip(
                [s2_rgb, pred_rgb, gt_rgb],
                col_titles,
            )
        ):
            ax = axes[row, col * 3 + panel]
            ax.imshow(img_arr, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            for spine in ax.spines.values():
                spine.set_edgecolor("#2d333b")
                spine.set_linewidth(0.5)

            # Column title on first row only
            if row == 0:
                ax.set_title(
                    title,
                    color="#8b949e",
                    fontsize=7,
                    fontfamily="monospace",
                    pad=3,
                )

            # Sample index label on S2 panel
            if panel == 0:
                ax.text(
                    2,
                    4,
                    f"#{ds_idx}",
                    color="white",
                    fontsize=5,
                    fontfamily="monospace",
                    va="top",
                    bbox=dict(facecolor="#0d1117", alpha=0.6, pad=1, linewidth=0),
                )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(
            facecolor=PALETTE[cls_id], edgecolor="#444", label=LEGEND_LABELS[cls_id]
        )
        for cls_id in range(NUM_CLASSES)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=NUM_CLASSES,
        frameon=True,
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="white",
        fontsize=9,
        handlelength=1.4,
        handleheight=0.9,
        borderpad=0.6,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 0.005),
    )

    plt.tight_layout(rect=[0, 0.04, 1, 0.97])
    plt.subplots_adjust(wspace=0.03, hspace=0.08)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  ✓ Saved → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    PLOTS_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ── Dataset ───────────────────────────────────────────────────────────────
    print("\n[INFO] Preparing dataset...")
    filtered_meta = filter_patches(
        data_path=PASTIS_ROOT,
        target_classes=TARGET_CLASSES,
        require_all=False,
        min_pixel_fraction=0.05,
    )
    train_meta, val_meta, test_meta = split_patches(filtered_meta)
    mean, std = compute_dataset_stats(PASTIS_ROOT, train_meta, max_patches=100)

    def make_ds(meta):
        return PASTISCropDataset(
            data_path=PASTIS_ROOT,
            filtered_meta=meta,
            target_classes=TARGET_CLASSES,
            normalize=True,
            s2_mean=mean.tolist(),
            s2_std=std.tolist(),
        )

    splits = {
        "train": make_ds(train_meta),
        "val": make_ds(val_meta),
        "test": make_ds(test_meta),
    }

    for name, ds in splits.items():
        print(f"  {name:<6}: {len(ds)} patches")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[INFO] Loading best checkpoint...")
    model = load_best_checkpoint(CHECKPOINT_DIR, device)

    if model is None:
        print("[WARN] No checkpoint found — predicted mask panels will be blank.")
    else:
        model.eval()

    # ── Plot each split ───────────────────────────────────────────────────────
    print("\n[INFO] Generating plots...")
    for split_name, dataset in splits.items():
        print(f"  Plotting {split_name}...")
        plot_split(
            split_name=split_name,
            dataset=dataset,
            model=model,
            device=device,
            output_path=PLOTS_DIR / f"{split_name}_masks.png",
            n_samples=N_SAMPLES,
            seed=SEED,
        )

    print(f"\n[INFO] All plots saved to {PLOTS_DIR.resolve()}/")


if __name__ == "__main__":
    main()
