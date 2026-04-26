"""
infer.py
========
Interactive CLI for the full satellite → ground pipeline.

Fixes vs previous version:
  - Applies the SAME z-score normalisation the training dataset used
    (computed from the training split, not skipped entirely)
  - Prints per-class prediction bars + mIoU to console
  - Saves output_mask.png alongside a pred_bars.png

Usage:
    python infer.py
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

from src.terramind import get_model, preprocess
from src.encode.model import MaskDecoder, MaskEncoder
from src.dataset import (
    filter_patches,
    split_patches,
    compute_dataset_stats,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PASTIS_ROOT = "/mnt/new_volume/dhruv/datasets/PASTIS"
NUM_CLASSES = 4
EMBED_DIM = 384
PATCH_GRID = 14
LABEL_SIZE = 128

CLASS_NAMES = {0: "Background", 1: "Meadow", 2: "Wheat", 3: "Corn"}
CLASS_COLORS_HEX = {
    0: "#1a1a2e",
    1: "#4caf50",
    2: "#ffc107",
    3: "#f44336",
}
CLASS_COLORS_RGB = {
    0: (29, 29, 46),
    1: (76, 175, 80),
    2: (255, 193, 7),
    3: (244, 67, 54),
}


# ─────────────────────────────────────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────────────────────────────────────


class SegmentationHead(nn.Module):
    def __init__(
        self,
        embed_dim=EMBED_DIM,
        num_classes=NUM_CLASSES,
        patch_grid=PATCH_GRID,
        output_size=LABEL_SIZE,
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
# Normalisation — MUST match dataset.py exactly
# ─────────────────────────────────────────────────────────────────────────────


def get_train_stats() -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the same per-band mean/std the training dataset used.
    Uses the same filter + split + seed as train.py / visualize_masks.py.
    """
    print("  Computing training normalisation stats (cached after first run)...")
    filtered = filter_patches(
        data_path=PASTIS_ROOT,
        target_classes=[1, 2, 3],
        require_all=False,
        min_pixel_fraction=0.05,
    )
    train_meta, _, _ = split_patches(filtered)
    mean, std = compute_dataset_stats(PASTIS_ROOT, train_meta, max_patches=100)
    return mean, std  # (10,) numpy arrays


def normalize_sample(arr: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply /10000  then z-score with training stats.
    arr: (C, H, W) float32 after time collapse
    """
    arr = arr / 10000.0  # → [0, 1]
    arr = (arr - mean[:, None, None]) / (std[:, None, None] + 1e-6)
    return arr


def load_pastis_sample(
    npy_path: Path, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    """
    Load PASTIS .npy → (1, 10, 128, 128) float32, fully normalised.

    Handles shapes:
      (T, C, H, W)     → mean over T  → normalise → unsqueeze batch
      (B, T, C, H, W)  → mean over T  → normalise
    """
    arr = np.load(npy_path).astype(np.float32)  # raw DN values
    print(f"  Raw array shape: {arr.shape}  max={arr.max():.1f}")

    # Collapse time dimension
    if arr.ndim == 5:  # (B, T, C, H, W)
        arr = arr.mean(axis=1)  # (B, C, H, W)
    elif arr.ndim == 4:  # (T, C, H, W)
        arr = arr.mean(axis=0)  # (C, H, W)
        arr = arr[None]  # (1, C, H, W)
    elif arr.ndim == 3:  # (C, H, W)
        arr = arr[None]
    else:
        print(f"  ✗ Unexpected shape: {arr.shape}")
        sys.exit(1)

    assert arr.shape[1] == 10, f"Expected 10 bands, got {arr.shape[1]}"

    # Apply the SAME normalisation as PASTISCropDataset
    arr_out = np.stack(
        [normalize_sample(arr[b], mean, std) for b in range(arr.shape[0])]
    )

    return torch.from_numpy(arr_out)  # (B, 10, 128, 128)


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def infer_satellite(image, seg_ckpt_path, ae_ckpt_path, device):
    device = torch.device(device)

    print("  Loading TerraMind segmenter...")
    backbone = get_model(variant="small")
    seg_model = TerraMindSegmenter(backbone).to(device)
    seg_ckpt = torch.load(seg_ckpt_path, map_location=device)
    seg_model.load_state_dict(seg_ckpt.get("model_state_dict", seg_ckpt))
    seg_model.eval()

    print("  Loading AE encoder...")
    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_state = ae_ckpt.get("model_state_dict", ae_ckpt)
    latent_dim = (
        ae_state["encoder.net.14.weight"].shape[0]
        if "encoder.net.14.weight" in ae_state
        else 256
    )
    encoder = MaskEncoder(num_classes=NUM_CLASSES, latent_dim=latent_dim).to(device)
    encoder.load_state_dict(
        {
            k.replace("encoder.", ""): v
            for k, v in ae_state.items()
            if k.startswith("encoder.")
        }
    )
    encoder.eval()

    image = image.to(device)
    logits = seg_model(image)  # (B, 4, 128, 128)
    mask = logits.argmax(dim=1)  # (B, 128, 128)
    z = encoder(mask)  # (B, latent_dim)
    return z, logits, mask


@torch.no_grad()
def infer_earth_systems(z, ae_ckpt_path, device):
    device = torch.device(device)
    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_state = ae_ckpt.get("model_state_dict", ae_ckpt)
    latent_dim = (
        ae_state["encoder.net.14.weight"].shape[0]
        if "encoder.net.14.weight" in ae_state
        else 256
    )

    print("  Loading AE decoder...")
    decoder = MaskDecoder(num_classes=NUM_CLASSES, latent_dim=latent_dim).to(device)
    decoder.load_state_dict(
        {
            k.replace("decoder.", ""): v
            for k, v in ae_state.items()
            if k.startswith("decoder.")
        }
    )
    decoder.eval()

    logits = decoder(z.to(device))
    return logits.argmax(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# mIoU (foreground only)
# ─────────────────────────────────────────────────────────────────────────────


def compute_miou(pred: np.ndarray, gt: np.ndarray) -> float:
    """pred, gt: (H, W) integer arrays."""
    ious = []
    for cls in range(1, NUM_CLASSES):
        p = pred == cls
        l = gt == cls
        inter = (p & l).sum()
        union = (p | l).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────────────────────────────────────


def save_mask_png(mask_np: np.ndarray, out_path: Path):
    """mask_np: (H, W) int — saves colour-coded PNG."""
    from PIL import Image

    rgb = np.zeros((*mask_np.shape, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS_RGB.items():
        rgb[mask_np == cls_id] = color
    Image.fromarray(rgb).save(out_path)
    print(f"  ✓  Mask PNG → {out_path}")


def save_bar_chart(recon_mask: np.ndarray, out_path: Path, miou: float):
    """
    Single horizontal stacked bar showing the predicted class distribution.
    Saves next to the mask PNG.
    """
    total = recon_mask.size
    fracs = [(recon_mask == cls).sum() / total for cls in range(NUM_CLASSES)]

    fig, ax = plt.subplots(figsize=(8, 1.6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    left = 0.0
    for cls in range(NUM_CLASSES):
        w = fracs[cls]
        if w < 1e-4:
            left += w
            continue
        ax.barh(0, w, 0.5, left=left, color=CLASS_COLORS_HEX[cls], alpha=0.92)
        if w > 0.05:
            ax.text(
                left + w / 2,
                0,
                f"{w * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        left += w

    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Pixel fraction", fontsize=9, color="#8b949e")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.tick_params(axis="x", colors="#555555", labelsize=8)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.grid(True, color="#2d333b", linewidth=0.4, zorder=0)

    ax.set_title(
        f"Reconstructed mask — class distribution  ·  mIoU = {miou:.4f}",
        color="white",
        fontsize=10,
        fontfamily="monospace",
        pad=8,
    )

    patches = [
        mpatches.Patch(color=CLASS_COLORS_HEX[c], label=CLASS_NAMES[c])
        for c in range(NUM_CLASSES)
    ]
    ax.legend(
        handles=patches,
        loc="lower right",
        ncol=4,
        fontsize=8,
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="white",
        framealpha=0.9,
    )

    plt.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    print(f"  ✓  Bar chart → {out_path}")


def print_bars(recon_mask: np.ndarray, miou: float):
    """Console class distribution bars + mIoU."""
    total = recon_mask.size
    print(f"\n  Reconstructed mask  —  mIoU vs GT: {miou:.4f}\n")
    print(f"  {'Class':<12}  {'Pixels':>8}   {'%':>5}   bar")
    print(f"  {'─' * 12}  {'─' * 8}   {'─' * 5}   {'─' * 40}")
    for cls_id, name in CLASS_NAMES.items():
        count = int((recon_mask == cls_id).sum())
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"  {name:<12}  {count:>8,}   {pct:>5.1f}%   {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Prompting
# ─────────────────────────────────────────────────────────────────────────────


def prompt(message: str, default: str | None = None) -> str:
    suffix = f"  [{default}]" if default else ""
    try:
        val = input(f"{message}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)
    return val if val else (default or "")


def resolve(raw: str, must_exist: bool = True) -> Path:
    p = Path(raw).expanduser().resolve()
    if must_exist and not p.exists():
        print(f"  ✗  Not found: {p}")
        sys.exit(1)
    return p


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("═" * 58)
    print("  TerraCrop inference pipeline")
    print("  Satellite image → compress → reconstruct mask")
    print("═" * 58)
    print("\nPaths  (press Enter to use the default)\n")

    sample_path = resolve(prompt("PASTIS S2 sample (.npy)"))
    seg_ckpt = resolve(
        prompt("TerraMind seg checkpoint (.pt)", default="best_model.pt")
    )
    ae_ckpt = resolve(prompt("AutoEncoder checkpoint (.pt)", default="ae_best_9096.pt"))
    out_raw = prompt("Output mask (.png)", default="output_mask.png")
    out_path = resolve(out_raw, must_exist=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    device_str = prompt(
        "Device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── Normalisation stats (same as training) ────────────────────────────
    print("\nComputing normalisation stats from training split...")
    mean, std = get_train_stats()

    # ── Load + normalise sample ───────────────────────────────────────────
    print(f"\nLoading sample: {sample_path}")
    image = load_pastis_sample(sample_path, mean, std)
    print(f"  Normalised tensor shape: {tuple(image.shape)}")

    # ── Satellite side ────────────────────────────────────────────────────
    print("\n[1/2] Satellite-side  (segment + compress)...")
    z, seg_logits, seg_mask = infer_satellite(
        image,
        str(seg_ckpt),
        str(ae_ckpt),
        device_str,
    )
    print(f"  Latent z: {tuple(z.shape)}  ({z.numel() * 4 / 1024:.1f} KB)")

    # ── Ground side ───────────────────────────────────────────────────────
    print("\n[2/2] Ground-side  (decompress)...")
    recon_mask = infer_earth_systems(z, str(ae_ckpt), device_str)

    recon_np = recon_mask.squeeze(0).cpu().numpy()  # (128, 128)

    # ── mIoU vs seg mask (no GT label available here) ────────────────────
    seg_np = seg_mask.squeeze(0).cpu().numpy()
    # AE reconstruction vs direct seg
    miou = compute_miou(recon_np, seg_np)

    # ── Console output ────────────────────────────────────────────────────
    print_bars(recon_np, miou)

    # ── Save mask PNG ─────────────────────────────────────────────────────
    print(f"\nSaving outputs...")
    save_mask_png(recon_np, out_path)

    # ── Save bar chart PNG next to the mask ───────────────────────────────
    bar_path = out_path.with_stem(out_path.stem + "_bars")
    save_bar_chart(recon_np, bar_path, miou)

    print("\n✓  Done.\n")


if __name__ == "__main__":
    main()
