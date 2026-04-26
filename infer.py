import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from pathlib import Path

from src.terramind import get_model, preprocess
from src.encode.model import MaskDecoder, MaskEncoder
from src.dataset import filter_patches, split_patches, compute_dataset_stats

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

PASTIS_ROOT = "/mnt/new_volume/dhruv/datasets/PASTIS"
NUM_CLASSES = 4
EMBED_DIM = 384
PATCH_GRID = 14
LABEL_SIZE = 128

CLASS_NAMES = {0: "Background", 1: "Meadow", 2: "Wheat", 3: "Corn"}
CLASS_COLORS_HEX = {0: "#1a1a2e", 1: "#4caf50", 2: "#ffc107", 3: "#f44336"}
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
# Normalisation — must match dataset.py exactly
# ─────────────────────────────────────────────────────────────────────────────


def get_train_stats():
    print("  Computing training normalisation stats...")
    filtered = filter_patches(
        data_path=PASTIS_ROOT,
        target_classes=[1, 2, 3],
        require_all=False,
        min_pixel_fraction=0.05,
    )
    train_meta, _, _ = split_patches(filtered)
    return compute_dataset_stats(PASTIS_ROOT, train_meta, max_patches=100)


def load_pastis_sample(npy_path: Path, mean: np.ndarray, std: np.ndarray):
    """
    Returns:
        normalised: (1, 10, 128, 128) tensor — model input
        raw_s2:     (10, H, W) float32 array in [0,1] — for display
    """
    arr = np.load(npy_path).astype(np.float32)
    print(f"  Raw array shape: {arr.shape}  max={arr.max():.1f}")

    if arr.ndim == 5:  # (B, T, C, H, W)
        arr = arr.mean(axis=1)
    elif arr.ndim == 4:  # (T, C, H, W)
        arr = arr.mean(axis=0)[None]
    elif arr.ndim == 3:  # (C, H, W)
        arr = arr[None]

    assert arr.shape[1] == 10, f"Expected 10 bands, got {arr.shape[1]}"

    # Raw [0,1] copy for display
    raw_s2 = (arr[0] / 10000.0).clip(0, 1)  # (10, H, W)

    # Normalised copy for model
    norm = arr / 10000.0
    norm = (norm - mean[None, :, None, None]) / (std[None, :, None, None] + 1e-6)
    return torch.from_numpy(norm), raw_s2


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def infer_satellite(image, seg_ckpt_path, ae_ckpt_path, device):
    device = torch.device(device)

    print("  Loading TerraMind segmenter...")
    backbone = get_model(variant="small")
    seg_model = TerraMindSegmenter(backbone).to(device)
    ckpt = torch.load(seg_ckpt_path, map_location=device)
    seg_model.load_state_dict(ckpt.get("model_state_dict", ckpt))
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
    logits = seg_model(image)
    mask = logits.argmax(dim=1)
    z = encoder(mask)
    return z, mask


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
    return decoder(z.to(device)).argmax(dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def compute_miou(pred: np.ndarray, ref: np.ndarray) -> float:
    ious = []
    for cls in range(1, NUM_CLASSES):
        p, l = pred == cls, ref == cls
        inter = (p & l).sum()
        union = (p | l).sum()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def s2_to_rgb(s2: np.ndarray) -> np.ndarray:
    """s2: (10, H, W) in [0,1].  Returns (H, W, 3) uint8 NIR-Red-Green composite."""
    nir, red, green = s2[6], s2[2], s2[1]
    rgb = np.stack([nir, red, green], axis=-1)
    lo, hi = np.percentile(rgb, (2, 98))
    rgb = np.clip((rgb - lo) / (hi - lo + 1e-6), 0, 1)
    return (rgb * 255).astype(np.uint8)


def mask_to_rgb(mask: np.ndarray) -> np.ndarray:
    """mask: (H, W) int.  Returns (H, W, 3) uint8."""
    rgb = np.zeros((*mask.shape, 3), dtype=np.uint8)
    for cls_id, color in CLASS_COLORS_RGB.items():
        rgb[mask == cls_id] = color
    return rgb


# ─────────────────────────────────────────────────────────────────────────────
# Combined output figure
# ─────────────────────────────────────────────────────────────────────────────


def save_combined(
    raw_s2: np.ndarray,  # (10, H, W) in [0, 1]
    recon_mask: np.ndarray,  # (H, W) int
    seg_mask: np.ndarray,  # (H, W) int  — direct seg, used as reference
    miou: float,
    out_path: Path,
):
    """
    Three-panel figure:
      Left   — S2 false-colour composite (NIR-Red-Green)
      Centre — reconstructed segmentation mask (colour-coded)
      Right  — class distribution bar + mIoU

    Dark theme matching visualize_masks.py.
    """
    BG = "#0d1117"

    fig = plt.figure(figsize=(13, 4.8), facecolor=BG)
    gs = gridspec.GridSpec(
        1,
        3,
        width_ratios=[1, 1, 1.4],
        wspace=0.06,
        left=0.02,
        right=0.98,
        top=0.88,
        bottom=0.14,
    )

    ax_s2 = fig.add_subplot(gs[0])
    ax_mask = fig.add_subplot(gs[1])
    ax_bar = fig.add_subplot(gs[2])

    # ── Panel 1: S2 false-colour ──────────────────────────────────────────
    ax_s2.imshow(s2_to_rgb(raw_s2), interpolation="nearest")
    ax_s2.set_title(
        "S2 false colour\n(NIR · Red · Green)",
        color="#8b949e",
        fontsize=9,
        fontfamily="monospace",
        pad=6,
    )
    ax_s2.set_xticks([])
    ax_s2.set_yticks([])
    for sp in ax_s2.spines.values():
        sp.set_edgecolor("#2d333b")
        sp.set_linewidth(0.5)

    # ── Panel 2: reconstructed mask ───────────────────────────────────────
    ax_mask.imshow(mask_to_rgb(recon_mask), interpolation="nearest")
    ax_mask.set_title(
        f"Reconstructed mask\nmIoU (AE fidelity) = {miou:.4f}",
        color="#8b949e",
        fontsize=9,
        fontfamily="monospace",
        pad=6,
    )
    ax_mask.set_xticks([])
    ax_mask.set_yticks([])
    for sp in ax_mask.spines.values():
        sp.set_edgecolor("#2d333b")
        sp.set_linewidth(0.5)

    # ── Panel 3: class distribution bar ───────────────────────────────────
    ax_bar.set_facecolor(BG)
    total = recon_mask.size
    fracs = [(recon_mask == cls).sum() / total for cls in range(NUM_CLASSES)]

    bar_h, y = 0.45, 0.0
    left = 0.0
    for cls in range(NUM_CLASSES):
        w = fracs[cls]
        if w < 1e-4:
            left += w
            continue
        ax_bar.barh(y, w, bar_h, left=left, color=CLASS_COLORS_HEX[cls], alpha=0.92)
        if w > 0.06:
            ax_bar.text(
                left + w / 2,
                y,
                f"{w * 100:.1f}%",
                ha="center",
                va="center",
                fontsize=9,
                color="white",
                fontweight="bold",
            )
        left += w

    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.6, 0.6)
    ax_bar.set_yticks([])
    ax_bar.set_xlabel("Pixel fraction", fontsize=9, color="#8b949e", labelpad=6)
    ax_bar.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax_bar.tick_params(axis="x", colors="#555555", labelsize=8)
    for sp in ax_bar.spines.values():
        sp.set_visible(False)
    ax_bar.xaxis.grid(True, color="#2d333b", linewidth=0.4, zorder=0)
    ax_bar.set_title(
        "Class distribution\n(reconstructed mask)",
        color="#8b949e",
        fontsize=9,
        fontfamily="monospace",
        pad=6,
    )

    # Legend inside bar panel
    patches = [
        mpatches.Patch(color=CLASS_COLORS_HEX[c], label=CLASS_NAMES[c])
        for c in range(NUM_CLASSES)
    ]
    ax_bar.legend(
        handles=patches,
        loc="lower center",
        ncol=2,
        fontsize=8,
        facecolor="#161b22",
        edgecolor="#30363d",
        labelcolor="white",
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.52),
    )

    # ── Super-title ───────────────────────────────────────────────────────
    fig.suptitle(
        "TerraCrop  ·  satellite → compress → reconstruct",
        color="white",
        fontsize=11,
        fontweight="bold",
        fontfamily="monospace",
        y=0.97,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  Output PNG → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Console bar
# ─────────────────────────────────────────────────────────────────────────────


def print_bars(recon_mask: np.ndarray, miou: float):
    total = recon_mask.size
    print(f"\n  Reconstructed mask  —  AE fidelity mIoU: {miou:.4f}\n")
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
    out_raw = prompt("Output path (.png)", default="output.png")
    out_path = resolve(out_raw, must_exist=False)
    device_str = prompt(
        "Device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # ── Normalisation ─────────────────────────────────────────────────────
    print("\nPreparing normalisation stats...")
    mean, std = get_train_stats()

    # ── Load sample ───────────────────────────────────────────────────────
    print(f"\nLoading: {sample_path}")
    image, raw_s2 = load_pastis_sample(sample_path, mean, std)
    print(f"  Model input shape: {tuple(image.shape)}")

    # ── Satellite side ────────────────────────────────────────────────────
    print("\n[1/2] Satellite-side  (segment + compress)...")
    z, seg_mask = infer_satellite(image, str(seg_ckpt), str(ae_ckpt), device_str)
    print(f"  Latent z: {tuple(z.shape)}  ({z.numel() * 4 / 1024:.1f} KB)")

    # ── Ground side ───────────────────────────────────────────────────────
    print("\n[2/2] Ground-side  (decompress)...")
    recon_mask = infer_earth_systems(z, str(ae_ckpt), device_str)

    recon_np = recon_mask.squeeze(0).cpu().numpy()
    seg_np = seg_mask.squeeze(0).cpu().numpy()
    miou = compute_miou(recon_np, seg_np)

    # ── Console output ────────────────────────────────────────────────────
    print_bars(recon_np, miou)

    # ── Save combined PNG ─────────────────────────────────────────────────
    print(f"\nSaving → {out_path}")
    save_combined(raw_s2, recon_np, seg_np, miou, out_path)

    print("\n✓  Done.\n")


if __name__ == "__main__":
    main()
