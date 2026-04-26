"""
infer.py
========
Interactive CLI for the full satellite → ground pipeline.

Usage:
    python infer.py

The script will prompt you for:
  1. Path to a PASTIS S2 .npy sample  (B, T, 10, 128, 128)  or  (10, 128, 128)
  2. Path to the TerraMind seg checkpoint
  3. Path to the AE checkpoint
  4. Output path to save the reconstructed mask  (.npy or .png)
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from src.terramind import get_model, preprocess
from src.encode.model import MaskDecoder, MaskEncoder

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

NUM_CLASSES = 4
EMBED_DIM = 384
PATCH_GRID = 14
LABEL_SIZE = 128

CLASS_NAMES = {0: "Background", 1: "Meadow", 2: "Wheat", 3: "Corn"}

CLASS_COLORS = {  # RGB uint8 — used when saving as .png
    0: (29, 29, 46),  # dark navy  — background
    1: (76, 175, 80),  # green      — meadow
    2: (255, 193, 7),  # amber      — wheat
    3: (244, 67, 54),  # red        — corn
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

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.seg_head = SegmentationHead()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = preprocess(images)
        tokens = self.backbone({"S2L2A": x})
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]
        return self.seg_head(tokens)


# ─────────────────────────────────────────────────────────────────────────────
# Inference functions
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def infer_satellite(
    image: torch.Tensor,
    seg_ckpt_path: str,
    ae_ckpt_path: str,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Satellite-side inference: segment image and compress mask.

    Args:
        image:         (B, 10, 128, 128) float32, values in [0, 1]
        seg_ckpt_path: path to TerraMind segmentation checkpoint
        ae_ckpt_path:  path to autoencoder checkpoint

    Returns:
        z: (B, latent_dim) compressed latent vector
    """
    device = torch.device(device)

    # Load segmenter
    print("  Loading TerraMind segmenter...")
    backbone = get_model(variant="small")
    seg_model = TerraMindSegmenter(backbone).to(device)
    seg_ckpt = torch.load(seg_ckpt_path, map_location=device)
    seg_model.load_state_dict(seg_ckpt.get("model_state_dict", seg_ckpt))
    seg_model.eval()

    # Load encoder
    print("  Loading AE encoder...")
    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_state = ae_ckpt.get("model_state_dict", ae_ckpt)
    latent_dim = (
        ae_state["encoder.net.14.weight"].shape[0]
        if "encoder.net.14.weight" in ae_state
        else 256
    )
    encoder = MaskEncoder(num_classes=NUM_CLASSES, latent_dim=latent_dim).to(device)
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in ae_state.items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    # Forward
    image = image.to(device)
    logits = seg_model(image)  # (B, 4, 128, 128)
    mask = logits.argmax(dim=1)  # (B, 128, 128)
    z = encoder(mask)  # (B, latent_dim)
    return z


@torch.no_grad()
def infer_earth_systems(
    z: torch.Tensor,
    ae_ckpt_path: str,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Ground-side inference: decompress latent vector to segmentation mask.

    Args:
        z:            (B, latent_dim) compressed latent vector
        ae_ckpt_path: path to autoencoder checkpoint

    Returns:
        mask: (B, 128, 128) int64 reconstructed segmentation mask
    """
    device = torch.device(device)

    print("  Loading AE decoder...")
    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_state = ae_ckpt.get("model_state_dict", ae_ckpt)
    latent_dim = (
        ae_state["encoder.net.14.weight"].shape[0]
        if "encoder.net.14.weight" in ae_state
        else 256
    )
    decoder = MaskDecoder(num_classes=NUM_CLASSES, latent_dim=latent_dim).to(device)
    decoder_state = {
        k.replace("decoder.", ""): v
        for k, v in ae_state.items()
        if k.startswith("decoder.")
    }
    decoder.load_state_dict(decoder_state)
    decoder.eval()

    z = z.to(device)
    logits = decoder(z)  # (B, 4, 128, 128)
    mask = logits.argmax(dim=1)  # (B, 128, 128)
    return mask


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────


def prompt(message: str, default: str | None = None) -> str:
    """Prompt the user, with an optional default value shown in brackets."""
    suffix = f"  [{default}]" if default else ""
    try:
        val = input(f"{message}{suffix}: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.")
        sys.exit(0)
    return val if val else (default or "")


def resolve_path(raw: str, must_exist: bool = True) -> Path:
    p = Path(raw).expanduser().resolve()
    if must_exist and not p.exists():
        print(f"  ✗  Not found: {p}")
        sys.exit(1)
    return p


def load_pastis_sample(npy_path: Path) -> torch.Tensor:
    """
    Load a PASTIS .npy file and return a (1, 10, 128, 128) float32 tensor.

    Handles:
      - (T, C, H, W)  → mean over T, unsqueeze batch
      - (C, H, W)     → unsqueeze batch
      - (B, T, C, H, W) → mean over T
      - (B, C, H, W)  → passed through
    """
    arr = np.load(npy_path).astype(np.float32)

    # Normalise to [0, 1] if values look like raw DN (>10)
    if arr.max() > 10.0:
        arr = arr / 10000.0

    t = torch.from_numpy(arr)

    if t.ndim == 5:  # (B, T, C, H, W)
        t = t.mean(dim=1)
    elif t.ndim == 4 and t.shape[0] > 4:  # (T, C, H, W) — T is large
        t = t.mean(dim=0).unsqueeze(0)
    elif t.ndim == 4:  # (B, C, H, W) or (T≤4, C, H, W)
        if t.shape[1] == 10:  # looks like (B, 10, H, W)
            pass
        else:  # (T, C, H, W) with small T
            t = t.mean(dim=0).unsqueeze(0)
    elif t.ndim == 3:  # (C, H, W)
        t = t.unsqueeze(0)
    else:
        print(f"  ✗  Unexpected array shape: {arr.shape}")
        sys.exit(1)

    assert t.shape[1] == 10, (
        f"Expected 10 S2 bands, got {t.shape[1]}. Check your .npy file."
    )
    return t  # (B, 10, 128, 128)


def save_mask(mask: torch.Tensor, out_path: Path):
    """
    Save the reconstructed mask (B, H, W) as either .npy or .png.

    .npy  → integer class IDs array
    .png  → colour-coded RGB image (one frame per batch item)
    """
    mask_np = mask.cpu().numpy().astype(np.int64)  # (B, H, W)

    if out_path.suffix.lower() == ".npy":
        np.save(out_path, mask_np)
        print(f"  ✓  Mask saved as NumPy array → {out_path}")

    elif out_path.suffix.lower() in (".png", ".jpg", ".jpeg"):
        try:
            from PIL import Image
        except ImportError:
            print("  [WARN] Pillow not installed. Saving as .npy instead.")
            np.save(out_path.with_suffix(".npy"), mask_np)
            return

        for b in range(mask_np.shape[0]):
            frame = mask_np[b]  # (H, W)
            rgb = np.zeros((*frame.shape, 3), dtype=np.uint8)
            for cls_id, color in CLASS_COLORS.items():
                rgb[frame == cls_id] = color
            img = Image.fromarray(rgb)
            # If batch > 1, append index to filename
            save_as = (
                out_path
                if mask_np.shape[0] == 1
                else out_path.with_stem(f"{out_path.stem}_{b}")
            )
            img.save(save_as)
            print(f"  ✓  Mask saved as colour PNG → {save_as}")
    else:
        # Unknown extension — default to .npy
        npy_path = out_path.with_suffix(".npy")
        np.save(npy_path, mask_np)
        print(f"  ✓  Unknown extension '{out_path.suffix}' — saved as → {npy_path}")


def print_mask_summary(mask: torch.Tensor):
    """Print a quick class distribution summary."""
    mask_np = mask.cpu().numpy()
    total = mask_np.size
    print("\n  Class distribution:")
    for cls_id, name in CLASS_NAMES.items():
        count = int((mask_np == cls_id).sum())
        pct = count / total * 100
        bar = "█" * int(pct / 2)
        print(f"    {cls_id}  {name:<12}  {count:>7,} px  ({pct:5.1f}%)  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# Main CLI
# ─────────────────────────────────────────────────────────────────────────────


def main():
    print("═" * 58)
    print("  TerraCrop inference pipeline")
    print("  Satellite image → compress → reconstruct mask")
    print("═" * 58)

    # ── Prompt for inputs ─────────────────────────────────────────────────
    print("\nPaths  (press Enter to use the default shown in brackets)\n")

    sample_path = resolve_path(prompt("PASTIS S2 sample (.npy)"))
    seg_ckpt = resolve_path(
        prompt(
            "TerraMind seg checkpoint (.pt)",
            default="best_model.pt",
        )
    )
    ae_ckpt = resolve_path(
        prompt(
            "AutoEncoder checkpoint (.pt)",
            default="ae_best_9096.pt",
        )
    )
    out_raw = prompt(
        "Output path (.npy for class IDs, .png for colour image)",
        default="output_mask.png",
    )
    out_path = resolve_path(out_raw, must_exist=False)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────
    device_str = prompt(
        "Device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    device = torch.device(device_str)

    # ── Load sample ───────────────────────────────────────────────────────
    print(f"\nLoading sample from  {sample_path} ...")
    image = load_pastis_sample(sample_path)
    print(f"  Input shape after preprocessing: {tuple(image.shape)}")

    # ── Satellite side ────────────────────────────────────────────────────
    print("\n[1/2] Satellite-side  (segment + compress)...")
    z = infer_satellite(
        image=image,
        seg_ckpt_path=str(seg_ckpt),
        ae_ckpt_path=str(ae_ckpt),
        device=device,
    )
    print(
        f"  Latent z shape: {tuple(z.shape)}  "
        f"({z.numel() * 4 / 1024:.1f} KB as float32)"
    )

    # ── Ground side ───────────────────────────────────────────────────────
    print("\n[2/2] Ground-side  (decompress + reconstruct)...")
    mask = infer_earth_systems(
        z=z,
        ae_ckpt_path=str(ae_ckpt),
        device=device,
    )
    print(f"  Reconstructed mask shape: {tuple(mask.shape)}")

    # ── Summary + save ────────────────────────────────────────────────────
    print_mask_summary(mask)

    print(f"\nSaving mask → {out_path} ...")
    save_mask(mask, out_path)

    print("\n✓  Done.\n")


if __name__ == "__main__":
    main()
