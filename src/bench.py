"""
baseline_benchmark.py
=====================
Evaluates:
  1. Small baselines     (zero-shot, ImageNet, <21M params)
  2. Larger baselines    (zero-shot, ImageNet, >21M params)
  3. TerraMind runs      (fine-tuned, from ./checkpoints/<run>/best_model.pt)
  4. End-to-end pipeline (TerraMind → AE compress → AE decompress → mIoU)

Produces a console table and bench_report.md.

Requirements:
    pip install segmentation-models-pytorch timm
"""

import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from dataclasses import dataclass

import segmentation_models_pytorch as smp

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
CHECKPOINT_AE = Path("./checkpoints_ae")
REPORT_PATH = Path("./bench_report.md")

TARGET_CLASSES = [1, 2, 3]
NUM_CLASSES = 4
IN_CHANNELS = 10
INPUT_SIZE = 128
EMBED_DIM = 384
PATCH_GRID = 14

# AE architecture constants — must match autoencoder.py exactly
AE_LATENT_DIM = 128 * 2  # 256

BATCH_SIZE = 8
NUM_WORKERS = 2

CLASS_NAMES = {1: "Meadow", 2: "Wheat", 3: "Corn"}

PUBLISHED_RESULTS = [
    {
        "model_name": "U-TAE† (published)",
        "params_M": 1.3,
        "mean_iou": 0.631,
        "note": "ICCV 2021, 18 classes, full time series",
    },
    {
        "model_name": "TSViT† (published)",
        "params_M": 1.7,
        "mean_iou": 0.654,
        "note": "CVPR 2023, 18 classes, full time series",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# EvalResult
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class EvalResult:
    model_name: str
    model_type: str  # "small_baseline" | "large_baseline" | "terramind" | "pipeline"
    params_M: float
    val_loss: float
    mean_iou: float
    iou_meadow: float
    iou_wheat: float
    iou_corn: float
    accuracy: float
    inference_ms: float
    note: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# TerraMind segmenter (must match train.py exactly)
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
# MaskAutoencoder (must match autoencoder.py exactly)
# ─────────────────────────────────────────────────────────────────────────────


class MaskEncoder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, latent_dim=AE_LATENT_DIM):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Conv2d(num_classes, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256 * 8 * 8, latent_dim),
        )

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        x = F.one_hot(mask, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        return self.net(x)


class MaskDecoder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, latent_dim=AE_LATENT_DIM):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(latent_dim, 256 * 8 * 8), nn.Dropout(0.3))
        self.net = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.ConvTranspose2d(128, 64, 2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.ConvTranspose2d(32, num_classes, 2, stride=2),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).reshape(-1, 256, 8, 8)
        return self.net(x)


class MaskAutoencoder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, latent_dim=AE_LATENT_DIM):
        super().__init__()
        self.encoder = MaskEncoder(num_classes, latent_dim)
        self.decoder = MaskDecoder(num_classes, latent_dim)

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        z = self.encoder(mask)
        return self.decoder(z)


# ─────────────────────────────────────────────────────────────────────────────
# AE checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────


def discover_best_ae_checkpoint(ae_dir: Path) -> Path | None:
    """
    Scans ae_dir for files matching ae_best_<number>.pt and returns the
    path with the highest number.

    Expected filename format: ae_best_0042.pt, ae_best_100.pt, etc.
    """
    if not ae_dir.exists():
        print(f"[WARN] AE checkpoint dir not found: {ae_dir}")
        return None

    candidates = []
    for f in ae_dir.glob("ae_best_*.pt"):
        m = re.search(r"ae_best_(\d+)\.pt$", f.name)
        if m:
            candidates.append((int(m.group(1)), f))

    if not candidates:
        print(f"[WARN] No ae_best_*.pt files found in {ae_dir}")
        return None

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_score, best_path = candidates[0]
    print(f"[INFO] Best AE checkpoint: {best_path.name}  (score={best_score})")
    return best_path


def load_ae(ckpt_path: Path, device: torch.device) -> MaskAutoencoder:
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt)

    # Infer latent_dim from the saved weights instead of using the hardcoded constant
    latent_dim = state["encoder.net.14.weight"].shape[0]
    print(f"  AE latent_dim inferred from checkpoint: {latent_dim}")

    model = MaskAutoencoder(latent_dim=latent_dim)
    model.load_state_dict(state)
    model.to(device).eval()
    print(f"  AE loaded from {ckpt_path.name}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# End-to-end pipeline
# ─────────────────────────────────────────────────────────────────────────────


@torch.no_grad()
def run_pipeline(
    images: torch.Tensor,  # (B, C, H, W) — time-collapsed S2
    seg_model: nn.Module,
    ae_model: MaskAutoencoder,
) -> torch.Tensor:
    """
    Full end-to-end inference pipeline:

        S2 image
            │
            ▼
        TerraMindSegmenter   →  seg logits  (B, 4, 128, 128)
            │  argmax
            ▼
        seg mask  (B, 128, 128)  int64
            │
            ▼  MaskEncoder  (compress)
        latent z  (B, 256)
            │
            ▼  MaskDecoder  (decompress)
        reconstructed logits  (B, 4, 128, 128)
            │  argmax
            ▼
        reconstructed mask  (B, 128, 128)  int64

    Returns:
        reconstructed_mask: (B, 128, 128) int64  — final class prediction
    """
    # Step 1 — segmentation
    seg_logits = seg_model(images)  # (B, 4, 128, 128)
    seg_mask = seg_logits.argmax(dim=1)  # (B, 128, 128) int64

    # Step 2 — AE compress → decompress
    recon_logits = ae_model(seg_mask)  # (B, 4, 128, 128)
    recon_mask = recon_logits.argmax(dim=1)  # (B, 128, 128) int64

    return recon_mask


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery & loading (TerraMind)
# ─────────────────────────────────────────────────────────────────────────────


def discover_checkpoints(checkpoint_dir: Path) -> list[tuple[str, Path]]:
    found = []
    if not checkpoint_dir.exists():
        print(f"[WARN] Checkpoint dir not found: {checkpoint_dir}")
        return found
    for subdir in sorted(checkpoint_dir.iterdir()):
        if not subdir.is_dir():
            continue
        ckpt = subdir / "best_model.pt"
        if ckpt.exists():
            found.append((subdir.name, ckpt))
        else:
            print(f"[WARN] No best_model.pt in {subdir.name}/ — skipping")
    print(f"[INFO] Found {len(found)} TerraMind checkpoint(s):")
    for name, _ in found:
        print(f"       {name}")
    return found


def load_terramind_checkpoint(
    run_name: str, ckpt_path: Path, device: torch.device
) -> nn.Module:
    variant = "small"
    for v in ["tiny", "small", "base", "large"]:
        if v in run_name.lower():
            variant = v
            break
    print(f"  Loading TerraMind-{variant} from {ckpt_path.name} ...")
    backbone = get_model(variant=variant)
    model = TerraMindSegmenter(backbone)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    if "val_stats" in ckpt:
        s = ckpt["val_stats"]
        print(
            f"    Saved mIoU={s.get('mean_iou', '?'):.4f}  epoch={
                ckpt.get('epoch', '?')
            }"
        )
    return model.to(device)


# ─────────────────────────────────────────────────────────────────────────────
# SMP wrappers
# ─────────────────────────────────────────────────────────────────────────────


class SMPWrapper(nn.Module):
    def __init__(self, model: nn.Module, output_size: int = INPUT_SIZE):
        super().__init__()
        self.model = model
        self.output_size = output_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        if isinstance(out, dict):
            out = out["out"]
        if out.shape[-2:] != (self.output_size, self.output_size):
            out = F.interpolate(
                out,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        return out


class MostFrequentClassBaseline(nn.Module):
    def __init__(self, cls: int, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.cls = cls
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        logits = torch.zeros(B, self.num_classes, H, W, device=x.device)
        logits[:, self.cls] = 100.0
        return logits


def find_most_frequent_class(loader: DataLoader) -> int:
    counts = torch.zeros(NUM_CLASSES)
    for batch in loader:
        for cls in range(1, NUM_CLASSES):
            counts[cls] += (batch["label"] == cls).sum()
    return int(counts[1:].argmax().item()) + 1


def build_small_baselines(most_frequent_class: int) -> dict[str, nn.Module]:
    return {
        "MostFrequent": MostFrequentClassBaseline(most_frequent_class),
        "SegFormer-B0": SMPWrapper(
            smp.create_model(
                arch="segformer",
                encoder_name="mit_b0",
                encoder_weights="imagenet",
                in_channels=IN_CHANNELS,
                classes=NUM_CLASSES,
            )
        ),
        "UNet-ResNet18": SMPWrapper(
            smp.Unet(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                in_channels=IN_CHANNELS,
                classes=NUM_CLASSES,
                decoder_channels=(256, 128, 64, 32, 16),
            )
        ),
        "UNet++-ResNet18": SMPWrapper(
            smp.UnetPlusPlus(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                in_channels=IN_CHANNELS,
                classes=NUM_CLASSES,
                decoder_channels=(256, 128, 64, 32, 16),
            )
        ),
        "FPN-ResNet18": SMPWrapper(
            smp.FPN(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                in_channels=IN_CHANNELS,
                classes=NUM_CLASSES,
            )
        ),
        "DeepLabV3+-ResNet18": SMPWrapper(
            smp.DeepLabV3Plus(
                encoder_name="resnet18",
                encoder_weights="imagenet",
                in_channels=IN_CHANNELS,
                classes=NUM_CLASSES,
            )
        ),
    }


def build_large_baselines() -> dict[str, nn.Module]:
    return {
        "SegFormer-B2": SMPWrapper(
            smp.create_model(
                arch="segformer",
                encoder_name="mit_b2",
                encoder_weights="imagenet",
                in_channels=IN_CHANNELS,
                classes=NUM_CLASSES,
            )
        ),
        "SegFormer-B3": SMPWrapper(
            smp.create_model(
                arch="segformer",
                encoder_name="mit_b3",
                encoder_weights="imagenet",
                in_channels=IN_CHANNELS,
                classes=NUM_CLASSES,
            )
        ),
        "UNet-ResNet50": SMPWrapper(
            smp.Unet(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=IN_CHANNELS,
                classes=NUM_CLASSES,
                decoder_channels=(256, 128, 64, 32, 16),
            )
        ),
        "UNet-EfficientNetB4": SMPWrapper(
            smp.Unet(
                encoder_name="efficientnet-b4",
                encoder_weights="imagenet",
                in_channels=IN_CHANNELS,
                classes=NUM_CLASSES,
                decoder_channels=(256, 128, 64, 32, 16),
            )
        ),
        "DeepLabV3+-ResNet50": SMPWrapper(
            smp.DeepLabV3Plus(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=IN_CHANNELS,
                classes=NUM_CLASSES,
            )
        ),
        "MAnet-ResNet50": SMPWrapper(
            smp.MAnet(
                encoder_name="resnet50",
                encoder_weights="imagenet",
                in_channels=IN_CHANNELS,
                classes=NUM_CLASSES,
            )
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _compute_metrics(
    preds: torch.Tensor,  # (B, H, W) int64 — argmax predictions
    labels: torch.Tensor,  # (B, H, W) int64 — ground truth
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (intersection, union, correct_fg, total_fg) accumulators."""
    intersection = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)
    for cls in range(NUM_CLASSES):
        p = preds == cls
        l = labels == cls
        intersection[cls] += (p & l).sum()
        union[cls] += (p | l).sum()
    fg_mask = labels > 0
    correct = (preds[fg_mask] == labels[fg_mask]).sum()
    total = fg_mask.sum()
    return intersection, union, correct, total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
    model_type: str,
    note: str = "",
) -> EvalResult:
    """Standard evaluation — model receives mean-collapsed S2 and outputs logits."""
    model.eval()
    model.to(device)

    total_loss = 0.0
    total_samples = 0
    intersection = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)
    correct = torch.tensor(0, device=device)
    total_fg = torch.tensor(0, device=device)
    batch_times = []

    for batch in tqdm(loader, desc=f"  {model_name:<32}", leave=False):
        images = batch["image"].to(device).mean(dim=1)  # (B, 10, 128, 128)
        labels = batch["label"].to(device)  # (B, 128, 128)

        t0 = time.perf_counter()
        logits = model(images)
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_times.append((time.perf_counter() - t0) * 1000)

        loss = criterion(logits, labels)
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        preds = logits.argmax(dim=1)
        i, u, c, t = _compute_metrics(preds, labels, device)
        intersection += i
        union += u
        correct += c
        total_fg += t

    iou = intersection / (union + 1e-6)
    fg_iou = iou[1:]
    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    return EvalResult(
        model_name=model_name,
        model_type=model_type,
        params_M=round(n_params, 2),
        val_loss=round(total_loss / total_samples, 4),
        mean_iou=round(fg_iou.mean().item(), 4),
        iou_meadow=round(fg_iou[0].item(), 4),
        iou_wheat=round(fg_iou[1].item(), 4),
        iou_corn=round(fg_iou[2].item(), 4),
        accuracy=round((correct / (total_fg + 1e-6)).item(), 4),
        inference_ms=round(sum(batch_times) / len(batch_times), 2),
        note=note,
    )


@torch.no_grad()
def evaluate_pipeline(
    seg_model: nn.Module,
    ae_model: MaskAutoencoder,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    model_name: str,
) -> EvalResult:
    """
    Evaluates the full end-to-end pipeline:
        S2 image → TerraMindSegmenter → seg mask
                → MaskEncoder → latent z
                → MaskDecoder → reconstructed mask

    The mIoU is computed between the reconstructed mask and the
    ground-truth label — this measures the combined quality of
    segmentation + compression + decompression.
    """
    seg_model.eval()
    ae_model.eval()
    seg_model.to(device)
    ae_model.to(device)

    total_loss = 0.0
    total_samples = 0
    intersection = torch.zeros(NUM_CLASSES, device=device)
    union = torch.zeros(NUM_CLASSES, device=device)
    correct = torch.tensor(0, device=device)
    total_fg = torch.tensor(0, device=device)
    batch_times = []

    for batch in tqdm(loader, desc=f"  {model_name:<32}", leave=False):
        images = batch["image"].to(device).mean(dim=1)  # (B, 10, 128, 128)
        labels = batch["label"].to(device)  # (B, 128, 128)

        t0 = time.perf_counter()
        recon_mask = run_pipeline(images, seg_model, ae_model)
        if device.type == "cuda":
            torch.cuda.synchronize()
        batch_times.append((time.perf_counter() - t0) * 1000)

        # Loss: compare reconstructed logits (from AE decoder) against GT
        # Re-run decoder for logits (needed for loss — run_pipeline only returns argmax)
        seg_logits = seg_model(images)
        seg_mask = seg_logits.argmax(dim=1)
        recon_logits = ae_model(seg_mask)
        loss = criterion(recon_logits, labels)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        i, u, c, t = _compute_metrics(recon_mask, labels, device)
        intersection += i
        union += u
        correct += c
        total_fg += t

    iou = intersection / (union + 1e-6)
    fg_iou = iou[1:]

    seg_params = sum(p.numel() for p in seg_model.parameters()) / 1e6
    ae_params = sum(p.numel() for p in ae_model.parameters()) / 1e6

    return EvalResult(
        model_name=model_name,
        model_type="pipeline",
        params_M=round(seg_params + ae_params, 2),
        val_loss=round(total_loss / total_samples, 4),
        mean_iou=round(fg_iou.mean().item(), 4),
        iou_meadow=round(fg_iou[0].item(), 4),
        iou_wheat=round(fg_iou[1].item(), 4),
        iou_corn=round(fg_iou[2].item(), 4),
        accuracy=round((correct / (total_fg + 1e-6)).item(), 4),
        inference_ms=round(sum(batch_times) / len(batch_times), 2),
        note=f"TerraMind → AE compress → decompress  (latent_dim={AE_LATENT_DIM})",
    )


# ─────────────────────────────────────────────────────────────────────────────
# Console report
# ─────────────────────────────────────────────────────────────────────────────


def print_console_report(results: list[EvalResult]):
    best_miou = max(r.mean_iou for r in results)

    groups = [
        (
            "Small baselines  (zero-shot, ImageNet, <21M)",
            [r for r in results if r.model_type == "small_baseline"],
        ),
        (
            "Larger baselines  (zero-shot, ImageNet, >21M)",
            [r for r in results if r.model_type == "large_baseline"],
        ),
        (
            "TerraMind  (fine-tuned on PASTIS)",
            [r for r in results if r.model_type == "terramind"],
        ),
        (
            "End-to-end pipeline  (TerraMind + AE)",
            [r for r in results if r.model_type == "pipeline"],
        ),
    ]

    header = (
        f"  {'Model':<32} {'Params(M)':>9} {'Loss':>7} "
        f"{'mIoU':>7} {'Meadow':>8} {'Wheat':>8} {'Corn':>8} "
        f"{'Acc':>7} {'ms/batch':>9}"
    )
    sep = "  " + "─" * (len(header) - 2)

    print("\n" + "═" * 110)
    print("  BENCHMARK RESULTS")
    print("═" * 110)

    for title, rows in groups:
        if not rows:
            continue
        print(f"\n  ── {title}")
        print(header)
        print(sep)
        for r in rows:
            star = " ★" if r.mean_iou == best_miou else "  "
            print(
                f"  {r.model_name:<32} {r.params_M:>9.2f} {r.val_loss:>7.4f} "
                f"{r.mean_iou:>7.4f}{star}"
                f"{r.iou_meadow:>8.4f} {r.iou_wheat:>8.4f} {r.iou_corn:>8.4f} "
                f"{r.accuracy:>7.4f} {r.inference_ms:>9.2f}"
            )
            if r.note:
                print(f"  {'':32}  ↳ {r.note}")
        print(sep)

    print(f"\n  ── Published PASTIS results  (reference — not directly comparable)")
    print(f"  {'Model':<32} {'Params(M)':>9} {'mIoU':>7}   Note")
    print(sep)
    for p in PUBLISHED_RESULTS:
        print(
            f"  {p['model_name']:<32} {p['params_M']:>9.1f} {p['mean_iou']:>7.4f}   {
                p['note']
            }"
        )
    print(sep)
    print("\n" + "═" * 110)


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────


def save_markdown(results: list[EvalResult], path: Path):
    best_miou = max(r.mean_iou for r in results)

    def table_rows(rows: list[EvalResult]) -> list[str]:
        lines = []
        for r in rows:
            miou_str = (
                f"**{r.mean_iou:.4f}**"
                if r.mean_iou == best_miou
                else f"{r.mean_iou:.4f}"
            )
            note_str = f" _{r.note}_" if r.note else ""
            lines.append(
                f"| {r.model_name}{note_str} | {r.params_M:.2f} | {r.val_loss:.4f} | "
                f"{miou_str} | {r.iou_meadow:.4f} | {r.iou_wheat:.4f} | "
                f"{r.iou_corn:.4f} | {r.accuracy:.4f} | {r.inference_ms:.2f} |"
            )
        return lines

    header = (
        "| Model | Params (M) | Val Loss | mIoU | "
        "IoU Meadow | IoU Wheat | IoU Corn | Accuracy | ms/batch |"
    )
    sep = "|---|---|---|---|---|---|---|---|---|"

    small_bl = [r for r in results if r.model_type == "small_baseline"]
    large_bl = [r for r in results if r.model_type == "large_baseline"]
    terramind = [r for r in results if r.model_type == "terramind"]
    pipeline = [r for r in results if r.model_type == "pipeline"]

    lines = [
        "# Benchmark Report",
        "",
        "**Task:** Crop semantic segmentation — Meadow / Wheat / Corn  ",
        "**Dataset:** PASTIS (Sentinel-2, 128×128 patches, France)  ",
        "**Input:** 10-band S2, temporal mean collapse → (B, 10, 128, 128)  ",
        "**Metric:** mIoU over foreground classes 1–3 (background ignored)  ",
        "",
        "---",
        "",
        "## 1. Small baselines  (zero-shot, ImageNet, <21M params)",
        "",
        "> No PASTIS fine-tuning — lower bound.",
        "",
        header,
        sep,
        *table_rows(small_bl),
        "",
        "---",
        "",
        "## 2. Larger baselines  (zero-shot, ImageNet, >21M params)",
        "",
        "> Still zero-shot — no PASTIS fine-tuning.",
        "",
        header,
        sep,
        *table_rows(large_bl),
        "",
        "---",
        "",
    ]

    if terramind:
        lines += [
            "## 3. TerraMind  (fine-tuned on PASTIS)",
            "",
            "> Loaded from `./checkpoints/<run>/best_model.pt`.  ",
            "> EO-pretrained on TerraMesh (9M samples).",
            "",
            header,
            sep,
            *table_rows(terramind),
            "",
            "---",
            "",
        ]

    if pipeline:
        lines += [
            "## 4. End-to-end pipeline  (TerraMind + MaskAutoencoder)",
            "",
            "> Full pipeline: `S2 → TerraMind → seg mask → AE encoder → latent z → AE decoder → reconstructed mask`  ",
            f"> AE checkpoint: best `ae_best_*.pt` from `{CHECKPOINT_AE}/`  ",
            f"> Latent dimension: {AE_LATENT_DIM}  ",
            "> mIoU measured between **reconstructed mask** and **ground truth** — includes compression loss.",
            "",
            header,
            sep,
            *table_rows(pipeline),
            "",
            "---",
            "",
        ]

    # Summary
    baselines = small_bl + large_bl
    if terramind or pipeline:
        best_b = max(baselines, key=lambda r: r.mean_iou) if baselines else None
        best_tm = max(terramind, key=lambda r: r.mean_iou) if terramind else None
        best_pipe = max(pipeline, key=lambda r: r.mean_iou) if pipeline else None

        lines += ["## 5. Summary", "", "| | Model | mIoU |", "|---|---|---|"]
        if best_b:
            lines.append(
                f"| Best baseline   | {best_b.model_name} | {best_b.mean_iou:.4f} |"
            )
        if best_tm:
            delta = best_tm.mean_iou - (best_b.mean_iou if best_b else 0)
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| Best TerraMind  | {best_tm.model_name} | {best_tm.mean_iou:.4f} ({
                    sign
                }{delta:.4f} vs baseline) |"
            )
        if best_pipe and best_tm:
            ae_drop = best_pipe.mean_iou - best_tm.mean_iou
            sign = "+" if ae_drop >= 0 else ""
            lines.append(
                f"| Pipeline (+ AE) | {best_pipe.model_name} | {
                    best_pipe.mean_iou:.4f} ({sign}{ae_drop:.4f} vs TerraMind) |"
            )
        lines += ["", "---", ""]

    lines += [
        "## 6. Published PASTIS results  (reference only)",
        "",
        "> † Not directly comparable — 18 classes, full time series.",
        "",
        "| Model | Params (M) | mIoU | Setting | Source |",
        "|---|---|---|---|---|",
        "| U-TAE† | 1.3 | 0.6310 | 18 classes, full time series | ICCV 2021 |",
        "| TSViT† | 1.7 | 0.6540 | 18 classes, full time series | CVPR 2023 |",
        "",
        "---",
        "",
        "## 7. Notes",
        "",
        "- **Bold** mIoU = best live result across all models.",
        "- All single-frame models receive `images.mean(dim=1)` → (B, 10, 128, 128).",
        "- TerraMind applies `preprocess()`: B01/B09 zero-inserted, standardised, resized 128→224.",
        f"- Pipeline AE latent dim = {
            AE_LATENT_DIM
        }. The mIoU drop vs TerraMind alone quantifies compression loss.",
        "- `ms/batch` includes full pipeline latency for the pipeline row.",
    ]

    path.write_text("\n".join(lines))
    print(f"\n📄 Report saved → {path.resolve()}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
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
    train_meta, val_meta, _ = split_patches(filtered_meta)
    mean, std = compute_dataset_stats(PASTIS_ROOT, train_meta, max_patches=100)

    val_ds = PASTISCropDataset(
        data_path=PASTIS_ROOT,
        filtered_meta=val_meta,
        target_classes=TARGET_CLASSES,
        normalize=True,
        s2_mean=mean.tolist(),
        s2_std=std.tolist(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=PASTISCropDataset.collate_fn,
        pin_memory=device.type == "cuda",
    )
    print(f"Val samples: {len(val_ds)}")

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    results: list[EvalResult] = []

    # ── Small baselines ───────────────────────────────────────────────────────
    print("\n[INFO] Finding most frequent class...")
    most_frequent = find_most_frequent_class(val_loader)
    print(f"Most frequent: class {most_frequent} ({CLASS_NAMES[most_frequent]})")

    print("\n[INFO] Evaluating small baselines...")
    for name, model in build_small_baselines(most_frequent).items():
        try:
            results.append(
                evaluate(model, val_loader, criterion, device, name, "small_baseline")
            )
            print(f"  ✓ {name:<32}  mIoU={results[-1].mean_iou:.4f}")
        except Exception as e:
            print(f"  ✗ {name} — FAILED: {e}")

    # ── Large baselines ───────────────────────────────────────────────────────
    print("\n[INFO] Evaluating large baselines...")
    for name, model in build_large_baselines().items():
        try:
            results.append(
                evaluate(model, val_loader, criterion, device, name, "large_baseline")
            )
            print(f"  ✓ {name:<32}  mIoU={results[-1].mean_iou:.4f}")
        except Exception as e:
            print(f"  ✗ {name} — FAILED: {e}")

    # ── TerraMind checkpoints + pipeline ──────────────────────────────────────
    checkpoints = discover_checkpoints(CHECKPOINT_DIR)
    ae_ckpt = discover_best_ae_checkpoint(CHECKPOINT_AE)
    ae_model = load_ae(ae_ckpt, device) if ae_ckpt else None

    if checkpoints:
        print("\n[INFO] Evaluating TerraMind checkpoints...")
        for run_name, ckpt_path in checkpoints:
            try:
                seg_model = load_terramind_checkpoint(run_name, ckpt_path, device)

                # Standard TerraMind eval
                result = evaluate(
                    seg_model, val_loader, criterion, device, run_name, "terramind"
                )
                results.append(result)
                print(f"  ✓ {run_name:<32}  mIoU={result.mean_iou:.4f}")

                # End-to-end pipeline eval (only if AE available)
                if ae_model is not None:
                    print(f"  → Running end-to-end pipeline for {run_name}...")
                    pipe_name = f"{run_name} + AE"
                    pipe_result = evaluate_pipeline(
                        seg_model,
                        ae_model,
                        val_loader,
                        criterion,
                        device,
                        pipe_name,
                    )
                    results.append(pipe_result)
                    drop = pipe_result.mean_iou - result.mean_iou
                    sign = "+" if drop >= 0 else ""
                    print(
                        f"  ✓ {pipe_name:<32}  mIoU={pipe_result.mean_iou:.4f}  "
                        f"(AE delta: {sign}{drop:.4f})"
                    )

            except Exception as e:
                print(f"  ✗ {run_name} — FAILED: {e}")
    else:
        print("\n[WARN] No TerraMind checkpoints found.")

    if ae_model is None:
        print("\n[WARN] No AE checkpoint found — pipeline rows skipped.")

    # ── Reports ───────────────────────────────────────────────────────────────
    print_console_report(results)
    save_markdown(results, REPORT_PATH)


if __name__ == "__main__":
    main()
