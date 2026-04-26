import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from terramind import get_model, preprocess
from dataset import (
    PASTISCropDataset,
    compute_dataset_stats,
    filter_patches,
    split_patches,
)
from utils import SegmentationMetrics, validate

# Config
BACKBONE_VARIANT = "small"
PASTIS_ROOT = "/mnt/new_volume/dhruv/datasets/PASTIS"
TARGET_CLASSES = [1, 2, 3]  # Meadow=1, Wheat=2, Corn=3
NUM_CLASSES = 4
EMBED_DIM = 384
PATCH_GRID = 14
LABEL_SIZE = 128

BATCH_SIZE = 32
NUM_WORKERS = 2
EPOCHS = 150
LR = 5e-4
WEIGHT_DECAY = 1e-5
CLASS_WEIGHTS = torch.tensor([
    0.2,    # 0 = Background
    1.0,    # 1 = Meadow
    1.0,    # 2 = Wheat
    1.0,    # 3 = Corn
], dtype=torch.float32)


# Checkpoint 'config'
chkpt_name = f"{BACKBONE_VARIANT}_epochs{EPOCHS}_b{BATCH_SIZE}_lr{LR}"
CHECKPOINT_DIR = Path(f"./checkpoints/{chkpt_name}")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


# Segmentation head
# Input:  (B, N_patches, embed_dim)  — TerraMind backbone output
# Output: (B, num_classes, 128, 128) — pixel-wise class logits


class SegmentationHead(nn.Module):
    def __init__(
        self,
        embed_dim: int = EMBED_DIM,
        num_classes: int = NUM_CLASSES,
        patch_grid: int = PATCH_GRID,
        output_size: int = LABEL_SIZE,
    ):
        super().__init__()
        self.patch_grid = patch_grid
        self.output_size = output_size

        self.decoder = nn.Sequential(
            # (B, embed_dim, 14, 14) -> (B, 256, 28, 28)
            nn.ConvTranspose2d(embed_dim, 256, kernel_size=2, stride=2),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # (B, 256, 28, 28) -> (B, 128, 56, 56)
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # (B, 128, 56, 56) -> (B, 64, 112, 112)
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # (B, 64, 112, 112) -> (B, num_classes, 112, 112)
            nn.Conv2d(64, num_classes, kernel_size=1),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: (B, N_patches, embed_dim)
        Returns:
            logits: (B, num_classes, output_size, output_size)
        """
        B, N, D = tokens.shape
        g = self.patch_grid
        assert N == g * g, f"Expected {g * g} patches, got {N}. Check patch_grid."

        # (B, N, D) -> (B, D, g, g)
        x = tokens.permute(0, 2, 1).reshape(B, D, g, g)

        # (B, D, 14, 14) -> (B, num_classes, 112, 112)
        x = self.decoder(x)

        # Upsample to label resolution if needed (112 -> 128)
        if x.shape[-1] != self.output_size:
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )

        return x  # (B, num_classes, 128, 128)


# Full model
class TerraMindSegmenter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.seg_head = SegmentationHead()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, 10, 128, 128) — time-collapsed, /10000 normalised
        Returns:
            logits: (B, num_classes, 128, 128)
        """
        x = preprocess(images)  # (B, 12, 224, 224)

        tokens = self.backbone({"S2L2A": x})

        # terramind was generating list of tensors...
        if isinstance(tokens, (list, tuple)):
            tokens = tokens[0]  # (B, 196, 768)

        assert tokens.ndim == 3, (
            f"Expected 3D token tensor (B, N, D), got shape {tokens.shape}"
        )

        return self.seg_head(tokens)  # (B, num_classes, 128, 128)


# Train / validate loops
def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(loader, desc="  Train", leave=False):
        images = batch["image"].to(device)  # (B, T, 10, 128, 128)
        labels = batch["label"].to(device)  # (B, 128, 128)

        images = images.mean(dim=1)  # (B, 10, 128, 128)

        optimizer.zero_grad()
        logits = model(images)  # (B, num_classes, 128, 128)
        loss = criterion(logits, labels)  # CE, (B,C,H,W) + (B,H,W)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    return total_loss / total_samples


def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Dataset
    print("\n[INFO] Preparing dataset...")
    filtered_meta = filter_patches(
        data_path=PASTIS_ROOT,
        target_classes=TARGET_CLASSES,
        require_all=False,
        min_pixel_fraction=0.05,
    )
    train_meta, val_meta, test_meta = split_patches(filtered_meta)
    mean, std = compute_dataset_stats(PASTIS_ROOT, train_meta, max_patches=100)

    def make_dataset(meta):
        return PASTISCropDataset(
            data_path=PASTIS_ROOT,
            filtered_meta=meta,
            target_classes=TARGET_CLASSES,
            normalize=True,
            s2_mean=mean.tolist(),
            s2_std=std.tolist(),
        )

    def make_loader(ds, shuffle):
        return DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            collate_fn=PASTISCropDataset.collate_fn,
            pin_memory=device.type == "cuda",
        )

    train_ds = make_dataset(train_meta)
    val_ds = make_dataset(val_meta)
    train_loader = make_loader(train_ds, shuffle=True)
    val_loader = make_loader(val_ds, shuffle=False)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_meta)}")

    # Model
    print("\n[INFO] Loading TerraMind backbone...")
    backbone = get_model(variant=BACKBONE_VARIANT)
    model = TerraMindSegmenter(backbone).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params — total: {n_total:,} | trainable: {n_trainable:,}")

    # Loss / optimiser / scheduler
    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS.to(device))
    optimizer = optim.AdamW(
        [
            # smaller LR for base m
            {"params": model.backbone.parameters(), "lr": LR * 0.1},
            {"params": model.seg_head.parameters(), "lr": LR},
        ],
        weight_decay=WEIGHT_DECAY,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6,
    )

    # Training loop
    print("\n[INFO] Training...\n")
    best_miou = 0.0
    best_val_loss = float("inf")

    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_stats = validate(
            model, val_loader, criterion, device, NUM_CLASSES
        )

        scheduler.step()

        print(f"  Train loss : {train_loss:.4f}")
        print(f"  Val loss   : {val_loss:.4f}")
        SegmentationMetrics(NUM_CLASSES).print_summary(val_stats)

        # Checkpoint on best mIoU — more meaningful than loss for segmentation
        if val_stats["mean_iou"] > best_miou:
            best_miou = val_stats["mean_iou"]
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_stats": val_stats,
                },
                CHECKPOINT_DIR / "best_model.pt",
            )
            print(f"New best — mIoU={best_miou:.4f}, saved checkpoint")
        # save_metrics(train_loss, val_loss.item(), val_stats)

        print()

    print(f"[INFO] Done.  Best mIoU: {best_miou:.4f}  |  Val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
