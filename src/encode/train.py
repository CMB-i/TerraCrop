import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from pathlib import Path
from tqdm import tqdm
from model import MaskAutoencoder, LATENT_DIM

# ── CONFIG ── only change PASTIS_ROOT ──────────────────────────────
PASTIS_ROOT = "/mnt/new_volume/dhruv/datasets/PASTIS"
ANNOTATIONS_DIR = Path(PASTIS_ROOT) / "ANNOTATIONS"
CHECKPOINT_DIR = Path("./checkpoints_ae")
BATCH_SIZE = 32
EPOCHS = 100
LR = 3e-3
VAL_SPLIT = 0.15
TARGET_CLASSES = [1, 2, 3]
DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
CLASS_WEIGHTS = torch.tensor([
    0.3,    # 0 = Background
    1.8,    # 1 = Meadow
    1.8,    # 2 = Wheat
    1.8,    # 3 = Corn
], dtype=torch.float32).to(DEVICE)
# ───────────────────────────────────────────────────────────────────


class MaskDataset(Dataset):
    def __init__(
        self, annotations_dir: Path, synthetic: bool = False, n_synthetic: int = 500
    ):
        self.synthetic = synthetic
        if synthetic or not annotations_dir.exists():
            print(f"[INFO] Using synthetic masks ({n_synthetic} samples)")
            self.synthetic = True
            self.n = n_synthetic
            return
        self.files = sorted(annotations_dir.glob("TARGET_*.npy"))
        self.files = [f for f in self.files if self._has_target(f)]
        print(f"[INFO] Found {len(self.files)} patches with target classes")

    def _has_target(self, path: Path) -> bool:
        label = np.load(path)
        if label.ndim == 3:
            label = label[0]
        return any(np.any(label == c) for c in TARGET_CLASSES)

    def _remap(self, label: np.ndarray) -> np.ndarray:
        out = np.zeros_like(label, dtype=np.int64)
        for new_id, old_id in enumerate(TARGET_CLASSES, start=1):
            out[label == old_id] = new_id
        return out

    def __len__(self):
        return self.n if self.synthetic else len(self.files)

    def __getitem__(self, idx):
        if self.synthetic:
            mask = torch.zeros(128, 128, dtype=torch.long)
            for i in range(4):
                r, c = (i // 2) * 64, (i % 2) * 64
                mask[r : r + 64, c : c + 64] = torch.randint(0, 4, (1,)).item()
            return mask
        label = np.load(self.files[idx])
        if label.ndim == 3:
            label = label[0]
        label = self._remap(label)
        return torch.from_numpy(label)


def pixel_accuracy(preds, targets):
    return (preds == targets).float().mean().item()


def foreground_accuracy(preds, targets):
    fg = targets > 0
    if fg.sum() == 0:
        return 1.0
    return (preds[fg] == targets[fg]).float().mean().item()


def compression_stats(model, mask):
    raw_bytes = mask.numel()
    compressed = model.compress(mask.unsqueeze(0).to(DEVICE))
    return {
        "raw_bytes": raw_bytes,
        "compressed_bytes": len(compressed),
        "ratio": round(raw_bytes / len(compressed), 1),
    }


def train():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    print(f"Device: {DEVICE}\n")

    use_synthetic = not ANNOTATIONS_DIR.exists()
    full_ds = MaskDataset(ANNOTATIONS_DIR, synthetic=use_synthetic)

    n_val = int(len(full_ds) * VAL_SPLIT)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    print(f"Train: {n_train} | Val: {n_val}\n")

    model = MaskAutoencoder().to(DEVICE)
    mb = sum(p.numel() for p in model.parameters()) * 4 / 1e6
    print(
        f"Autoencoder params: {sum(p.numel() for p in model.parameters()):,} ({
            mb:.1f} MB)\n"
    )

    criterion = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=EPOCHS, eta_min=1e-5
    )

    best_fg_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for masks in tqdm(train_loader, desc=f"Epoch {epoch:02d} train", leave=False):
            masks = masks.to(DEVICE)
            logits = model(masks)
            loss = criterion(logits, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_pix_acc, all_fg_acc = [], []
        with torch.no_grad():
            for masks in val_loader:
                masks = masks.to(DEVICE)
                logits = model(masks)
                val_loss += criterion(logits, masks).item()
                preds = logits.argmax(dim=1)
                all_pix_acc.append(pixel_accuracy(preds, masks))
                all_fg_acc.append(foreground_accuracy(preds, masks))

        scheduler.step()

        avg_pix = np.mean(all_pix_acc)
        avg_fg = np.mean(all_fg_acc)
        print(
            f"Epoch {epoch:02d} | "
            f"train {train_loss / len(train_loader):.4f} | "
            f"val {val_loss / len(val_loader):.4f} | "
            f"px acc {avg_pix:.4f} | fg acc {avg_fg:.4f}"
        )

        if avg_fg > best_fg_acc:
            best_fg_acc = avg_fg
            torch.save(model.state_dict(), CHECKPOINT_DIR / "ae_best.pt")
            print(f"  --> saved (fg acc {best_fg_acc:.4f})")

    print("\n--- Compression report ---")
    sample_mask = next(iter(val_loader))[0]
    stats = compression_stats(model, sample_mask)
    print(f"Raw mask   : {stats['raw_bytes']:,} bytes")
    print(f"Compressed : {stats['compressed_bytes']:,} bytes")
    print(f"Ratio      : {stats['ratio']}x")
    print(f"\nBest foreground accuracy: {best_fg_acc:.4f}")


if __name__ == "__main__":
    train()
