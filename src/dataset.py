"""
PASTIS Dataset — Filter and Loader
Retains only patches that contain at least one pixel of:
  - Meadow   (class 1)
  - Wheat    (class 2)
  - Corn     (class 3)

IMPORTANT — PASTIS label format:
    TARGET_*.npy files are stored as (T, H, W) int16 arrays.
    Each timestep holds the same semantic label repeated.
    We always take index [0] to get a single (H, W) label map.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# Class definitions
PASTIS_CLASSES = {
    0: "Background",
    1: "Meadow",
    2: "Soft winter wheat",
    3: "Corn",
    4: "Winter barley",
    5: "Winter rapeseed",
    6: "Spring barley",
    7: "Sunflower",
    8: "Grapevine",
    9: "Beet",
    10: "Winter triticale",
    11: "Winter durum wheat",
    12: "Fruits/vegetables/flowers",
    13: "Potatoes",
    14: "Leguminous fodder",
    15: "Soybeans",
    16: "Orchard",
    17: "Mixed cereal",
    18: "Sorghum",
    19: "Void label",
}

S2_BANDS = {
    0: "B02 Blue",
    1: "B03 Green",
    2: "B04 Red",
    3: "B05 Red Edge 1",
    4: "B06 Red Edge 2",
    5: "B07 Red Edge 3",
    6: "B08 NIR",
    7: "B8A Narrow NIR",
    8: "B11 SWIR 1",
    9: "B12 SWIR 2",
}


# Helper — load a label array and always return (H, W)
def _load_label(path: str) -> np.ndarray:
    """
    Load a PASTIS TARGET_*.npy file and return a 2D (H, W) int64 array.

    PASTIS stores labels as (T, H, W) where every time slice is identical.
    Taking index [0] gives the canonical semantic label map.
    If the array is already 2D (some versions of PASTIS), it is returned as-is.
    """
    label = np.load(path)  # (T, H, W) or (H, W)
    if label.ndim == 3:
        label = label[0]  # (H, W) — all slices are identical
    assert label.ndim == 2, f"Unexpected label shape {label.shape} in {
        path
    }. Expected (H, W) or (T, H, W)."
    return label.astype(np.int64)  # (H, W)


# filter_patches
def filter_patches(
    data_path: str,
    target_classes: list[int],
    require_all: bool = False,
    min_pixel_fraction: float = 0.0,
) -> list[dict]:
    """
    Scan all TARGET_*.npy annotation files and retain patches that contain
    at least one pixel (or all, if require_all=True) of the target classes.

    Args:
        data_path:           Path to the PASTIS/ root folder.
        target_classes:      List of class IDs to look for e.g. [1, 2, 3].
        require_all:         If True  — patch must contain ALL target classes.
                             If False — patch needs at least ONE.
        min_pixel_fraction:  Fraction of total pixels that must belong to any
                             target class combined. 0.0 = no minimum.

    Returns:
        List of dicts, one per kept patch:
            patch_id          : int
            classes_present   : list[int]
            class_pixel_counts: dict[int, int]
            total_pixels      : int
            target_fraction   : float
    """
    annotation_dir = os.path.join(data_path, "ANNOTATIONS")
    patch_files = sorted(
        f for f in os.listdir(annotation_dir) if f.startswith("TARGET_")
    )

    kept = []
    skipped = 0

    print(
        f"Scanning {len(patch_files)} patches for classes: "
        f"{[PASTIS_CLASSES[c] for c in target_classes]}"
    )
    print(f"Mode: {'ALL classes required' if require_all else 'ANY class sufficient'}")
    print(f"Min pixel fraction: {min_pixel_fraction:.1%}\n")

    for fname in tqdm(patch_files, desc="Filtering"):
        patch_id = int(fname.replace("TARGET_", "").replace(".npy", ""))
        label_path = os.path.join(annotation_dir, fname)

        # Always load as (H, W) — handles both (T,H,W) and (H,W) files
        label = _load_label(label_path)  # (H, W)
        total_px = label.size  # H * W  (correct — 2D now)

        # Count pixels per target class
        class_pixel_counts = {}
        classes_present = []
        for cls in target_classes:
            count = int(np.sum(label == cls))
            class_pixel_counts[cls] = count
            if count > 0:
                classes_present.append(cls)

        # Class presence check
        if require_all:
            passes = set(target_classes) == set(classes_present)
        else:
            passes = len(classes_present) > 0

        if not passes:
            skipped += 1
            continue

        # Minimum pixel fraction check
        target_px_total = sum(class_pixel_counts.values())
        if target_px_total / total_px < min_pixel_fraction:
            skipped += 1
            continue

        kept.append(
            {
                "patch_id": patch_id,
                "classes_present": classes_present,
                "class_pixel_counts": class_pixel_counts,
                "total_pixels": total_px,
                "target_fraction": round(target_px_total / total_px, 4),
            }
        )

    print(f"\nKept:    {len(kept)} patches")
    print(f"Skipped: {skipped} patches")

    print("\nPer-class breakdown in kept patches:")
    for cls in target_classes:
        patches_with_cls = sum(1 for p in kept if cls in p["classes_present"])
        total_px_cls = sum(p["class_pixel_counts"][cls] for p in kept)
        print(
            f"  Class {cls} ({PASTIS_CLASSES[cls]:20s}): "
            f"{patches_with_cls} patches, {total_px_cls:,} pixels total"
        )

    return kept


# PASTISCropDataset
class PASTISCropDataset(Dataset):
    """
    PyTorch Dataset for PASTIS filtered to target crop classes.

    Each sample:
        image    : (T, C, H, W)  float32 — normalised S2 time series
        label    : (H, W)        int64   — class IDs (0=bg, 1=meadow, 2=wheat, 3=corn)
        mask     : (H, W)        bool    — True where any target class present
        patch_id : int
    """

    def __init__(
        self,
        data_path: str,
        filtered_meta: list[dict],
        target_classes: list[int] = [1, 2, 3],
        normalize: bool = True,
        s2_mean: list[float] | None = None,
        s2_std: list[float] | None = None,
    ):
        self.data_path = data_path
        self.meta = filtered_meta
        # list — deterministic order
        self.target_classes = sorted(target_classes)
        self.normalize = normalize
        self.s2_mean = np.array(s2_mean, dtype=np.float32) if s2_mean else None
        self.s2_std = np.array(s2_std, dtype=np.float32) if s2_std else None

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, idx: int) -> dict:
        patch_id = self.meta[idx]["patch_id"]

        # Sentinel-2 time series
        s2_path = os.path.join(self.data_path, "DATA_S2", f"S2_{patch_id}.npy")
        s2 = np.load(s2_path).astype(np.float32)  # (T, C, H, W)

        if self.normalize:
            s2 = s2 / 10000.0  # → [0, 1]

        if self.s2_mean is not None and self.s2_std is not None:
            mean = self.s2_mean[None, :, None, None]  # (1, C, 1, 1)
            std = self.s2_std[None, :, None, None]
            s2 = (s2 - mean) / (std + 1e-6)

        # Semantic label
        label_path = os.path.join(
            self.data_path, "ANNOTATIONS", f"TARGET_{patch_id}.npy"
        )
        label = _load_label(label_path)  # (H, W)  ← always 2D

        # Zero out all non-target classes; keep original class ID for targets
        filtered_label = np.zeros_like(label)  # (H, W)
        for cls in self.target_classes:
            filtered_label[label == cls] = cls  # pixel value = class ID

        mask = filtered_label > 0  # (H, W) bool

        assert filtered_label.ndim == 2, (
            f"filtered_label must be 2D, got {filtered_label.shape}"
        )

        return {
            "image": torch.from_numpy(s2),  # (T, C, H, W)
            "label": torch.from_numpy(filtered_label),  # (H, W)
            "mask": torch.from_numpy(mask),  # (H, W)
            "patch_id": patch_id,
        }

    @staticmethod
    def collate_fn(batch: list[dict]) -> dict:
        """
        Pad variable-length time series so all samples in a batch share T.
        Labels and masks are 2D per sample — stacked directly, no padding.
        """
        max_t = max(b["image"].shape[0] for b in batch)

        images = []
        labels = []
        masks = []
        patch_ids = []

        for b in batch:
            img = b["image"]  # (T, C, H, W)
            t = img.shape[0]
            if t < max_t:
                pad = torch.zeros(max_t - t, *img.shape[1:], dtype=img.dtype)
                img = torch.cat([img, pad], dim=0)  # pad time dim only

            images.append(img)
            # (H, W)  — no padding needed
            labels.append(b["label"])
            masks.append(b["mask"])  # (H, W)
            patch_ids.append(b["patch_id"])

        return {
            "image": torch.stack(images),  # (B, T, C, H, W)
            "label": torch.stack(labels),  # (B, H, W)
            "mask": torch.stack(masks),  # (B, H, W)
            "patch_id": patch_ids,
        }


# Helpers for debugging
# compute_dataset_stats
def compute_dataset_stats(
    data_path: str,
    filtered_meta: list[dict],
    max_patches: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-band mean and std over a subset of patches (training split only).

    Args:
        data_path:     Path to PASTIS/ root.
        filtered_meta: Filtered patch metadata list (training split).
        max_patches:   Number of patches to sample. More = more accurate.

    Returns:
        mean: (C,) float32 — per-band mean in [0, 1] space
        std:  (C,) float32 — per-band std  in [0, 1] space
    """
    subset = filtered_meta[:max_patches]
    all_px = []

    print(f"Computing stats from {len(subset)} patches...")
    for entry in tqdm(subset):
        pid = entry["patch_id"]
        s2 = (
            np.load(os.path.join(data_path, "DATA_S2", f"S2_{pid}.npy")).astype(
                np.float32
            )
            / 10000.0
        )  # (T, C, H, W) in [0, 1]

        _, C, _, _ = s2.shape
        # Flatten all timesteps and spatial pixels → (N, C)
        s2_flat = s2.transpose(1, 0, 2, 3).reshape(C, -1).T
        all_px.append(s2_flat)

    all_px = np.concatenate(all_px, axis=0)  # (N_total, C)
    mean = all_px.mean(axis=0)
    std = all_px.std(axis=0)

    print("\nPer-band mean:", np.round(mean, 4))
    print("Per-band std: ", np.round(std, 4))
    return mean, std


def split_patches(
    filtered_meta: list[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Randomly split filtered patch metadata into train / val / test.
    test_ratio = 1 - train_ratio - val_ratio.

    Args:
        filtered_meta: Output of filter_patches().
        train_ratio:   Fraction of patches for training.
        val_ratio:     Fraction of patches for validation.
        seed:          Random seed for reproducibility.

    Returns:
        (train_meta, val_meta, test_meta)
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(filtered_meta)).tolist()

    n = len(indices)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train_meta = [filtered_meta[i] for i in indices[:n_train]]
    val_meta = [filtered_meta[i] for i in indices[n_train : n_train + n_val]]
    test_meta = [filtered_meta[i] for i in indices[n_train + n_val :]]

    print(
        f"Split → Train: {len(train_meta)} | Val: {len(val_meta)} | Test: {
            len(test_meta)
        }"
    )
    return train_meta, val_meta, test_meta


# Sanity check (run as script)
if __name__ == "__main__":
    PASTIS_ROOT = "/mnt/new_volume/dhruv/datasets/PASTIS"
    TARGET_CLASSES = [1, 2, 3]
    BATCH_SIZE = 4
    NUM_WORKERS = 2

    filtered_meta = filter_patches(
        data_path=PASTIS_ROOT,
        target_classes=TARGET_CLASSES,
        require_all=False,
        min_pixel_fraction=0.05,
    )

    train_meta, val_meta, test_meta = split_patches(filtered_meta)
    mean, std = compute_dataset_stats(PASTIS_ROOT, train_meta, max_patches=100)

    train_ds = PASTISCropDataset(
        data_path=PASTIS_ROOT,
        filtered_meta=train_meta,
        target_classes=TARGET_CLASSES,
        normalize=True,
        s2_mean=mean.tolist(),
        s2_std=std.tolist(),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=PASTISCropDataset.collate_fn,
    )

    print("\n[INFO] Sanity check")
    batch = next(iter(train_loader))
    print(f"image shape : {batch['image'].shape}")  # (B, T, 10, 128, 128)
    print(f"label shape : {batch['label'].shape}")  # (B, 128, 128)
    print(f"mask shape  : {batch['mask'].shape}")  # (B, 128, 128)
    print(f"patch IDs   : {batch['patch_id']}")

    assert batch["label"].ndim == 3, (
        f"label must be (B, H, W), got {batch['label'].shape}"
    )
    assert batch["image"].ndim == 5, (
        f"image must be (B, T, C, H, W), got {batch['image'].shape}"
    )

    for cls_id, cls_name in {1: "Meadow", 2: "Wheat", 3: "Corn"}.items():
        px = (batch["label"] == cls_id).sum().item()
        print(f"  {cls_name:6s} pixels in batch: {px:,}")

    print("\nAll checks passed.")
    print(f"   Train: {len(train_ds)} | Val: {len(val_meta)} | Test: {len(test_meta)}")
