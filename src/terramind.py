"""
Loads the TerraMind backbone and provides preprocessing utilities
to convert PASTIS Sentinel-2 patches into TerraMind's expected format.

TerraMind expects:
  - Modality key : "S2L2A"
  - Shape        : (B, 12, 224, 224)
  - Value range  : standardised with TerraMind pretraining stats
                   (mean ~1000-3000, std ~1400-2100 in raw DN)

PASTIS provides:
  - Shape        : (B, 10, 128, 128)   after time collapse
  - Value range  : [0, 1]              after /10000
  - Missing bands: B01 (coastal aerosol) and B09 (water vapour)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from terratorch import BACKBONE_REGISTRY
from typing import Literal


# TerraMind v1 pretraining standardisation constants for S2L2A
#
# These are the mean and std values computed over the TerraMesh dataset
# (9M spatiotemporally aligned EO samples) used to pretrain TerraMind.
#
# Band order matches full S2L2A spec (12 bands):
#   B01, B02, B03, B04, B05, B06, B07, B08, B8A, B09, B11, B12
#
# Units: raw Digital Numbers (DN) — i.e. values BEFORE dividing by 10000.
# If your pipeline normalises to [0,1] first (/10000), multiply these by
# 1/10000 to get the equivalent stats in [0,1] space.

S2L2A_MEAN_DN = [
    1390.458,  # B01 — Coastal aerosol (60m)
    1503.317,  # B02 — Blue            (10m)
    1718.197,  # B03 — Green           (10m)
    1853.910,  # B04 — Red             (10m)
    2199.100,  # B05 — Red Edge 1      (20m)
    2779.975,  # B06 — Red Edge 2      (20m)
    2987.011,  # B07 — Red Edge 3      (20m)
    3083.234,  # B08 — NIR             (10m)
    3132.220,  # B8A — Narrow NIR      (20m)
    3162.988,  # B09 — Water vapour    (60m)
    2424.884,  # B11 — SWIR 1          (20m)
    1857.648,  # B12 — SWIR 2          (20m)
]

S2L2A_STD_DN = [
    2106.761,  # B01
    2141.107,  # B02
    2038.973,  # B03
    2134.138,  # B04
    2085.321,  # B05
    1889.926,  # B06
    1820.257,  # B07
    1871.918,  # B08
    1753.829,  # B8A
    1797.379,  # B09
    1434.261,  # B11
    1334.311,  # B12
]

# Convert to [0, 1] space (matching PASTIS /10000 normalisation)
S2L2A_MEAN = [v / 10000.0 for v in S2L2A_MEAN_DN]
S2L2A_STD = [v / 10000.0 for v in S2L2A_STD_DN]


# Band mapping
#
# PASTIS exports 10 S2 bands (the 10m and 20m bands only):
#   idx  0  1  2  3  4  5  6  7  8  9
#   band B02 B03 B04 B05 B06 B07 B08 B8A B11 B12
#
# TerraMind expects all 12 S2L2A bands:
#   idx  0   1   2   3   4   5   6   7   8   9   10  11
#   band B01 B02 B03 B04 B05 B06 B07 B08 B8A B09 B11 B12
#
# Missing from PASTIS: B01 (idx 0) and B09 (idx 9) in TerraMind ordering.
# These are inserted as zero-filled channels.

PASTIS_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]
TM_BANDS = [
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
]


def get_model(
    variant: Literal["tiny", "small", "base", "large"] = "small",
) -> nn.Module:
    """
    Load the pretrained TerraMind ViT backbone from Hugging Face.

    TerraMind is a Vision Transformer pretrained on 9M spatiotemporally
    aligned multimodal EO samples (TerraMesh dataset, 500B tokens).
    The backbone encodes raw modality inputs into patch embeddings.

    Architecture (ViT):
        - Image divided into 16×16 patches
        - 224×224 input → 14×14 = 196 patches
        - Each patch encoded as a D-dimensional embedding vector

    Output shape: (B, 196, D) where D depends on variant:
        tiny  →  D = 192
        small →  D = 384   ← default
        base  →  D = 768
        large →  D = 1024

    The model is called with a modality dict:
        output = model({"S2L2A": tensor})   # tensor: (B, 12, 224, 224)

    Args:
        variant: Model size. Larger = better features, more memory.

    Returns:
        nn.Module — TerraMind backbone, weights frozen by default from HF.
                    Gradients flow through it during fine-tuning unless
                    you explicitly freeze with:
                        for p in backbone.parameters(): p.requires_grad = False
    """
    model = BACKBONE_REGISTRY.build(
        f"terramind_v1_{variant}",
        pretrained=True,  # download weights from HF on first call
        modalities=["S2L2A"],  # initialise only the S2L2A encoder embedding
        # (proj, pos_emb, mod_emb for this modality)
    )
    return model  # pyright: ignore


def insert_missing_bands(images: torch.Tensor) -> torch.Tensor:
    """
    Insert zero-filled channels for the two S2 bands missing from PASTIS.

    PASTIS has 10 bands: B02–B04, B05–B07, B08, B8A, B11, B12
    TerraMind needs 12 : B01, B02–B04, B05–B07, B08, B8A, B09, B11, B12

    Insertion points (TerraMind band ordering):
        Position 0  → B01 (coastal aerosol, 60m) — not in PASTIS
        Position 9  → B09 (water vapour,    60m) — not in PASTIS

    Zeroing these out is a reasonable approximation because:
      1. B01 and B09 are 60m resolution bands with limited spatial info
      2. The backbone has seen them during pretraining so their expected
         range is known — zero is within a few std of the normalised mean
      3. The model learns to discount uninformative channels

    Args:
        images: (B, 10, H, W) — 10-band PASTIS S2 tensor

    Returns:
        (B, 12, H, W) — 12-band tensor with zeros at B01 and B09 positions
    """
    B, C, H, W = images.shape
    assert C == 10, f"Expected 10 PASTIS bands, got {C}. PASTIS bands: {PASTIS_BANDS}"

    zero = torch.zeros(B, 1, H, W, device=images.device, dtype=images.dtype)

    return torch.cat(
        [
            zero,  # idx  0 → B01 (coastal aerosol) — missing
            images[:, :8],  # idx 1–8 -> B02 B03 B04 B05 B06 B07 B08 B8A
            zero,  # idx  9 -> B09 (water vapour)    — missing
            images[:, 8:],  # idx 10–11 -> B11 B12
        ],
        dim=1,
    )  # -> (B, 12, H, W)


def standardize(images: torch.Tensor) -> torch.Tensor:
    """
    Apply TerraMind pretraining standardisation to a 12-band S2L2A tensor.

    Formula: z = (x - mean) / std
    where mean and std are the per-band statistics from the TerraMesh
    pretraining dataset, converted to [0, 1] space (divided by 10000).

    This aligns your input distribution with what TerraMind saw during
    pretraining, which is critical for the pretrained weights to produce
    meaningful features. Without this, the backbone effectively sees
    out-of-distribution inputs and the embeddings are unreliable.

    Args:
        images: (B, 12, H, W) float32, values in [0, 1] (after /10000)

    Returns:
        (B, 12, H, W) float32, standardised
    """
    B, C, H, W = images.shape
    assert C == 12, f"Expected 12 bands after band insertion, got {C}"

    mean = torch.tensor(S2L2A_MEAN, dtype=torch.float32, device=images.device)
    std = torch.tensor(S2L2A_STD, dtype=torch.float32, device=images.device)

    # Reshape (12,) -> (1, 12, 1, 1) for broadcasting over (B, C, H, W)
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]

    return (images - mean) / (std + 1e-6)


def resize(images: torch.Tensor, size: int = 224) -> torch.Tensor:
    """
    Bilinear resize to TerraMind's expected spatial resolution.

    TerraMind was pretrained exclusively on 224×224 patches.
    Passing a different resolution causes the positional embeddings
    to misalign with the patch grid, degrading feature quality.

    PASTIS patches are 128×128 → must be upsampled to 224×224.

    Args:
        images: (B, C, H, W) any spatial size
        size:   target height = width (default 224)

    Returns:
        (B, C, size, size)
    """
    if images.shape[-1] == size and images.shape[-2] == size:
        return images  # already correct size, skip interpolation

    return F.interpolate(
        images,
        size=(size, size),
        mode="bilinear",
        align_corners=False,
    )


def preprocess(images: torch.Tensor) -> torch.Tensor:
    """
    Full preprocessing pipeline: PASTIS S2 → TerraMind S2L2A input.

    Pipeline:
        (B, 10, 128, 128)  [0, 1]
              ↓  insert_missing_bands()
        (B, 12, 128, 128)  [0, 1]
              ↓  standardize()
        (B, 12, 128, 128)  standardised
              ↓  resize()
        (B, 12, 224, 224)  standardised  ← ready for TerraMind

    Note on ordering — standardize BEFORE resize:
        Standardisation is channel-wise (no spatial op), so order relative
        to resize doesn't affect correctness. We standardise first to avoid
        interpolating raw reflectance values, then resize the already-scaled
        tensor. Both orderings give identical results but this is cleaner.

    Args:
        images: (B, 10, 128, 128) float32
                Values in [0, 1] — must have been divided by 10000 already.
                This is guaranteed by PASTISCropDataset (normalize=True).

    Returns:
        (B, 12, 224, 224) float32 — standardised, ready for TerraMind backbone
    """
    images = insert_missing_bands(images)  # (B, 10, H, W) -> (B, 12, H, W)
    # (B, 12, H, W) -> (B, 12, H, W) z-scored
    images = standardize(images)
    images = resize(images, size=224)  # (B, 12, H, W) -> (B, 12, 224, 224)
    return images
