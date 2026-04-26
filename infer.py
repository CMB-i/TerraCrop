import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from terramind import get_model, preprocess
from encode.model import MaskDecoder, MaskEncoder

NUM_CLASSES = 4
EMBED_DIM = 384
PATCH_GRID = 14
LABEL_SIZE = 128


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
        image: (B, 10, 128, 128) float32, pixel values in [0, 1] or /10000 normalized
        seg_ckpt_path: path to TerraMind segmentation model checkpoint
        ae_ckpt_path: path to autoencoder checkpoint (for encoder weights)
        device: device to run inference on

    Returns:
        z: (B, latent_dim) compressed latent vector
    """
    device = torch.device(device)

    # Load TerraMind segmenter
    backbone = get_model(variant="small")
    seg_model = TerraMindSegmenter(backbone).to(device)
    seg_ckpt = torch.load(seg_ckpt_path, map_location=device)
    seg_state = seg_ckpt.get("model_state_dict", seg_ckpt)
    seg_model.load_state_dict(seg_state)
    seg_model.eval()

    # Load MaskEncoder
    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_state = ae_ckpt.get("model_state_dict", ae_ckpt)

    # Infer latent_dim from checkpoint
    if "encoder.net.14.weight" in ae_state:
        latent_dim = ae_state["encoder.net.14.weight"].shape[0]
    else:
        latent_dim = 256

    encoder = MaskEncoder(num_classes=NUM_CLASSES, latent_dim=latent_dim).to(device)

    # Extract encoder state (keys like "encoder.*")
    encoder_state = {k.replace("encoder.", ""): v for k, v in ae_state.items() if k.startswith("encoder.")}
    encoder.load_state_dict(encoder_state)
    encoder.eval()

    # Forward pass
    image = image.to(device)

    # Segmentation: (B, 10, 128, 128) -> (B, num_classes, 128, 128)
    logits = seg_model(image)

    # Argmax to get mask: (B, num_classes, 128, 128) -> (B, 128, 128)
    mask = logits.argmax(dim=1)

    # Compress: (B, 128, 128) -> (B, latent_dim)
    z = encoder(mask)

    return z


@torch.no_grad()
def infer_earth_systems(
    z: torch.Tensor,
    ae_ckpt_path: str,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Earth-systems-side inference: decompress latent vector to mask.

    Args:
        z: (B, latent_dim) compressed latent vector
        ae_ckpt_path: path to autoencoder checkpoint (for decoder weights)
        device: device to run inference on

    Returns:
        mask: (B, 128, 128) int64 reconstructed segmentation mask
    """
    device = torch.device(device)

    # Load checkpoint and infer latent_dim
    ae_ckpt = torch.load(ae_ckpt_path, map_location=device)
    ae_state = ae_ckpt.get("model_state_dict", ae_ckpt)

    if "encoder.net.14.weight" in ae_state:
        latent_dim = ae_state["encoder.net.14.weight"].shape[0]
    else:
        latent_dim = 256

    # Load MaskDecoder
    decoder = MaskDecoder(num_classes=NUM_CLASSES, latent_dim=latent_dim).to(device)

    # Extract decoder state (keys like "decoder.*")
    decoder_state = {k.replace("decoder.", ""): v for k, v in ae_state.items() if k.startswith("decoder.")}
    decoder.load_state_dict(decoder_state)
    decoder.eval()

    # Forward pass
    z = z.to(device)

    # Decompress: (B, latent_dim) -> (B, num_classes, 128, 128)
    logits = decoder(z)

    # Argmax to get mask: (B, num_classes, 128, 128) -> (B, 128, 128)
    mask = logits.argmax(dim=1)

    return mask
