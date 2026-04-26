import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NUM_CLASSES = 4
LATENT_DIM = 384


class MaskEncoder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM):
        super().__init__()
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
        x = F.one_hot(mask, num_classes=NUM_CLASSES).permute(0, 3, 1, 2).float()
        return self.net(x)


class MaskDecoder(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM):
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
    def __init__(self, num_classes=NUM_CLASSES, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = MaskEncoder(num_classes, latent_dim)
        self.decoder = MaskDecoder(num_classes, latent_dim)

    def forward(self, mask: torch.Tensor):
        z = self.encoder(mask)
        return self.decoder(z)

    def compress(self, mask: torch.Tensor) -> bytes:
        self.eval()
        with torch.no_grad():
            z = self.encoder(mask)
        return z.cpu().numpy().tobytes()

    def decompress(self, data: bytes, device) -> torch.Tensor:
        z = (
            torch.tensor(np.frombuffer(data, dtype=np.float32).copy())
            .reshape(1, LATENT_DIM)
            .to(device)
        )
        with torch.no_grad():
            logits = self.decoder(z)
        return logits.argmax(dim=1)
