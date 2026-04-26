import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm.auto import tqdm


class SegmentationMetrics:
    """Accumulates pixel-wise IoU and accuracy over an epoch."""

    CLASS_NAMES = {1: "Meadow", 2: "Wheat ", 3: "Corn  "}

    def __init__(
        self,
        num_classes: int,
        device: torch.device = torch.device("cpu"),
    ):
        self.num_classes = num_classes
        self.device = device
        self.intersection = torch.zeros(num_classes, device=device)
        self.union = torch.zeros(num_classes, device=device)
        self.correct = torch.tensor(0, device=device)
        self.total = torch.tensor(0, device=device)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """preds: (B, H, W) int64 | labels: (B, H, W) int64"""
        for cls in range(self.num_classes):
            p = preds == cls
            l = labels == cls
            self.intersection[cls] += (p & l).sum()
            self.union[cls] += (p | l).sum()

        # Accuracy — foreground only (exclude background class 0)
        fg_mask = labels > 0
        self.correct += (preds[fg_mask] == labels[fg_mask]).sum()
        self.total += fg_mask.sum()

    def compute(self) -> dict:
        iou = self.intersection / (self.union + 1e-6)  # (num_classes,)
        fg_iou = iou[1:]  # classes 1, 2, 3
        mean_iou = fg_iou.mean().item()
        accuracy = (self.correct / (self.total + 1e-6)).item()

        result = {"mean_iou": mean_iou, "accuracy": accuracy}
        for i, (cls_id, name) in enumerate(self.CLASS_NAMES.items()):
            result[f"iou_{name.strip().lower()}"] = fg_iou[i].item()
        return result

    def print_summary(self, result: dict):
        print(f"    mIoU     : {result['mean_iou']:.4f}")
        print(f"    Accuracy : {result['accuracy']:.4f}")
        for cls_id, name in self.CLASS_NAMES.items():
            key = f"iou_{name.strip().lower()}"
            print(f"    IoU {name}: {result[key]:.4f}")


def save_metrics(train_loss, val_loss, val_stats):
    pass


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
) -> tuple[float, dict]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metrics = SegmentationMetrics(num_classes=num_classes, device=device)

    for batch in tqdm(loader, desc="  Val  ", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        images = images.mean(dim=1)  # (B, 10, 128, 128)
        logits = model(images)  # (B, num_classes, 128, 128)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

        preds = logits.argmax(dim=1)  # (B, 128, 128)
        metrics.update(preds, labels)

    avg_loss = total_loss / total_samples
    result = metrics.compute()
    return avg_loss, result
