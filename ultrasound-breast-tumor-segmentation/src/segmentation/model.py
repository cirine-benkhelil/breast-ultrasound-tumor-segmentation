"""
segmentation/model.py
---------------------
UNet-based segmentation models with progressively stronger encoders.

Three variants are provided, reflecting the iterative development of the pipeline:
  - v1: ResNet34-UNet + BCE+Dice loss (baseline)
  - v2: ResNet50-UNet + SCSE attention + Dice+Focal loss
  - v3: MiT-B2 UNet (Transformer encoder) + SCSE attention + Dice+Focal loss

The SCSE (Spatial and Channel Squeeze & Excitation) attention mechanism is applied
in the decoder, allowing the model to focus on the most relevant spatial locations
and channel activations — which is especially helpful for small breast lesions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

from src.config import CFG


# ── Model Factory ─────────────────────────────────────────────────────────────

def build_seg_model(version: str = "v2") -> nn.Module:
    """
    Build a UNet segmentation model.

    Args:
        version: "v1" (ResNet34), "v2" (ResNet50 + SCSE), or "v3" (MiT-B2 + SCSE)

    Returns:
        A SMP UNet model ready for training.
    """
    configs = {
        "v1": dict(encoder_name="resnet34", encoder_weights="imagenet",
                   in_channels=3, classes=1),
        "v2": dict(encoder_name="resnet50", encoder_weights="imagenet",
                   in_channels=3, classes=1, decoder_attention_type="scse"),
        "v3": dict(encoder_name="mit_b2",   encoder_weights="imagenet",
                   in_channels=3, classes=1, decoder_attention_type="scse"),
    }
    if version not in configs:
        raise ValueError(f"Unknown model version '{version}'. Choose from: {list(configs)}")

    model = smp.Unet(**configs[version])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[SEG] Built UNet-{version} | encoder={configs[version]['encoder_name']} "
          f"| {n_params / 1e6:.1f}M trainable parameters")
    return model.to(CFG.DEVICE)


# ── Loss Functions ────────────────────────────────────────────────────────────

class BCEDiceLoss(nn.Module):
    """
    Combined Binary Cross-Entropy + Dice Loss.

    BCE ensures stable training on the background (majority class),
    while Dice directly optimizes mask overlap between prediction and ground truth.

    Both losses are weighted equally by default (0.5 / 0.5), which worked well
    on the BUSI dataset. For very small lesions, a higher Dice weight is recommended.
    """

    def __init__(self, bce_w: float = 0.5, dice_w: float = 0.5, smooth: float = 1e-6):
        super().__init__()
        self.bce_w  = bce_w
        self.dice_w = dice_w
        self.smooth = smooth
        self.bce    = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)

        probs     = torch.sigmoid(logits)
        inter     = (probs * targets).sum(dim=(2, 3))
        union     = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1 - (2 * inter + self.smooth) / (union + self.smooth)

        return self.bce_w * bce_loss + self.dice_w * dice_loss.mean()


class DiceFocalLoss(nn.Module):
    """
    Combined Dice + Focal Loss.

    Focal Loss penalizes hard-to-classify pixels more strongly (via gamma),
    making the model pay closer attention to difficult lesion boundaries and
    small lesion regions. This combination outperforms BCE+Dice on BUS-BRA
    due to the greater morphological variability across scanner types.

    Args:
        dice_w:  Weight for the Dice component (default: 0.6)
        focal_w: Weight for the Focal component (default: 0.4)
        gamma:   Focal loss focusing parameter (default: 2.0)
        smooth:  Smoothing term to avoid zero division
    """

    def __init__(self, dice_w: float = 0.6, focal_w: float = 0.4,
                 gamma: float = 2.0, smooth: float = 1e-6):
        super().__init__()
        self.dice_w  = dice_w
        self.focal_w = focal_w
        self.gamma   = gamma
        self.smooth  = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)

        # Dice component
        inter     = (probs * targets).sum(dim=(2, 3))
        union     = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_loss = 1 - (2 * inter + self.smooth) / (union + self.smooth)
        dice_loss = dice_loss.mean()

        # Focal component
        bce        = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        focal_map  = (1 - probs.detach()) ** self.gamma
        focal_loss = (focal_map * bce).mean()

        return self.dice_w * dice_loss + self.focal_w * focal_loss


# ── Metrics ───────────────────────────────────────────────────────────────────

def seg_metrics(pred_logits: torch.Tensor, targets: torch.Tensor,
                threshold: float = 0.5) -> dict:
    """
    Compute segmentation metrics from raw logits.

    Returns:
        dict with keys: dice, iou, precision, recall
    """
    preds = (torch.sigmoid(pred_logits) > threshold).float()
    inter = (preds * targets).sum(dim=(2, 3))
    union = preds.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))

    dice  = ((2 * inter + 1e-6) / (union + 1e-6)).mean().item()
    iou   = ((inter + 1e-6) / (union - inter + 1e-6)).mean().item()
    prec  = ((inter + 1e-6) / (preds.sum(dim=(2, 3)) + 1e-6)).mean().item()
    rec   = ((inter + 1e-6) / (targets.sum(dim=(2, 3)) + 1e-6)).mean().item()

    return {"dice": dice, "iou": iou, "precision": prec, "recall": rec}
