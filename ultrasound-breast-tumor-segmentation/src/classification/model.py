"""
classification/model.py
-----------------------
EfficientNetB4-based classifier for 3-class breast lesion categorization.

Design choices:
  - EfficientNetB4 was selected over B0 for its larger feature space (1792 vs 1280),
    providing richer representations at moderate computational cost.
  - Only the last 4 feature blocks are unfrozen during fine-tuning, preserving
    the low-level texture features learned from ImageNet while adapting high-level
    semantic representations to the ultrasound domain.
  - The classifier head uses two linear layers with dropout to reduce overfitting
    on the relatively small BUSI+BUS-BRA dataset (~2,655 images).
"""

import torch
import torch.nn as nn
import torchvision.models as tv_models

from src.config import CFG


def build_clf_model(freeze_mode: str = "partial") -> nn.Module:
    """
    Build a fine-tuned EfficientNetB4 classifier for 3-class ultrasound classification.

    Args:
        freeze_mode: "partial" — unfreeze last 4 blocks (recommended for fine-tuning)
                     "full"    — unfreeze all layers (for phase 2/3 fine-tuning with very low LR)
                     "head"    — freeze all backbone layers, train head only

    Returns:
        EfficientNetB4 model with custom 3-class head, moved to CFG.DEVICE
    """
    model = tv_models.efficientnet_b4(weights="IMAGENET1K_V1")

    # ── Freeze strategy ───────────────────────────────────────────────────────
    if freeze_mode == "partial":
        # Freeze all layers first, then selectively unfreeze the last 4 blocks
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if any(f"features.{i}" in name for i in [5, 6, 7, 8]):
                param.requires_grad = True

    elif freeze_mode == "full":
        for param in model.parameters():
            param.requires_grad = True

    elif freeze_mode == "head":
        for param in model.parameters():
            param.requires_grad = False

    else:
        raise ValueError(f"Unknown freeze_mode '{freeze_mode}'")

    # ── Replace classifier head ───────────────────────────────────────────────
    # EfficientNetB4 outputs 1792 features before the classifier
    in_feats = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=CFG.DROPOUT_CLF, inplace=True),
        nn.Linear(in_feats, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.3),
        nn.Linear(256, 3),  # 3 classes: Normal / Benign / Malignant
    )
    # Always keep the head trainable regardless of freeze_mode
    for param in model.classifier.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[CLF] Built EfficientNetB4 | freeze_mode={freeze_mode} "
          f"| {n_trainable:,} trainable parameters")

    return model.to(CFG.DEVICE)
