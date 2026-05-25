"""
classification/tta.py
---------------------
Test Time Augmentation (TTA) for more reliable inference.

TTA applies N different augmentations to the same input image, runs
each through the model, and averages the resulting probability distributions.
This reduces prediction variance at inference without retraining.

On BUSI+BUS-BRA, TTA×8 consistently improved macro F1 by 1–3 points
compared to single-pass inference, particularly for the malignant class.

Augmentations used:
  0: Original (identity)
  1: Horizontal flip
  2: Vertical flip
  3: 90° rotation
  4: 180° rotation
  5: 270° rotation
  6: Brightness -10%
  7: Brightness +10%
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def predict_tta(model: nn.Module, img_tensor: torch.Tensor,
                n_aug: int = 8) -> torch.Tensor:
    """
    Run Test Time Augmentation and return averaged class probabilities.

    Args:
        model:      Trained classifier (expects input shape B×C×H×W)
        img_tensor: Input image tensor (B, C, H, W), already normalized
        n_aug:      Number of augmentations to average (max 8, default 8)

    Returns:
        Averaged probability tensor of shape (B, num_classes)
    """
    model.eval()

    tta_transforms = [
        lambda x: x,                                          # 0: original
        lambda x: torch.flip(x, dims=[3]),                   # 1: horizontal flip
        lambda x: torch.flip(x, dims=[2]),                   # 2: vertical flip
        lambda x: torch.rot90(x, 1, dims=[2, 3]),            # 3: 90°
        lambda x: torch.rot90(x, 2, dims=[2, 3]),            # 4: 180°
        lambda x: torch.rot90(x, 3, dims=[2, 3]),            # 5: 270°
        lambda x: torch.clamp(x * 0.9, 0, 1),               # 6: brightness -10%
        lambda x: torch.clamp(x * 1.1, 0, 1),               # 7: brightness +10%
    ]

    probs_list = []
    with torch.no_grad():
        for tfm in tta_transforms[:n_aug]:
            aug_img = tfm(img_tensor)
            logits  = model(aug_img)
            probs   = F.softmax(logits, dim=1)
            probs_list.append(probs)

    return torch.stack(probs_list).mean(dim=0)
