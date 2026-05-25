"""
classification/losses.py
------------------------
Advanced loss functions and training utilities for breast lesion classification.

The standard CrossEntropyLoss was progressively replaced with more sophisticated
alternatives to address two key challenges on BUSI+BUS-BRA:
  1. Class imbalance (fewer malignant cases)
  2. Ambiguous boundary cases (benign vs malignant is not always clear-cut)

Implemented utilities:
  - LabelSmoothingFocalLoss: combines label smoothing with focal weighting
  - mixup_data: Mixup augmentation for pairs of training samples
  - mixup_criterion: Combined loss for Mixup training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothingFocalLoss(nn.Module):
    """
    Combined Label Smoothing + Focal Loss for multi-class classification.

    Label Smoothing (ε=0.1):
        Replaces hard 0/1 targets with soft values (ε/(C-1) and 1-ε).
        Prevents the model from becoming overconfident on ambiguous ultrasound cases,
        which is especially important when radiologists themselves disagree on labels.

    Focal Loss (γ=2):
        Down-weights easy examples and focuses training on hard ones.
        In practice, this helps the model better learn the malignant class,
        which is often the smallest and hardest to classify correctly.

    Args:
        num_classes:   Number of output classes (3 for this pipeline)
        smoothing:     Label smoothing coefficient ε (default: 0.1)
        gamma:         Focal loss focusing parameter (default: 2.0)
        class_weights: Optional per-class weight tensor (for imbalance correction)
    """

    def __init__(self, num_classes: int = 3, smoothing: float = 0.1,
                 gamma: float = 2.0, class_weights: torch.Tensor = None):
        super().__init__()
        self.num_classes   = num_classes
        self.smoothing     = smoothing
        self.gamma         = gamma
        self.class_weights = class_weights

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        confidence = 1.0 - self.smoothing
        smooth_val = self.smoothing / (self.num_classes - 1)

        # Build smoothed target distribution
        with torch.no_grad():
            smooth_targets = torch.full_like(logits, smooth_val)
            smooth_targets.scatter_(1, targets.unsqueeze(1), confidence)

        # Focal weighting: (1 - p_correct)^gamma
        log_probs = F.log_softmax(logits, dim=1)
        probs     = log_probs.exp()
        focal_w   = (1 - probs) ** self.gamma

        loss = -(smooth_targets * focal_w * log_probs).sum(dim=1)

        # Optional per-class weighting
        if self.class_weights is not None:
            w    = self.class_weights[targets]
            loss = loss * w

        return loss.mean()


def mixup_data(x: torch.Tensor, y: torch.Tensor,
               alpha: float = 0.4) -> tuple:
    """
    Apply Mixup augmentation to a batch of images and labels.

    Mixup creates convex combinations of training examples and their labels:
        x_mix = λ·x_i + (1-λ)·x_j
        y_mix = (λ·y_i, (1-λ)·y_j)  [handled at loss time]

    This encourages the model to behave linearly in-between training samples,
    acting as a strong regularizer particularly effective on small medical datasets.

    Args:
        x:     Image batch tensor (B, C, H, W)
        y:     Label batch tensor (B,)
        alpha: Beta distribution parameter; higher → more mixing (default: 0.4)

    Returns:
        (mixed_x, y_a, y_b, lam) — mixed images and both label sets with lambda
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    batch_size = x.size(0)
    idx        = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[idx]
    y_a, y_b = y, y[idx]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(criterion: nn.Module, logits: torch.Tensor,
                    y_a: torch.Tensor, y_b: torch.Tensor,
                    lam: float) -> torch.Tensor:
    """
    Compute Mixup loss as a weighted combination of two label losses.

    Args:
        criterion: The base loss function (e.g. LabelSmoothingFocalLoss)
        logits:    Model output logits
        y_a, y_b: Original and permuted labels
        lam:       Mixup interpolation factor

    Returns:
        Scalar loss tensor
    """
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
