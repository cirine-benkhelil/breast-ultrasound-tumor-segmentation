"""
utils/transforms.py
--------------------
Albumentations augmentation pipelines for segmentation and classification.

Three progressive augmentation strategies are defined:

  - get_transforms(mode):       Standard pipeline — used for both branches
  - get_transforms_seg_v2(mode): Enhanced pipeline for Transformer encoder (320×320)
                                  with ElasticTransform and GridDistortion

Design notes:
  - GaussNoise and RandomGamma were added specifically to handle the cross-scanner
    variability introduced by the BUS-BRA dataset (4 scanner types).
  - CLAHE (Contrast Limited Adaptive Histogram Equalization) improves lesion
    visibility in low-contrast ultrasound regions.
  - ElasticTransform simulates tissue deformation, improving generalization
    to diverse patient anatomy.
  - All augmentations are within medically reasonable bounds to avoid producing
    unrealistic-looking images that could mislead the model.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.config import CFG


def get_transforms(mode: str = "train") -> A.Compose:
    """
    Standard augmentation pipeline (256×256).

    Used for:
      - Segmentation baseline (v1 + v2)
      - Classification training and evaluation

    Args:
        mode: "train" applies full augmentation; "val"/"test" applies only resize + normalize
    """
    if mode == "train":
        return A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.2),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.4),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4),
            A.GaussianBlur(blur_limit=(3, 5), p=0.2),
            A.GaussNoise(std_range=(0.02, 0.1), p=0.3),     # multi-scanner robustness
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),    # scanner brightness variability
            A.CLAHE(clip_limit=2.0, p=0.3),
            A.Normalize(mean=CFG.MEAN, std=CFG.STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CFG.IMG_SIZE, CFG.IMG_SIZE),
            A.Normalize(mean=CFG.MEAN, std=CFG.STD),
            ToTensorV2(),
        ])


def get_transforms_seg_v2(mode: str = "train") -> A.Compose:
    """
    Enhanced augmentation pipeline (320×320) for MiT-B2 Transformer encoder.

    Uses a larger input resolution and additional geometric augmentations
    (ElasticTransform, GridDistortion) that are particularly effective
    for segmentation of deformable anatomical structures.

    Args:
        mode: "train" or "val"/"test"
    """
    if mode == "train":
        return A.Compose([
            A.Resize(CFG.IMG_SIZE_V2, CFG.IMG_SIZE_V2),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15, p=0.5),
            A.ElasticTransform(p=0.3),       # tissue deformation simulation
            A.GridDistortion(p=0.2),         # scanner-specific geometric variation
            A.RandomBrightnessContrast(p=0.4),
            A.CLAHE(clip_limit=3.0, p=0.4),
            A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
            A.Normalize(mean=CFG.MEAN, std=CFG.STD),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(CFG.IMG_SIZE_V2, CFG.IMG_SIZE_V2),
            A.Normalize(mean=CFG.MEAN, std=CFG.STD),
            ToTensorV2(),
        ])
