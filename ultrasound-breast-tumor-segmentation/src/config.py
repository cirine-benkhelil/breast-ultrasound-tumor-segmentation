"""
config.py
---------
Global configuration for the BUS-AI pipeline.
All hyperparameters, paths, and model settings are centralized here.
"""

import torch


class CFG:
    # ── Image & Training ──────────────────────────────────────────────────────
    IMG_SIZE    = 256        # Default input resolution (256×256)
    IMG_SIZE_V2 = 320        # Larger resolution for Transformer encoder (MiT-B2)
    BATCH_SIZE  = 16         # Reduce to 8 when using 320×320 or limited VRAM
    SEED        = 42

    # ── Epochs ────────────────────────────────────────────────────────────────
    EPOCHS_SEG = 60          # Max segmentation epochs (with early stopping)
    EPOCHS_CLF = 30          # Max classification epochs (with early stopping)

    # ── Learning Rates ────────────────────────────────────────────────────────
    LR_SEG = 1e-4            # Initial segmentation LR (warmup applies)
    LR_CLF = 3e-4            # Initial classification LR

    # ── Data Splits ───────────────────────────────────────────────────────────
    VAL_SPLIT  = 0.15
    TEST_SPLIT = 0.10

    # ── Normalization (ImageNet stats) ────────────────────────────────────────
    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    # ── Label Mapping ─────────────────────────────────────────────────────────
    LABEL2IDX = {"normal": 0, "benign": 1, "malignant": 2}
    IDX2LABEL = {0: "Normal", 1: "Benign", 2: "Malignant"}
    CLASS_NAMES = ["Normal", "Benign", "Malignant"]

    # ── Loss Weights ──────────────────────────────────────────────────────────
    BCE_WEIGHT  = 0.5        # For BCE+Dice combined loss
    DICE_WEIGHT = 0.5
    DICE_FOCAL_WEIGHT = 0.6  # Dice weight in Dice+Focal
    FOCAL_WEIGHT      = 0.4  # Focal weight in Dice+Focal
    FOCAL_GAMMA       = 2.0  # Focal loss gamma (focus on hard examples)

    # ── Classification ────────────────────────────────────────────────────────
    LABEL_SMOOTHING = 0.1
    MIXUP_ALPHA     = 0.4
    N_TTA           = 8      # Number of TTA augmentations at inference
    DROPOUT_CLF     = 0.4

    # ── Segmentation Post-processing ──────────────────────────────────────────
    SEG_THRESHOLD = 0.5      # Sigmoid threshold for binary mask
    MIN_LESION_AREA = 100    # Minimum connected component area (pixels²)

    # ── AI Risk Score Weights ─────────────────────────────────────────────────
    RISK_W_MALIGNANT = 0.60
    RISK_W_SEG       = 0.25
    RISK_W_GRADCAM   = 0.15

    # ── Training stability ────────────────────────────────────────────────────
    GRAD_CLIP_NORM = 1.0
    WEIGHT_DECAY   = 1e-4
    WARMUP_EPOCHS  = 3       # LR warmup before cosine decay

    # ── Early Stopping ────────────────────────────────────────────────────────
    SEG_PATIENCE = 15
    CLF_PATIENCE = 8

    # ── Device ────────────────────────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
