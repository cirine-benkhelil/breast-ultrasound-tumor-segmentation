# Technical Methodology

## Overview

This document provides a detailed technical description of the BUS-AI pipeline, covering architectural decisions, training strategies, loss function design, and lessons learned during development.

---

## 1. Problem Framing

Breast ultrasound analysis involves two complementary tasks:

- **Segmentation**: identifying *where* the tumor is (pixel-level binary mask)
- **Classification**: determining *what* the tumor is (Normal / Benign / Malignant)

These tasks are handled by separate branches of the pipeline, each with its own architecture and training procedure, then combined at inference time to produce a unified risk assessment.

---

## 2. Segmentation Branch

### 2.1 Architecture Evolution

The segmentation model was developed through three iterations, each addressing specific limitations of the previous version:

**Version 1 — ResNet34-UNet (baseline)**
- Standard U-Net with ResNet34 encoder pre-trained on ImageNet
- Input resolution: 256×256
- Loss: BCE + Dice (0.5/0.5)
- This served as a solid baseline and validated that transfer learning from ImageNet generalizes reasonably well to grayscale-converted ultrasound images

**Version 2 — ResNet50-UNet + SCSE Attention**
- Upgraded to ResNet50 for a richer feature hierarchy
- Added SCSE (Spatial and Channel Squeeze & Excitation) attention in the decoder
- SCSE allows the model to selectively amplify relevant spatial locations and feature channels, which is particularly valuable for small lesions that can be easily overwhelmed by background tissue features
- Changed loss to Dice + Focal (0.6/0.4) to better handle class imbalance within images (lesion pixels are a minority)

**Version 3 — MiT-B2 UNet (Transformer encoder)**
- Replaced the CNN encoder with a Mix Transformer (MiT-B2) from SegFormer
- Transformers capture global context via self-attention, which helps with diffuse or multi-focal lesions where long-range spatial reasoning matters
- Input resolution increased to 320×320 to provide more spatial context
- Batch size reduced to 8 (vs 16) due to the larger image size and model parameters

### 2.2 Loss Function Design

The progression from BCE+Dice to Dice+Focal was motivated by two observations:

1. **Focal loss handles hard pixels better.** With standard BCE, easy background pixels dominate the gradient. Focal loss (γ=2) down-weights these easy examples by a factor of (1-p)^γ, forcing the model to focus on difficult foreground pixels — particularly lesion boundaries.

2. **Dice loss optimizes global overlap.** While Focal handles pixel-level difficulty, Dice ensures the overall mask shape is correct, not just individual pixels.

The 0.6/0.4 split was chosen empirically: higher Dice weight ensures the mask is globally correct, while a moderate Focal weight focuses on boundary refinement without destabilizing training.

### 2.3 Training Schedule

A warmup + cosine annealing schedule was used to prevent training instability:
- Epochs 1–3: Linear warmup from 0 → initial LR (prevents exploding gradients at the start)
- Epochs 4+: Cosine annealing decay to η_min = 1e-7

Early stopping with patience=15 was applied on validation Dice score.

### 2.4 Post-processing

Raw sigmoid predictions contain noise, particularly around the lesion boundary and occasionally spurious activations in non-lesion areas. The post-processing pipeline:

1. Threshold at 0.5 → binary mask
2. Morphological opening (5×5 elliptical kernel) → removes isolated noise pixels
3. Morphological closing (5×5 elliptical kernel) → fills small holes within the lesion
4. Connected component filtering → removes any remaining components smaller than 100 pixels²

The elliptical kernel was chosen because breast lesion boundaries tend to be smooth and rounded rather than angular.

---

## 3. Classification Branch

### 3.1 Architecture

EfficientNetB4 was selected as the backbone for several reasons:

- **Compound scaling**: EfficientNet scales width, depth, and resolution simultaneously, leading to better accuracy/efficiency tradeoffs than ResNet architectures of similar parameter count
- **Pre-training**: ImageNet pre-training provides rich texture and shape features that transfer well to ultrasound, even though the imaging modality is very different
- **B4 vs B0**: B4 (1792 features) outperformed B0 (1280 features) in our experiments, justifying the modest increase in compute

The classifier head replaces the original single linear layer with a two-layer MLP:
```
Dropout(0.4) → Linear(1792→256) → ReLU → Dropout(0.3) → Linear(256→3)
```
The intermediate layer allows the network to learn a compact lesion representation before the final class decision.

### 3.2 Fine-tuning Strategy

Three sequential training phases with progressively lower learning rates:

| Phase | LR | Epochs | Frozen layers |
|---|---|---|---|
| 1 | 3e-4 | 30 | All except last 4 blocks + head |
| 2 | 2e-5 | 20 | None (full fine-tuning) |
| 3 | 5e-6 | 20 | None |

This progressive unfreezing approach is standard practice in transfer learning and prevents the "catastrophic forgetting" that can occur when all layers are unfrozen immediately at a high learning rate.

### 3.3 Advanced Training Techniques

**Label Smoothing (ε=0.1)**

Instead of training with hard labels [0, 0, 1], the model is trained with soft labels [ε/2, ε/2, 1-ε]. This prevents the model from becoming overconfident and improves calibration. On ultrasound images, where benign and malignant cases can look very similar even to human experts, this softness is medically meaningful.

**Focal Loss (γ=2)**

Applied in combination with label smoothing. The focal weighting (1-p)^γ ensures that the gradient contribution from already well-classified examples is reduced, focusing learning on hard cases — particularly important for the malignant class.

**Mixup (α=0.4)**

At each training step, two random images are blended: x_mix = λ·x_i + (1-λ)·x_j. The labels are blended accordingly. This acts as a strong data augmentation and regularizer, reducing overfitting on the relatively small dataset (~2,000 training images).

The α=0.4 parameter controls the intensity of mixing: values too low (≈0.1) have minimal effect; values too high (>0.5) can produce images that are too unrealistic for medical imaging.

**Test Time Augmentation (TTA×8)**

At inference, 8 augmented versions of the input image are generated:
- Original
- Horizontal flip, Vertical flip
- 90°, 180°, 270° rotations
- Brightness ±10%

The class probabilities are averaged across all 8 predictions. This reduces prediction variance and consistently improved macro F1 by 1–3 percentage points in experiments.

**Model Selection on F1-macro**

The best checkpoint is saved based on macro F1-score rather than accuracy. This is important because the dataset is class-imbalanced: accuracy can remain high even if the rare malignant class is poorly classified, whereas F1-macro treats all classes equally.

---

## 4. Explainability — Grad-CAM

Grad-CAM was applied on the last feature block of EfficientNetB4. The implementation uses PyTorch hooks to capture activations and gradients during a single forward+backward pass.

Key implementation note: `register_full_backward_hook` is used instead of the deprecated `register_backward_hook`, which can produce incorrect gradients for non-differentiable operations.

The resulting heatmap is upsampled to the original image resolution using bilinear interpolation (via `cv2.resize`), then blended with the image using JET colormap (blue=low attention → red=high attention).

---

## 5. Multi-Dataset Fusion Challenges

Merging BUSI and BUS-BRA introduced several non-trivial engineering challenges:

**Different mask conventions**: BUSI uses `image_mask.png` naming; BUS-BRA uses `mask_XXXX-x.png`. The data loader auto-detects and normalizes both.

**Different image resolutions**: Images range from 200×200 to 800×600+ across both datasets. All images are resized to a fixed resolution (256 or 320) before training.

**Class label mismatch**: BUS-BRA only has benign/malignant labels (no normal). Normal images in the merged dataset come exclusively from BUSI. This creates a minor distribution shift for the normal class that could affect generalization.

**Scanner variability**: BUS-BRA spans 4 different ultrasound scanner models, introducing acoustic shadow patterns, speckle noise levels, and overall image brightness that differ significantly. GaussNoise and RandomGamma augmentations were specifically added to simulate this cross-scanner variability during training.

---

## 6. Results Analysis

The pipeline achieves strong performance across both tasks:

**Segmentation**: Dice ~0.84–0.88 is competitive with published results on BUSI. The MiT-B2 Transformer encoder generally outperforms ResNet encoders on validation Dice, particularly on small (< 5% area) malignant lesions where global context from self-attention is most valuable.

**Classification**: Accuracy ~83%, AUC ~0.92. The AUC is particularly encouraging as it reflects the model's ability to rank cases correctly even without optimizing for the specific decision threshold.

**Failure modes observed**:
- Segmentation tends to under-predict on very small lesions (< 3% coverage), likely due to the limited number of such examples in the training set
- Classification of normal cases is generally reliable; benign/malignant confusion is the most common error type, which mirrors the difficulty human radiologists face on this task

---

## 7. Limitations

1. **Dataset size**: ~2,655 images is small by deep learning standards. More data would likely improve both segmentation Dice and classification F1 substantially.

2. **No external validation**: The pipeline has not been validated on datasets outside BUSI and BUS-BRA. Performance on images from different patient populations or imaging protocols is unknown.

3. **No calibration analysis**: While TTA improves reliability, the predicted probabilities have not been calibrated (e.g., via temperature scaling). A predicted probability of 0.8 may not correspond to an 80% true positive rate.

4. **Coarse Grad-CAM**: Grad-CAM produces a coarse spatial map because it operates at the resolution of the last feature map (≈8×8 for a 256×256 input). Finer methods (GradCAM++, SHAP) could provide more precise localization.
