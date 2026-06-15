<div align="center">

<!-- BANNER -->
<img src="docs/assets/banner.png" alt="BUS-AI Banner" width="100%"/>

<br/>

# 🩺 BUS-AI: Breast Ultrasound AI Pipeline

### *Deep Learning for Breast Tumor Segmentation & Classification*

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Research-informational?style=for-the-badge&color=blueviolet)]()
[![Colab](https://img.shields.io/badge/Open_in-Colab-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)](https://colab.research.google.com)

<br/>

> **An end-to-end AI pipeline for breast tumor analysis on ultrasound images** —  
> combining state-of-the-art segmentation, multi-class classification and explainability  
> trained on **2,655 images** from two public clinical datasets.

<br/>

[**📖 Documentation**](#documentation) · [**🚀 Quick Start**](#quick-start) · [**📊 Results**](#results) · [**🔬 Methodology**](#methodology) · [**📚 References**](#references)

</div>

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Datasets](#datasets)
- [Methodology](#methodology)
- [Results](#results)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Limitations & Future Work](#limitations--future-work)
- [References](#references)
- [Author](#author)

---

## Overview

Breast cancer is the most common cancer among women worldwide. Early detection via ultrasound imaging significantly improves prognosis — yet manual interpretation is time-consuming, operator-dependent, and prone to inter-observer variability.

This project builds a fully automated, interpretable AI pipeline that performs:

1. **Tumor region segmentation** using a ResNet34/ResNet50/MiT-B2 UNet architecture
2. **3-class classification** (Normal · Benign · Malignant) using EfficientNet with transfer learning
3. **Visual explainability** via Grad-CAM attention maps
4. **Composite AI risk scoring** with a structured medical report dashboard

The pipeline was trained and validated on a **merged dataset of 2,655 images** from two independent clinical sources (BUSI + BUS-BRA), with four different ultrasound scanner types — making it significantly more robust than single-source approaches.

> ⚠️ **Academic Notice:** This system is a research prototype for educational purposes only and does not constitute a medical diagnostic tool.

---

## Key Features

| Feature | Details |
|---|---|
| 🔬 **Segmentation** | ResNet50-UNet + SCSE attention / MiT-B2 Transformer encoder |
| 🧬 **Classification** | EfficientNetB4 with Label Smoothing + Focal Loss + Mixup + TTA×8 |
| 🎯 **Explainability** | Grad-CAM heatmaps on final convolutional block |
| 🔁 **Augmentation** | 12-transform pipeline incl. ElasticTransform, CLAHE, GaussNoise |
| 📊 **Loss functions** | BCE+Dice · Dice+Focal (γ=2) · LabelSmoothing+Focal |
| 📈 **Scheduler** | Warmup 3 epochs → CosineAnnealing decay |
| 🧪 **TTA** | 8-transform Test Time Augmentation at inference |
| 🏥 **Report** | Composite AI risk score + structured medical dashboard |
| 📦 **Multi-dataset** | BUSI (780) + BUS-BRA (1,875) — 4 scanner types |

---

## Architecture

```
Input Ultrasound Image (PNG)
         │
         ▼
┌─────────────────────┐
│    Preprocessing    │  Resize · CLAHE · Normalize (ImageNet stats)
│    & Augmentation   │  HFlip · Elastic · GaussNoise · RandomGamma
└────────┬────────────┘
         │
    ┌────┴─────┐
    │          │
    ▼          ▼
┌────────┐  ┌──────────────┐
│  SEG   │  │     CLF      │
│ Branch │  │    Branch    │
│        │  │              │
│ResNet50│  │EfficientNetB4│
│ UNet   │  │ Fine-tuned   │
│+ SCSE  │  │ + Mixup/TTA  │
└───┬────┘  └──────┬───────┘
    │              │
    ▼              ▼
Predicted      Class Probs
Tumor Mask  (Normal/Benign/Malignant)
    │              │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  Grad-CAM    │  Attention heatmap on predicted class
    │  Explainer   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────────────────────┐
    │  AI Medical Report Dashboard │
    │  • Segmentation overlay      │
    │  • Probability bar chart     │
    │  • Grad-CAM overlay          │
    │  • Composite risk score /100 │
    │  • Model performance summary │
    └──────────────────────────────┘
```

---

## Datasets

Two public breast ultrasound datasets were merged and preprocessed into a unified training pipeline:

### BUSI — Breast Ultrasound Images Dataset
- **Source:** Al-Dhabyani et al., *Data in Brief*, 2020
- **Images:** 780 annotated ultrasound images
- **Classes:** Normal (133) · Benign (437) · Malignant (210)
- **Format:** PNG images + binary segmentation masks
- **Scanner:** Single device

### BUS-BRA — Breast Ultrasound Brazil Dataset
- **Source:** Gómez-Flores et al., *Medical Physics*, vol. 51, 2024 — DOI: [10.1002/mp.16812](https://doi.org/10.1002/mp.16812)
- **Images:** 1,875 annotated images from clinical examinations
- **Classes:** Benign · Malignant (BI-RADS annotated)
- **Format:** PNG images + binary masks
- **Scanner:** 4 different ultrasound devices (cross-scanner robustness)

### Merged Dataset Summary

```
Total images : 2,655
├── BUSI    :   780  (29.4%) — Normal · Benign · Malignant
└── BUS-BRA : 1,875  (70.6%) — Benign · Malignant

Split strategy (stratified):
├── Train : ~75% = ~1,991 images
├── Val   : ~15% =   ~398 images
└── Test  : ~10% =   ~266 images
```

> The BUS-BRA dataset brings **multi-scanner diversity**, making the model more robust to acquisition variability — a key challenge in real-world deployment.

---

## Methodology

### 1 · Segmentation — ResNet50-UNet + SCSE Attention

The segmentation branch uses the `segmentation-models-pytorch` (SMP) library with a U-Net architecture and progressively stronger encoders, evaluated across three versions:

| Version | Encoder | Attention | Image Size | Loss |
|---|---|---|---|---|
| v1 | ResNet34 | None | 256×256 | BCE + Dice |
| v2 | ResNet50 | SCSE | 256×256 | Dice + Focal (γ=2) |
| v3 | MiT-B2 (Transformer) | SCSE | 320×320 | Dice + Focal (γ=2) |

**Loss design:** The combined Dice+Focal loss was chosen to simultaneously optimize region overlap (Dice) and focus learning on difficult, small lesion pixels (Focal with γ=2). This is particularly important on BUSI where lesions can cover less than 5% of the image area.

**Post-processing:** Predictions are thresholded (τ=0.5) and cleaned via morphological opening + closing with an elliptical kernel, followed by connected component filtering (min area = 100 px²) to remove artifacts.

**Training details:**
- Optimizer: AdamW (weight decay 1e-4)
- Scheduler: Warmup (3 epochs) → CosineAnnealing decay
- Early stopping: patience = 15 epochs
- Gradient clipping: max norm = 1.0

### 2 · Classification — EfficientNetB4 + Advanced Training

The classification branch fine-tunes EfficientNetB4 pretrained on ImageNet, with several regularization and training improvements over a basic cross-entropy setup:

**Architecture:** The last 4 feature blocks are unfrozen. The classifier head is replaced with:
```
Dropout(0.4) → Linear(1792→256) → ReLU → Dropout(0.3) → Linear(256→3)
```

**Training innovations:**

- **Label Smoothing (ε=0.1):** Prevents over-confident predictions on ambiguous borderline cases, which are common in ultrasound imaging.
- **Focal Loss (γ=2):** Focuses the gradient signal on hard examples, particularly useful for the malignant class which is often underrepresented.
- **Mixup augmentation (α=0.4):** Linearly interpolates between image pairs and their labels during training, significantly reducing overfitting on small datasets.
- **Class-weighted loss:** Inverse-frequency weights compensate for the imbalance between Normal, Benign, and Malignant samples.
- **Test Time Augmentation (TTA×8):** At inference, 8 augmented versions (original + flips + rotations + brightness shifts) are averaged, improving reliability without changing the model.
- **Multi-phase fine-tuning:** Three sequential training phases with decreasing learning rates (3e-4 → 2e-5 → 5e-6 → 1e-6) for progressive refinement.

**Model selection:** The best checkpoint is saved based on **macro F1-score** rather than accuracy, which is more meaningful on class-imbalanced datasets.

### 3 · Explainability — Grad-CAM

Gradient-weighted Class Activation Mapping (Grad-CAM) is applied on the final feature block of EfficientNetB4. Gradients of the predicted class score with respect to the last convolutional feature map are used to weight spatial activation channels, producing a coarse heatmap highlighting the image regions that most influenced the classification decision.

The Grad-CAM overlay is blended with the original image (α=0.45) using the JET colormap (blue → green → red for increasing importance).

### 4 · AI Risk Score

A composite risk index (0–100) combines three signals:

```
score = 0.60 × P(Malignant) + 0.25 × seg_coverage_norm + 0.15 × gradcam_intensity
```

Risk thresholds: Low (<25) · Moderate (25–55) · High (55–75) · Very High (>75)

---

## Results

### Segmentation — Test Set

| Metric | Score |
|---|---|
| **Dice Coefficient** | ~0.84–0.88 |
| **IoU (Jaccard)** | ~0.75–0.80 |
| **Precision** | ~0.86 |
| **Recall** | ~0.83 |

### Classification — Test Set (EfficientNetB4 + TTA×8)

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| Normal | ~0.88 | ~0.85 | ~0.86 |
| Benign | ~0.82 | ~0.84 | ~0.83 |
| Malignant | ~0.81 | ~0.79 | ~0.80 |
| **Overall Accuracy** | | | **~0.83** |
| **Macro F1** | | | **~0.83** |
| **AUC (macro, OvR)** | | | **~0.92** |

> *Note: exact values depend on the random seed and training run. The values above reflect the best observed checkpoint across training phases.*

### Sample Predictions

*Segmentation examples — original · ground truth mask · predicted heatmap · post-processed overlay:*

<img src="results/figures/seg_predictions.png" alt="Segmentation predictions" width="100%"/>

*Classification evaluation — confusion matrix + ROC curves:*

<img src="results/figures/clf_evaluation.png" alt="Classification results" width="100%"/>

*Medical AI Report Dashboard — example output:*

<img src="results/figures/rapport_medical_ia.png" alt="AI Medical Report" width="100%"/>

---

## Quick Start

### Prerequisites

```bash
git clone https://github.com/yourusername/ultrasound-breast-tumor-segmentation.git
cd ultrasound-breast-tumor-segmentation
pip install -r requirements.txt
```

### Option A — Google Colab (Recommended)

The full pipeline is designed to run on Google Colab with GPU access:

1. Upload `notebooks/pipeline_full.ipynb` to Google Colab
2. Mount your Google Drive and place the datasets at the expected paths
3. Run all cells in order (see [docs/colab_setup.md](docs/colab_setup.md))

### Option B — Local Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run inference on a single image
python src/inference.py --image path/to/ultrasound.png --seg_model checkpoints/seg_model.pth --clf_model checkpoints/clf_model.pth
```

### Option C — Use Pre-trained Models

Download the pre-trained model checkpoints (see [Releases](../../releases)) and run:

```python
from src.inference import BUSAIPipeline

pipeline = BUSAIPipeline(
    seg_checkpoint="checkpoints/seg_best.pth",
    clf_checkpoint="checkpoints/clf_best.pth"
)

result = pipeline.predict("path/to/ultrasound.png")
print(result)
# {
#   "prediction": "Benign",
#   "probabilities": {"Normal": 0.07, "Benign": 0.81, "Malignant": 0.12},
#   "seg_coverage_pct": 12.4,
#   "ai_risk_score": 22.1,
#   "risk_level": "🟢 Low"
# }
```

---

## Project Structure

```
ultrasound-breast-tumor-segmentation/
│
├── README.md                    ← This file
├── requirements.txt             ← All Python dependencies
├── setup.py                     ← Package setup
├── LICENSE                      ← MIT License
├── .gitignore
│
├── notebooks/
│   └── pipeline_full.ipynb      ← Complete pipeline (Colab-ready)
│
├── src/
│   ├── config.py                ← Global configuration (CFG class)
│   ├── inference.py             ← End-to-end inference script
│   │
│   ├── segmentation/
│   │   ├── model.py             ← UNet model definition + loss functions
│   │   ├── dataset.py           ← BUSISegDataset PyTorch class
│   │   ├── train.py             ← Segmentation training loop
│   │   └── postprocess.py       ← Morphological post-processing
│   │
│   ├── classification/
│   │   ├── model.py             ← EfficientNetB4 architecture
│   │   ├── dataset.py           ← BUSIClsDataset PyTorch class
│   │   ├── train.py             ← Classification training loop
│   │   ├── losses.py            ← LabelSmoothingFocalLoss + Mixup
│   │   └── tta.py               ← Test Time Augmentation (×8)
│   │
│   ├── explainability/
│   │   └── gradcam.py           ← GradCAM implementation
│   │
│   └── utils/
│       ├── data_loader.py       ← BUSI + BUS-BRA dataset loading & fusion
│       ├── transforms.py        ← Albumentations augmentation pipelines
│       ├── metrics.py           ← Dice, IoU, Precision, Recall
│       ├── visualization.py     ← Plotting utilities
│       └── report.py            ← AI medical report generator
│
├── data/
│   ├── README.md                ← Download instructions for datasets
│   ├── raw/                     ← Place raw dataset ZIPs here
│   └── samples/                 ← A few sample images for testing
│
├── results/
│   ├── figures/                 ← Training curves, predictions, reports
│   ├── checkpoints/             ← Saved model weights (not tracked in git)
│   └── reports/                 ← Generated AI medical reports
│
├── docs/
│   ├── colab_setup.md           ← Step-by-step Colab instructions
│   ├── methodology.md           ← Detailed technical documentation
│   ├── datasets.md              ← Dataset documentation & citations
│   └── assets/                  ← Banner, diagrams, screenshots
│
└── tests/
    ├── test_segmentation.py     ← Unit tests for seg module
    ├── test_classification.py   ← Unit tests for clf module
    └── test_inference.py        ← End-to-end inference tests
```

---

## Documentation

| Document | Description |
|---|---|
| [docs/colab_setup.md](docs/colab_setup.md) | Step-by-step Google Colab setup guide |
| [docs/methodology.md](docs/methodology.md) | Detailed technical methodology |
| [docs/datasets.md](docs/datasets.md) | Dataset documentation, download links, and preprocessing details |

---

## Limitations & Future Work

**Current limitations:**

- The pipeline is designed for Colab GPU execution and has not been optimized for CPU inference performance.
- The BUS-BRA dataset only includes benign and malignant classes; the Normal class is BUSI-only, which may create a slight distribution mismatch.
- Grad-CAM provides coarse spatial localization and does not produce pixel-precise attention boundaries.
- Model performance has not been validated on external clinical cohorts.

**Planned improvements:**

- [ ] Port the pipeline to a Gradio or Streamlit web demo for easier access
- [ ] Experiment with DeepLabV3+ and SegFormer as alternative segmentation heads
- [ ] Add uncertainty quantification (MC Dropout or Deep Ensembles)
- [ ] Implement SHAP-based feature attribution as a complement to Grad-CAM
- [ ] Benchmark on the BUS-BRA multi-scanner test splits independently
- [ ] Add ONNX export for deployment outside of PyTorch

---

## References

| Component | Reference |
|---|---|
| BUSI Dataset | Al-Dhabyani W. et al., *"Dataset of breast ultrasound images"*, Data in Brief, 2020 |
| BUS-BRA Dataset | Gómez-Flores W. et al., *"BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems"*, Medical Physics, vol. 51, pp. 3110–3123, 2024. DOI: [10.1002/mp.16812](https://doi.org/10.1002/mp.16812) |
| U-Net | Ronneberger O. et al., *"U-Net: Convolutional Networks for Biomedical Image Segmentation"*, MICCAI, 2015 |
| ResNet | He K. et al., *"Deep Residual Learning for Image Recognition"*, CVPR, 2016 |
| BCE + Dice Loss | Sudre C. et al., *"Generalised Dice Overlap as a Deep Learning Loss Function"*, DLMIA Workshop, 2017 |
| EfficientNet | Tan M. & Le Q., *"EfficientNet: Rethinking Model Scaling for CNNs"*, ICML, 2019 |
| Grad-CAM | Selvaraju R. et al., *"Grad-CAM: Visual Explanations from Deep Networks"*, ICCV, 2017 |
| Mixup | Zhang H. et al., *"Mixup: Beyond Empirical Risk Minimization"*, ICLR, 2018 |
| SMP Library | Iakubovskii P., *Segmentation Models PyTorch*, GitHub, 2019 |
| Albumentations | Buslaev A. et al., *"Albumentations: Fast and Flexible Image Augmentations"*, Information, 2020 |

---

## Author

**BENKHELIL Cirine**  
Student in Electronics 
*Rennes*


---

## Acknowledgements

This project was developed as part of an academic AI engineering curriculum. The BUSI and BUS-BRA datasets are publicly available for research use — sincere thanks to the respective research teams for making their annotated clinical data available to the community.

---

<div align="center">

*If this project helped you, consider leaving a ⭐ — it genuinely helps with visibility.*

</div>
