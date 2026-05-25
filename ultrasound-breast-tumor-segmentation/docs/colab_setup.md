# Google Colab Setup Guide

This guide walks you through running the full BUS-AI pipeline on Google Colab with GPU acceleration.

---

## Prerequisites

- A Google account with access to Google Drive
- The BUSI dataset ZIP (`archive_segmentation.zip`) — download from [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)
- (Optional) The BUS-BRA dataset ZIP (`BUSBRA.zip`) — available from [Zenodo](https://doi.org/10.5281/zenodo.8231412)

---

## Step 1 — Upload Datasets to Google Drive

Create a folder in your Google Drive (e.g., `MyDrive/MonProjetcancer/`) and upload:

```
MonProjetcancer/
├── archive_segmentation.zip   ← BUSI dataset
└── BUSBRA.zip                 ← BUS-BRA dataset (optional)
```

The pipeline will auto-detect and extract both ZIPs on first run.

---

## Step 2 — Open the Notebook in Colab

1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Click **File → Upload notebook**
3. Select `notebooks/pipeline_full.ipynb` from this repository

Or open directly via GitHub:
- **File → Open notebook → GitHub tab**
- Paste the repository URL

---

## Step 3 — Enable GPU Runtime

Go to **Runtime → Change runtime type → T4 GPU** (free tier) or **A100** (Colab Pro).

Verify your GPU:
```python
import torch
print(torch.cuda.is_available())   # should print True
print(torch.cuda.get_device_name(0))
```

---

## Step 4 — Update Drive Paths

In the notebook, update the `PROJECT_DIR` variable to match your Google Drive folder:

```python
PROJECT_DIR      = '/content/drive/MyDrive/MonProjetcancer'
SEGMENTATION_ZIP = os.path.join(PROJECT_DIR, 'archive_segmentation.zip')
BUSBRA_ZIP       = os.path.join(PROJECT_DIR, 'BUSBRA.zip')
```

---

## Step 5 — Run All Cells

Execute the notebook from top to bottom: **Runtime → Run all**.

Expected runtime on T4 GPU:
- Installation: ~2 min
- Data loading + EDA: ~3 min
- Segmentation training (60 epochs): ~45–90 min
- Classification training (30 epochs): ~20–40 min
- Evaluation + report generation: ~5 min

**Tip**: The pipeline uses early stopping, so training often completes well before the maximum epoch count.

---

## Step 6 — Save Outputs

Model checkpoints and output figures are automatically saved to your Google Drive folder:

```
MonProjetcancer/
├── BEST_ResNet50_UNet_SCSE.pth       ← Best segmentation model
├── EfficientNetB4_clf_BUSI_BUSBRA.pth ← Best classification model
└── figures/
    ├── eda_fusion.png
    ├── seg_training_curves.png
    ├── seg_predictions.png
    ├── clf_training_curves.png
    ├── clf_evaluation.png
    ├── metrics_summary.png
    └── rapport_medical_ia.png
```

---

## Common Issues

**"CUDA out of memory"**
- Reduce `CFG.BATCH_SIZE` to 8 (or 4 for 320×320 inputs)
- Restart the runtime and try again

**"archive_segmentation.zip not found"**
- Double-check the Google Drive path in `PROJECT_DIR`
- Ensure the Drive was mounted successfully (cell 1 output should say "Mounted at /content/drive")

**Slow training**
- Make sure the T4 GPU runtime is selected (not CPU)
- Check GPU usage with `!nvidia-smi` in a code cell

**BUS-BRA not found — pipeline runs on BUSI only**
- This is expected if you only have the BUSI dataset
- The pipeline handles this gracefully and trains on BUSI alone

---

## Memory Optimization Tips

For free Colab (15 GB RAM, T4 16 GB VRAM):

```python
# Use smaller batch for 320×320 inputs
CFG.BATCH_SIZE = 8   # instead of 16

# Use fewer TTA augmentations at inference
N_TTA = 4           # instead of 8

# Clear GPU cache between training runs
import torch
torch.cuda.empty_cache()
```
