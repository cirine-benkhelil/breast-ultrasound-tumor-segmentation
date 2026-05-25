# Datasets

This project uses two publicly available breast ultrasound datasets. Neither dataset is included in this repository — please download them directly from the sources below and respect the respective licenses.

---

## BUSI — Breast Ultrasound Images Dataset

| Field | Details |
|---|---|
| **Full name** | Breast Ultrasound Images Dataset |
| **Reference** | Al-Dhabyani W., Gomaa M., Khaled H., Fahmy A. *"Dataset of breast ultrasound images"*, Data in Brief, vol. 28, 2020. DOI: [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863) |
| **Download** | [Kaggle](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset) |
| **Images** | 780 PNG images + binary segmentation masks |
| **Scanner** | Single ultrasound device |
| **Image size** | Varies (approx. 500×400 to 800×600) |

### Class Distribution

| Class | Images |
|---|---|
| Normal | 133 |
| Benign | 437 |
| Malignant | 210 |
| **Total** | **780** |

### Directory Structure

```
Dataset_BUSI_with_GT/
├── normal/
│   ├── normal (1).png
│   ├── normal (1)_mask.png
│   └── ...
├── benign/
│   ├── benign (1).png
│   ├── benign (1)_mask.png
│   └── ...
└── malignant/
    ├── malignant (1).png
    ├── malignant (1)_mask.png
    └── ...
```

Mask files follow the naming convention: `<image_name>_mask.png`.

---

## BUS-BRA — Breast Ultrasound Brazil Dataset

| Field | Details |
|---|---|
| **Full name** | Breast Ultrasound Brazil Dataset |
| **Reference** | Gómez-Flores W., de Albuquerque Pereira W.C., Bhatt D.L. *"BUS-BRA: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems"*, Medical Physics, vol. 51, pp. 3110–3123, 2024. DOI: [10.1002/mp.16812](https://doi.org/10.1002/mp.16812) |
| **Download** | [Zenodo (upon request from authors)](https://doi.org/10.5281/zenodo.8231412) |
| **Images** | 1,875 PNG images + binary masks + CSV metadata |
| **Scanners** | 4 different ultrasound devices (multi-scanner) |

### Class Distribution

| Class | Images |
|---|---|
| Benign | ~1,100 |
| Malignant | ~775 |
| **Total** | **1,875** |

> BUS-BRA does not include normal cases.

### Directory Structure (after extraction)

```
BUSBRA/
├── Images/
│   ├── bus_0001-l.png
│   ├── bus_0001-r.png
│   └── ...
├── Masks/
│   ├── mask_0001-l.png
│   ├── mask_0001-r.png
│   └── ...
└── metadata.csv   (columns: ID, Pathology, BIRADS, Device)
```

### Metadata CSV Columns

| Column | Description |
|---|---|
| `ID` | Image identifier (e.g., `bus_0001-l`) |
| `Pathology` | `benign` or `malignant` |
| `BIRADS` | BI-RADS category (2–5) |
| `Device` | Ultrasound scanner model |

---

## Merged Dataset Summary

When both datasets are available, the pipeline automatically merges them:

| Source | Images | Classes | Scanners |
|---|---|---|---|
| BUSI | 780 | Normal / Benign / Malignant | 1 |
| BUS-BRA | 1,875 | Benign / Malignant | 4 |
| **Total** | **2,655** | 3 | 5 |

### Expected Data Directory Layout

Place datasets in the `data/raw/` directory or update the paths in `src/config.py`:

```
data/
└── raw/
    ├── archive_segmentation.zip   # BUSI dataset (ZIP from Kaggle)
    └── BUSBRA.zip                 # BUS-BRA dataset (ZIP from Zenodo)
```

The pipeline will extract and index both datasets automatically on first run.

---

## Preprocessing

All images undergo the following preprocessing before being fed to the model:

1. **Color conversion**: BGR → RGB (OpenCV loads as BGR)
2. **Resize**: to 256×256 (or 320×320 for MiT-B2 encoder)
3. **Normalization**: ImageNet mean/std normalization — `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`

Ultrasound images are grayscale by nature but are loaded as 3-channel (RGB) to be compatible with ImageNet pre-trained encoders. This is standard practice and does not hurt performance.

---

## Citation

If you use either dataset in your own work, please cite the original publications:

```bibtex
@article{aldhabyani2020busi,
  title     = {Dataset of breast ultrasound images},
  author    = {Al-Dhabyani, Walid and Gomaa, Mohammed and Khaled, Hussien and Fahmy, Aly},
  journal   = {Data in Brief},
  volume    = {28},
  year      = {2020},
  doi       = {10.1016/j.dib.2019.104863}
}

@article{gomezflores2024busbra,
  title     = {{BUS-BRA}: A Breast Ultrasound Dataset for Assessing Computer-aided Diagnosis Systems},
  author    = {G{\'o}mez-Flores, Wilfrido and de Albuquerque Pereira, Wagner Coelho and Bhatt, Darshan L.},
  journal   = {Medical Physics},
  volume    = {51},
  pages     = {3110--3123},
  year      = {2024},
  doi       = {10.1002/mp.16812}
}
```
