"""
utils/data_loader.py
--------------------
Dataset loading, parsing, and fusion utilities for BUSI and BUS-BRA.

The two datasets have very different directory structures and metadata formats,
so this module abstracts away all the parsing logic and exposes a clean
unified DataFrame for downstream use.

BUSI structure:
    Dataset_BUSI_with_GT/
    ├── normal/      image.png + image_mask.png
    ├── benign/      image.png + image_mask.png
    └── malignant/   image.png + image_mask.png

BUS-BRA structure:
    BUSBRA/
    ├── Images/      bus_XXXX-x.png
    ├── Masks/       mask_XXXX-x.png
    └── metadata.csv (columns: ID, Pathology, BIRADS, Device)
"""

import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import CFG


# ── BUSI Parser ───────────────────────────────────────────────────────────────

def load_busi(busi_root: str) -> pd.DataFrame:
    """
    Parse the BUSI dataset into a unified records DataFrame.

    Args:
        busi_root: Path to the "Dataset_BUSI_with_GT" folder

    Returns:
        DataFrame with columns: image, mask, label, label_idx, source
    """
    categories = ["normal", "benign", "malignant"]
    records    = []

    for cat in categories:
        cat_dir = Path(busi_root) / cat
        if not cat_dir.exists():
            continue
        for img_path in sorted(cat_dir.glob("*.png")):
            if "_mask" in img_path.name:
                continue
            mask_path = cat_dir / img_path.name.replace(".png", "_mask.png")
            records.append({
                "image":     str(img_path),
                "mask":      str(mask_path) if mask_path.exists() else None,
                "label":     cat,
                "label_idx": CFG.LABEL2IDX[cat],
                "source":    "BUSI",
            })

    df = pd.DataFrame(records)
    print(f"[BUSI] Loaded {len(df)} images | {dict(df['label'].value_counts())}")
    return df


# ── BUS-BRA Parser ────────────────────────────────────────────────────────────

def load_busbra(busbra_extract_dir: str) -> pd.DataFrame:
    """
    Parse the BUS-BRA dataset from its extracted directory.

    BUS-BRA only contains benign and malignant cases (no normal class).
    Image filenames follow the pattern: bus_XXXX-x.png
    Mask filenames follow the pattern:  mask_XXXX-x.png

    Args:
        busbra_extract_dir: Path to the extracted BUS-BRA directory (containing Images/, Masks/, CSV)

    Returns:
        DataFrame with columns: image, mask, label, label_idx, source
        Returns empty DataFrame if the directory or CSV is not found.
    """
    IMGS_DIR  = os.path.join(busbra_extract_dir, "BUSBRA", "Images")
    MASKS_DIR = os.path.join(busbra_extract_dir, "BUSBRA", "Masks")

    # Auto-detect CSV
    csv_path = None
    for root, _, files in os.walk(busbra_extract_dir):
        for f in files:
            if f.endswith(".csv"):
                csv_path = os.path.join(root, f)
                break
        if csv_path:
            break

    if not csv_path or not os.path.exists(IMGS_DIR):
        print("[BUS-BRA] Dataset not found — skipping.")
        return pd.DataFrame()

    df_csv    = pd.read_csv(csv_path)
    label_map = {"benign": "benign", "malignant": "malignant"}
    records   = []

    for _, row in df_csv.iterrows():
        img_id    = str(row["ID"]).strip()
        raw_label = str(row["Pathology"]).lower().strip()
        label     = label_map.get(raw_label)
        if label is None:
            continue

        img_path  = os.path.join(IMGS_DIR,  img_id + ".png")
        mask_name = img_id.replace("bus_", "mask_") + ".png"
        mask_path = os.path.join(MASKS_DIR, mask_name)

        if not os.path.exists(img_path):
            continue

        records.append({
            "image":     img_path,
            "mask":      mask_path if os.path.exists(mask_path) else None,
            "label":     label,
            "label_idx": CFG.LABEL2IDX[label],
            "source":    "BUS-BRA",
        })

    df = pd.DataFrame(records)
    print(f"[BUS-BRA] Loaded {len(df)} images | {dict(df['label'].value_counts())}")
    return df


def merge_datasets(df_busi: pd.DataFrame,
                   df_busbra: pd.DataFrame) -> pd.DataFrame:
    """
    Merge BUSI and BUS-BRA into a single unified DataFrame.

    Args:
        df_busi:   BUSI DataFrame from load_busi()
        df_busbra: BUS-BRA DataFrame from load_busbra()

    Returns:
        Merged DataFrame, or BUSI-only if BUS-BRA is empty
    """
    if len(df_busbra) == 0:
        print("[MERGE] Using BUSI only.")
        return df_busi.copy()

    df = pd.concat([df_busi, df_busbra], ignore_index=True)
    print(f"\n[MERGE] Total: {len(df)} images | BUSI: {len(df_busi)} | BUS-BRA: {len(df_busbra)}")
    print(f"        Distribution: {dict(df['label'].value_counts())}")
    return df


# ── PyTorch Datasets ──────────────────────────────────────────────────────────

class BUSISegDataset(Dataset):
    """PyTorch Dataset for segmentation (image + binary mask pairs)."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row["image"]), cv2.COLOR_BGR2RGB)

        if row["mask"] and os.path.exists(str(row["mask"])):
            msk = cv2.imread(row["mask"], cv2.IMREAD_GRAYSCALE)
            msk = (msk > 127).astype(np.float32)
        else:
            msk = np.zeros((img.shape[0], img.shape[1]), dtype=np.float32)

        if self.transform:
            aug = self.transform(image=img, mask=msk)
            img = aug["image"]
            msk = aug["mask"].unsqueeze(0)

        return img, msk


class BUSIClsDataset(Dataset):
    """PyTorch Dataset for 3-class classification (image + label)."""

    def __init__(self, df: pd.DataFrame, transform=None):
        self.df        = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        row = self.df.iloc[idx]
        img = cv2.cvtColor(cv2.imread(row["image"]), cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(image=img)["image"]

        return img, torch.tensor(row["label_idx"], dtype=torch.long)
