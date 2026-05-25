"""
segmentation/postprocess.py
----------------------------
Morphological post-processing for predicted segmentation masks.

Raw UNet predictions (after sigmoid) are noisy probability maps.
Post-processing cleans these predictions to produce crisp, clinically
interpretable binary masks. The pipeline applies:

  1. Thresholding       → convert probability map to binary mask
  2. Morphological open → remove small noise / thin connections
  3. Morphological close → fill small holes inside the lesion
  4. Connected components → discard regions smaller than min_area pixels

The elliptical kernel is chosen because breast lesions tend to have
rounded contours, making round-shaped morphological operations more
appropriate than square kernels.
"""

import cv2
import numpy as np


def postprocess_mask(
    pred_np: np.ndarray,
    threshold: float = 0.5,
    min_area: int = 100,
    kernel_size: int = 5,
) -> np.ndarray:
    """
    Clean a raw segmentation probability map into a binary lesion mask.

    Args:
        pred_np:     2D numpy array of sigmoid probabilities (H, W), values in [0, 1]
        threshold:   Binarization threshold (default: 0.5)
        min_area:    Minimum connected component area to keep (pixels²)
        kernel_size: Size of the elliptical morphological kernel

    Returns:
        2D binary numpy array (H, W) with values 0 or 1
    """
    # Step 1 — Binarize
    binary = (pred_np > threshold).astype(np.uint8)

    # Step 2 — Morphological cleaning (open + close)
    kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = cv2.morphologyEx(binary,  cv2.MORPH_OPEN,  kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    # Step 3 — Keep only components above min_area
    nb_comp, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
    final = np.zeros_like(cleaned)
    for i in range(1, nb_comp):  # skip background (component 0)
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            final[labels == i] = 1

    return final


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple = (1.0, 0.1, 0.1),
    alpha: float = 0.55,
) -> np.ndarray:
    """
    Overlay a binary mask on an RGB image with a semi-transparent color.

    Args:
        image: Float RGB image (H, W, 3), values in [0, 1]
        mask:  Binary mask (H, W), values 0 or 1
        color: RGBA color for the overlay (default: red)
        alpha: Transparency of the overlay (default: 0.55)

    Returns:
        Float RGB image with overlay applied
    """
    result = image.copy()
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[mask > 0] = [*color, alpha]

    mask_rgb = rgba[:, :, :3]
    mask_a   = rgba[:, :, 3:4]
    result   = result * (1 - mask_a) + mask_rgb * mask_a
    return np.clip(result, 0, 1)
