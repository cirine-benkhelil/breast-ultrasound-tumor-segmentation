"""
utils/report.py
---------------
AI Medical Report dashboard generator.

Generates a structured, multi-panel figure combining:
  - Original ultrasound image
  - Segmentation heatmap + binary overlay
  - Grad-CAM attention map + overlay
  - Class probability bar chart
  - Circular composite risk score gauge
  - Model performance summary table

The composite risk score integrates three signals:
  score = 0.60 × P(Malignant) + 0.25 × seg_coverage_norm + 0.15 × gradcam_intensity

Risk thresholds:
  [0-25)  → Low (green)
  [25-55) → Moderate (yellow)
  [55-75) → High (orange)
  [75-100] → Very High (red)

⚠️ This dashboard is for academic/research purposes only and does
   not constitute a medical diagnostic tool.
"""

import os

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F

from src.config import CFG


# ── Risk Scoring ──────────────────────────────────────────────────────────────

def compute_ai_risk_score(probs: np.ndarray, seg_coverage: float,
                          gradcam_intensity: np.ndarray) -> float:
    """
    Compute a composite AI risk index (0–100).

    Args:
        probs:            Class probability array [P(Normal), P(Benign), P(Malignant)]
        seg_coverage:     Percentage of image covered by the predicted tumor mask (%)
        gradcam_intensity: Resized Grad-CAM map (H, W), values in [0, 1]

    Returns:
        Composite risk score as a float in [0, 100]
    """
    p_malin   = float(probs[2])
    seg_score = min(seg_coverage / 30.0, 1.0)
    cam_score = float(np.mean(gradcam_intensity > 0.5))
    composite = (CFG.RISK_W_MALIGNANT * p_malin
                 + CFG.RISK_W_SEG     * seg_score
                 + CFG.RISK_W_GRADCAM * cam_score)
    return round(composite * 100, 1)


def risk_level(score: float) -> tuple[str, str]:
    """Return (label, hex_color) for a given risk score."""
    if score < 25: return "🟢 Low",       "#27ae60"
    if score < 55: return "🟡 Moderate",  "#f39c12"
    if score < 75: return "🟠 High",      "#e67e22"
    return              "🔴 Very High",  "#c0392b"


# ── Denormalization ───────────────────────────────────────────────────────────

def denorm(tensor: torch.Tensor) -> np.ndarray:
    """Reverse ImageNet normalization and convert tensor to numpy HWC float."""
    mean = torch.tensor(CFG.MEAN).view(3, 1, 1)
    std  = torch.tensor(CFG.STD).view(3, 1, 1)
    return torch.clamp(tensor.cpu() * std + mean, 0, 1).permute(1, 2, 0).numpy()


# ── Report Generator ──────────────────────────────────────────────────────────

def generate_report(image_path: str, seg_model, clf_model, gradcam,
                    postprocess_fn, transforms_fn, output_dir: str,
                    true_label: str = None, seg_metrics_final: dict = None,
                    overall_acc: float = None, macro_auc: float = None,
                    n_dataset: int = None) -> dict:
    """
    Generate a full AI medical report dashboard for a single ultrasound image.

    Args:
        image_path:        Path to the input PNG image
        seg_model:         Loaded segmentation model (eval mode)
        clf_model:         Loaded classification model (eval mode)
        gradcam:           GradCAM instance
        postprocess_fn:    Callable: pred_np → binary mask (postprocess_mask)
        transforms_fn:     Callable: mode → albumentations Compose
        output_dir:        Directory to save the output figure
        true_label:        Ground-truth class label (optional, for display only)
        seg_metrics_final: Dict with Dice/IoU/Precision/Recall on test set
        overall_acc:       Classification test accuracy
        macro_auc:         Classification macro AUC
        n_dataset:         Total number of training images

    Returns:
        Dict with prediction, probabilities, seg_coverage, risk_score, risk_level
    """
    # ── Load & preprocess ─────────────────────────────────────────────────────
    img_bgr  = cv2.imread(image_path)
    img_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    tf      = transforms_fn("val")
    tensor  = tf(image=img_orig)["image"].unsqueeze(0).to(CFG.DEVICE)
    img_disp = cv2.resize(img_orig, (CFG.IMG_SIZE, CFG.IMG_SIZE)).astype(np.float32) / 255.0

    # ── Segmentation ──────────────────────────────────────────────────────────
    seg_model.eval()
    with torch.no_grad():
        pred_raw = torch.sigmoid(seg_model(tensor))[0, 0].cpu().numpy()
    seg_clean    = postprocess_fn(pred_raw)
    seg_coverage = seg_clean.mean() * 100

    # ── Classification ────────────────────────────────────────────────────────
    clf_model.eval()
    with torch.no_grad():
        logits = clf_model(tensor)
        probs  = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class = int(probs.argmax())
    pred_label = CFG.IDX2LABEL[pred_class]

    # ── Grad-CAM ──────────────────────────────────────────────────────────────
    from src.explainability.gradcam import apply_gradcam_overlay
    cam_map, _ = gradcam(tensor, class_idx=pred_class)
    overlay_gc, cam_resized = apply_gradcam_overlay(img_disp, cam_map)

    # ── Risk Score ────────────────────────────────────────────────────────────
    ai_score   = compute_ai_risk_score(probs, seg_coverage, cam_resized)
    risk_lbl, risk_color = risk_level(ai_score)

    # ── Layout ────────────────────────────────────────────────────────────────
    class_names = CFG.CLASS_NAMES
    diag_color  = {"Normal": "#27ae60", "Benign": "#2980b9", "Malignant": "#c0392b"}[pred_label]
    true_str    = f" | GT: {true_label.capitalize()}" if true_label else ""
    n_total     = n_dataset or "?"

    fig = plt.figure(figsize=(22, 12), facecolor="#f8f9fa")
    gs  = fig.add_gridspec(3, 5, hspace=0.35, wspace=0.35,
                           left=0.04, right=0.96, top=0.88, bottom=0.05)

    fig.suptitle(
        f"🩺 AI REPORT — Breast Ultrasound Analysis\n"
        f"Prediction: {pred_label.upper()}{true_str}  |  "
        f"Suspicion Index: {ai_score}/100 {risk_lbl}  |  "
        f"Trained on {n_total} images (BUSI + BUS-BRA)",
        fontsize=12, fontweight="bold", color=diag_color, y=0.97
    )

    # Row 0: 5 image panels
    for i, (img_data, title, cmap, vmin, vmax) in enumerate([
        (img_disp,    "Original Ultrasound",    None,   None, None),
        (pred_raw,    "Segmentation Heatmap",   "hot",  0,    1),
        (None,        f"Tumor Region ({seg_coverage:.1f}%)", None, None, None),
        (cam_resized, "Grad-CAM",               "jet",  0,    1),
        (overlay_gc,  "Grad-CAM Overlay",       None,   None, None),
    ]):
        ax = fig.add_subplot(gs[0, i])
        if i == 2:
            ax.imshow(img_disp)
            if seg_clean.sum() > 0:
                rgba = np.zeros((*seg_clean.shape, 4), dtype=np.float32)
                rgba[seg_clean > 0] = [0.9, 0.1, 0.1, 0.55]
                ax.imshow(rgba)
        else:
            kw = {"cmap": cmap} if cmap else {}
            if vmin is not None:
                kw.update({"vmin": vmin, "vmax": vmax})
            ax.imshow(img_data, **kw)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    # Row 1: probability bars + gauge + performance table
    ax_prob = fig.add_subplot(gs[1, :2])
    bar_colors = ["#27ae60", "#2980b9", "#c0392b"]
    bars = ax_prob.bar(class_names, probs * 100, color=bar_colors, alpha=0.85,
                       edgecolor="white", linewidth=1.5)
    ax_prob.set_ylim(0, 115)
    ax_prob.set_ylabel("Probability (%)")
    ax_prob.set_title("AI Classifier — Class Probabilities", fontsize=10, fontweight="bold")
    for bar, val in zip(bars, probs * 100):
        ax_prob.text(bar.get_x() + bar.get_width() / 2, val + 2,
                     f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")
    bars[pred_class].set_edgecolor("black")
    bars[pred_class].set_linewidth(3)

    ax_gauge = fig.add_subplot(gs[1, 2])
    ax_gauge.axis("off")
    ax_gauge.set_xlim(-1.3, 1.3)
    ax_gauge.set_ylim(-1.3, 1.3)
    ax_gauge.add_patch(plt.Circle((0, 0), 1.1, color="#ecf0f1", zorder=0))
    arc = np.linspace(-np.pi / 2, -np.pi / 2 + 2 * np.pi * (ai_score / 100), 100)
    ax_gauge.plot(np.cos(arc), np.sin(arc), linewidth=12, color=risk_color, solid_capstyle="round")
    ax_gauge.text(0,  0.1, f"{ai_score}",   ha="center", va="center",
                  fontsize=28, fontweight="bold", color=risk_color)
    ax_gauge.text(0, -0.35, "/100",         ha="center", fontsize=11, color="gray")
    ax_gauge.text(0, -0.65, risk_lbl,       ha="center", fontsize=10,
                  fontweight="bold", color=risk_color)
    ax_gauge.set_title("AI Suspicion Index", fontsize=10, fontweight="bold")

    ax_tbl = fig.add_subplot(gs[1, 3:])
    ax_tbl.axis("off")
    sm = seg_metrics_final or {}
    tdata = [
        ["Segmentation Model",   "ResNet50-UNet + SCSE"],
        ["Dice (test set)",      f'{sm.get("dice",  "N/A"):.4f}' if sm else "N/A"],
        ["IoU (test set)",       f'{sm.get("iou",   "N/A"):.4f}' if sm else "N/A"],
        ["Classification Model", "EfficientNetB4 + TTA×8"],
        ["Accuracy",             f"{overall_acc:.4f}" if overall_acc else "N/A"],
        ["AUC macro (ROC)",      f"{macro_auc:.4f}"   if macro_auc  else "N/A"],
        ["Tumor coverage",       f"{seg_coverage:.1f}%"],
        ["Model confidence",     f"{float(probs.max()) * 100:.1f}%"],
    ]
    tbl = ax_tbl.table(cellText=tdata, colLabels=["Indicator", "Value"],
                       cellLoc="center", loc="center", colWidths=[0.55, 0.35])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#f2f3f4")
    ax_tbl.set_title("Model Performance Summary", fontsize=10, fontweight="bold")

    # Row 2: disclaimer
    ax_disc = fig.add_subplot(gs[2, :])
    ax_disc.axis("off")
    disclaimer = (
        "⚠️  MEDICAL DISCLAIMER: This report is generated by an AI system for academic purposes only.\n"
        "The AI suspicion index is an experimental composite score and does NOT constitute a medical diagnosis.\n"
        "All clinical decisions must be made by a qualified radiologist based on a complete examination."
    )
    ax_disc.text(0.5, 0.5, disclaimer, transform=ax_disc.transAxes,
                 fontsize=8, ha="center", va="center", style="italic",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#ffeeba",
                           alpha=0.8, edgecolor="#f39c12"))

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "ai_medical_report.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#f8f9fa")
    plt.show()

    return {
        "prediction":       pred_label,
        "probabilities":    {CFG.IDX2LABEL[i]: round(float(probs[i]), 4) for i in range(3)},
        "seg_coverage_pct": round(seg_coverage, 2),
        "ai_risk_score":    ai_score,
        "risk_level":       risk_lbl,
    }
