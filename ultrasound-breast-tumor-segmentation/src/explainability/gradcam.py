"""
explainability/gradcam.py
--------------------------
Gradient-weighted Class Activation Mapping (Grad-CAM) implementation.

Grad-CAM uses the gradients of the target class score flowing into the final
convolutional layer to produce a coarse localization map highlighting the image
regions most important to the classification decision.

Reference:
    Selvaraju, R. R. et al. "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization." ICCV, 2017.

Usage:
    target_layer = clf_model.features[-1]   # last EfficientNetB4 block
    gradcam = GradCAM(clf_model, target_layer)
    cam, class_idx = gradcam(image_tensor)  # image_tensor: (1, C, H, W)
    overlay, cam_resized = apply_gradcam_overlay(img_np, cam)
"""

import cv2
import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """
    Grad-CAM implementation using forward/backward hooks.

    Hooks capture:
      - activations: feature maps from the target layer (forward pass)
      - gradients:   gradients of the class score w.r.t. those feature maps (backward pass)

    The CAM is computed as: ReLU(sum_k(alpha_k * A_k))
    where alpha_k = global average of grad_k (importance weight for channel k).

    Args:
        model:        Classifier model (nn.Module)
        target_layer: The convolutional layer to hook into (e.g. model.features[-1])
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model      = model
        self.gradients  = None
        self.activations = None

        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, "activations", o)
        )
        target_layer.register_full_backward_hook(
            lambda m, gi, go: setattr(self, "gradients", go[0])
        )

    def __call__(self, x: torch.Tensor,
                 class_idx: int = None) -> tuple[np.ndarray, int]:
        """
        Generate a Grad-CAM attention map for a given input.

        Args:
            x:         Input image tensor (1, C, H, W)
            class_idx: Target class index. If None, uses the predicted class.

        Returns:
            (cam, class_idx): Normalized CAM as a 2D numpy array + predicted class
        """
        self.model.eval()
        out = self.model(x)
        self.model.zero_grad()

        if class_idx is None:
            class_idx = out.argmax(dim=1).item()

        out[0, class_idx].backward()

        if self.gradients is not None and self.activations is not None:
            # Importance weights: global average pooling over spatial dimensions
            weights = self.gradients.mean(dim=(2, 3), keepdim=True)

            # Weighted combination of activation maps, then ReLU
            cam = (weights * self.activations).sum(dim=1).squeeze()
            cam = cam.cpu().detach().numpy()
            cam = np.maximum(cam, 0)  # ReLU

            # Normalize to [0, 1]
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(x[0, 0].cpu().numpy())

        return cam, class_idx


def apply_gradcam_overlay(img_np: np.ndarray, cam: np.ndarray,
                          alpha: float = 0.45) -> tuple[np.ndarray, np.ndarray]:
    """
    Resize the Grad-CAM map and blend it onto the original image.

    Args:
        img_np: Float RGB image (H, W, 3), values in [0, 1]
        cam:    Raw Grad-CAM map (any resolution)
        alpha:  Blend factor for the heatmap (default: 0.45)

    Returns:
        (overlay, cam_resized): Blended image + resized CAM (both numpy)
    """
    cam_resized = cv2.resize(cam, (img_np.shape[1], img_np.shape[0]))
    heatmap     = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap     = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    overlay     = (1 - alpha) * img_np + alpha * heatmap
    return np.clip(overlay, 0, 1), cam_resized
