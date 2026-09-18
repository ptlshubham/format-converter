"""
De-Fringing & Color Decontamination Module
Removes background color halos (green spill, warm stage curtain reflections, ambient lighting)
strictly along semi-transparent boundary pixels without modifying interior subject colors.
"""

import numpy as np
import cv2


def defringe_boundary(
    rgb_image: np.ndarray,
    alpha: np.ndarray,
    defringe_strength: float = 0.5,
) -> np.ndarray:
    """
    Decontaminates boundary RGB pixels:
    - Identifies solid foreground (alpha > 235) vs transition boundary (5 < alpha <= 235)
    - Dilates true solid foreground colors into the transition zone
    - Blends dilated foreground colors with original RGB based on transparency and slider strength
    - Fully preserves interior subject pixels (alpha > 235) without color modification
    """
    if defringe_strength <= 0.01:
        return rgb_image.copy()

    # Support both float32 [0.0, 1.0] and uint8 [0, 255]
    if alpha.dtype != np.uint8:
        alpha_u8 = np.clip(np.round(alpha * 255.0), 0, 255).astype(np.uint8)
    else:
        alpha_u8 = alpha

    solid_fg = (alpha_u8 > 235).astype(np.uint8)
    transition = ((alpha_u8 > 5) & (alpha_u8 <= 235)).astype(np.uint8)

    # If no solid foreground or transition, return original
    if np.sum(solid_fg) == 0 or np.sum(transition) == 0:
        return rgb_image.copy()

    # Dilation kernel proportional to defringe strength
    k_size = max(3, int(defringe_strength * 7))
    if k_size % 2 == 0:
        k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))

    # Inpaint/dilate true foreground color into boundary
    dilated_rgb = np.zeros_like(rgb_image)
    for c in range(3):
        channel_solid = np.where(solid_fg > 0, rgb_image[:, :, c], 0)
        dilated_rgb[:, :, c] = cv2.dilate(channel_solid, kernel, iterations=1)

    # Blend original and decontaminated colors exclusively in transition zone
    orig_f = rgb_image.astype(np.float32)
    dilated_f = dilated_rgb.astype(np.float32)

    # Weight blending by (1 - alpha_normalized) and defringe_strength
    alpha_norm = (alpha.astype(np.float32) / 255.0)[:, :, np.newaxis]
    blend_factor = (1.0 - alpha_norm) * defringe_strength * (transition[:, :, np.newaxis] > 0)

    decontaminated = orig_f * (1.0 - blend_factor) + dilated_f * blend_factor
    return np.clip(decontaminated, 0, 255).astype(np.uint8)
