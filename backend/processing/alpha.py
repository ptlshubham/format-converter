"""
RGBA Assembly and Alpha Validation Module
Combines processed RGB channels with the refined alpha matte,
ensuring genuine RGBA transparency and original dimension preservation.
"""

from typing import Tuple, Dict, Any
import numpy as np


def assemble_rgba_result(
    rgb_image: np.ndarray,
    alpha_channel: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Stacks RGB channels with alpha matte and calculates quality validation metrics.
    """
    h, w = rgb_image.shape[:2]
    alpha_h, alpha_w = alpha_channel.shape[:2]

    if (h, w) != (alpha_h, alpha_w):
        raise ValueError(
            f"Dimension mismatch between RGB ({w}x{h}) and Alpha ({alpha_w}x{alpha_h})"
        )

    rgba = np.dstack([rgb_image, alpha_channel])

    # Quality and transparency metrics
    total_pixels = h * w
    alpha_min = int(np.min(alpha_channel))
    alpha_max = int(np.max(alpha_channel))
    opaque_count = int(np.sum(alpha_channel == 255))
    transparent_count = int(np.sum(alpha_channel == 0))
    translucent_count = int(np.sum((alpha_channel > 0) & (alpha_channel < 255)))

    metrics = {
        "width": w,
        "height": h,
        "total_pixels": total_pixels,
        "is_true_alpha": bool(transparent_count > 0 and opaque_count > 0),
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "opaque_pixels": opaque_count,
        "transparent_pixels": transparent_count,
        "boundary_pixels": translucent_count,
        "foreground_ratio": round(opaque_count / total_pixels, 4),
    }

    return rgba, metrics
