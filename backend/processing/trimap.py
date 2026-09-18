"""
Trimap Generation Module
Constructs a continuous three-band trimap (0 = Background, 128 = Transition/Unknown, 255 = Foreground)
from continuous float32 probability maps without premature binarization.
Preserves multi-object topological structures and fine hair/edge boundary bands.
"""

from typing import Tuple, Dict, Any
import numpy as np
import cv2


class TrimapResult:
    def __init__(
        self,
        trimap: np.ndarray,
        clean_prob_map: np.ndarray,
        foreground_pixels: int,
        background_pixels: int,
        transition_pixels: int,
    ):
        self.trimap = trimap  # uint8: 0, 128, 255
        self.clean_prob_map = clean_prob_map  # float32: [0.0, 1.0]
        self.foreground_pixels = foreground_pixels
        self.background_pixels = background_pixels
        self.transition_pixels = transition_pixels


def generate_trimap(
    prob_map: np.ndarray,
    sensitivity: float = 0.5,
) -> TrimapResult:
    """
    Generates a 3-band trimap from a continuous probability map.
    - sensitivity (0.0 to 1.0):
      Low (0.1) -> conservative background removal, preserves faint foreground/shadows
      Medium (0.5) -> balanced
      High (1.0) -> aggressive background removal
    """
    h, w = prob_map.shape[:2]
    total_px = h * w

    # Dynamic thresholding based on sensitivity
    # Low sensitivity lowers the foreground threshold to capture faint subjects/edges;
    # High sensitivity raises thresholds to discard ambiguous background.
    fg_thresh = float(np.clip(0.65 + (sensitivity - 0.5) * 0.25, 0.45, 0.85))
    bg_thresh = float(np.clip(0.20 + (sensitivity - 0.5) * 0.20, 0.05, 0.40))

    # Remove isolated background noise specks from prob_map
    clean_prob = prob_map.copy()
    candidate_fg = (clean_prob >= bg_thresh).astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(candidate_fg, connectivity=8)

    min_area = max(16, int(total_px * 0.0003))  # 0.03% min area
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        comp_mask = (labels == lbl)
        mean_p = float(np.mean(clean_prob[comp_mask]))
        max_p = float(np.max(clean_prob[comp_mask]))

        # Suppress tiny isolated specks with low maximum confidence
        if area < min_area and max_p < 0.70 and mean_p < 0.45:
            clean_prob[comp_mask] = 0.0

    # Build trimap
    trimap = np.full((h, w), 128, dtype=np.uint8)
    trimap[clean_prob <= bg_thresh] = 0
    trimap[clean_prob >= fg_thresh] = 255

    # Count pixels
    fg_px = int(np.sum(trimap == 255))
    bg_px = int(np.sum(trimap == 0))
    trans_px = int(np.sum(trimap == 128))

    return TrimapResult(
        trimap=trimap,
        clean_prob_map=clean_prob,
        foreground_pixels=fg_px,
        background_pixels=bg_px,
        transition_pixels=trans_px,
    )
