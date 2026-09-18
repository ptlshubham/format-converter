"""
Mask Quality Analyzer Module
Evaluates continuous float32 probability maps to detect segmentation pathologies
(empty masks, blown-out masks, heavy border leakage, or extreme fragmentation)
and trigger dynamic alternative model evaluation when necessary.
"""

from typing import Dict, Any, Tuple
import numpy as np
import cv2


class MaskQualityMetrics:
    def __init__(
        self,
        foreground_ratio: float,
        border_contact_ratio: float,
        uncertain_ratio: float,
        num_components: int,
        fragmentation_score: float,
        mean_confidence: float,
        is_acceptable: bool,
        quality_score: float,
        rejection_reason: str = "",
    ):
        self.foreground_ratio = foreground_ratio
        self.border_contact_ratio = border_contact_ratio
        self.uncertain_ratio = uncertain_ratio
        self.num_components = num_components
        self.fragmentation_score = fragmentation_score
        self.mean_confidence = mean_confidence
        self.is_acceptable = is_acceptable
        self.quality_score = quality_score
        self.rejection_reason = rejection_reason

    def to_dict(self) -> Dict[str, Any]:
        return {
            "foreground_ratio": round(self.foreground_ratio, 4),
            "border_contact_ratio": round(self.border_contact_ratio, 4),
            "uncertain_ratio": round(self.uncertain_ratio, 4),
            "num_components": self.num_components,
            "fragmentation_score": round(self.fragmentation_score, 4),
            "mean_confidence": round(self.mean_confidence, 4),
            "is_acceptable": self.is_acceptable,
            "quality_score": round(self.quality_score, 4),
            "rejection_reason": self.rejection_reason,
        }


class MaskQualityAnalyzer:
    """
    Analyzes the quality and structural integrity of a continuous float32 probability mask.
    """

    @classmethod
    def evaluate(cls, prob_map: np.ndarray) -> MaskQualityMetrics:
        h, w = prob_map.shape[:2]
        total_pixels = h * w

        fg_mask = (prob_map >= 0.5)
        fg_count = int(np.sum(fg_mask))
        fg_ratio = float(fg_count / total_pixels) if total_pixels > 0 else 0.0

        # 1. Border contact ratio: check perimeter pixels
        top_border = fg_mask[0, :]
        bottom_border = fg_mask[h - 1, :]
        left_border = fg_mask[:, 0]
        right_border = fg_mask[:, w - 1]
        border_pixels = 2 * w + 2 * h - 4
        border_fg = int(np.sum(top_border) + np.sum(bottom_border) + np.sum(left_border) + np.sum(right_border))
        border_contact_ratio = float(border_fg / border_pixels) if border_pixels > 0 else 0.0

        # 2. Uncertain pixel ratio (0.20 <= prob <= 0.80)
        uncertain_mask = (prob_map >= 0.20) & (prob_map <= 0.80)
        uncertain_ratio = float(np.sum(uncertain_mask) / total_pixels)

        # 3. Connected components analysis
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fg_mask.astype(np.uint8), connectivity=8
        )
        num_components = max(0, num_labels - 1)

        # 4. Fragmentation score: area in small specks (< 0.05% image area)
        min_comp_area = max(16, int(total_pixels * 0.0005))
        speck_area = 0
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_comp_area:
                speck_area += area

        fragmentation_score = float(speck_area / fg_count) if fg_count > 0 else 0.0

        # 5. Mean confidence of foreground pixels
        mean_conf = float(np.mean(prob_map[fg_mask])) if fg_count > 0 else 0.0

        # 6. Viability & Pathology Check
        is_acceptable = True
        rejection_reason = ""

        # Empty mask check
        if fg_ratio < 0.002:
            is_acceptable = False
            rejection_reason = "Empty mask: foreground ratio is near zero (< 0.2%)"
        # Blown out mask check (almost entirely foreground + touching entire border)
        elif fg_ratio > 0.96 and border_contact_ratio > 0.90:
            is_acceptable = False
            rejection_reason = "Blown-out mask: foreground covers > 96% with heavy border contact"
        # Severe fragmentation check
        elif fragmentation_score > 0.60 and num_components > 40:
            is_acceptable = False
            rejection_reason = "Severe fragmentation: subject broken into dozens of noisy specks"

        # 7. Aggregate Quality Score [0.0 - 1.0]
        # High confidence, moderate border contact, low fragmentation, controlled uncertainty
        base_score = 1.0
        if not is_acceptable:
            base_score = 0.2
        else:
            base_score -= min(0.3, fragmentation_score * 0.5)
            base_score -= min(0.2, uncertain_ratio * 0.4)
            if border_contact_ratio > 0.7:
                base_score -= 0.2
            base_score = max(0.1, min(1.0, base_score * (mean_conf if mean_conf > 0 else 0.5)))

        return MaskQualityMetrics(
            foreground_ratio=fg_ratio,
            border_contact_ratio=border_contact_ratio,
            uncertain_ratio=uncertain_ratio,
            num_components=num_components,
            fragmentation_score=fragmentation_score,
            mean_confidence=mean_conf,
            is_acceptable=is_acceptable,
            quality_score=base_score,
            rejection_reason=rejection_reason,
        )
