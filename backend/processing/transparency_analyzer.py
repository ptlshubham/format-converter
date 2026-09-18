"""
Transparency Analyzer Module
Detects whether an image contains transparent or semi-transparent subjects
(e.g., glassware, wine glasses, perfume bottles, clear plastic cups, spectacles, sunglasses)
using classical computer vision heuristics (internal fractional probability distribution,
edge gradients, and local contrast) to ensure fractional alpha and internal reflections
are preserved without premature binarization.
"""

from typing import Dict, Any
import numpy as np
import cv2


class TransparencyResult:
    def __init__(
        self,
        transparency_type: str,
        is_transparent: bool,
        fractional_ratio: float,
        confidence: float,
        reasoning: str,
    ):
        self.transparency_type = transparency_type  # "OPAQUE", "SEMI_TRANSPARENT", "TRANSPARENT"
        self.is_transparent = is_transparent
        self.fractional_ratio = fractional_ratio
        self.confidence = confidence
        self.reasoning = reasoning

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transparency_type": self.transparency_type,
            "is_transparent": self.is_transparent,
            "fractional_ratio": round(self.fractional_ratio, 4),
            "confidence": round(self.confidence, 4),
            "reasoning": self.reasoning,
        }


class TransparencyAnalyzer:
    """
    Pure Classical CV analyzer for detecting transparency and translucency.
    Operates directly on raw float probability maps, image color variance, and edge gradients.
    Zero AI or neural network dependencies.
    """

    @classmethod
    def analyze(
        cls,
        rgb_image: np.ndarray,
        prob_map: np.ndarray,
        trimap: np.ndarray,
    ) -> TransparencyResult:
        """
        Analyzes spatial probability distribution, internal color variance,
        and specular edge gradients to detect transparent/semi-transparent subjects.
        """
        fg_candidate = (trimap != 0)
        fg_px_count = int(np.sum(fg_candidate))

        if fg_px_count == 0:
            return TransparencyResult(
                transparency_type="OPAQUE",
                is_transparent=False,
                fractional_ratio=0.0,
                confidence=0.9,
                reasoning="Empty foreground mask",
            )

        # 1. Fractional probability ratio inside the candidate object
        # Truly opaque objects have high probability (>0.90) across their interior.
        # Transparent glass/plastic transmits background light, causing continuous fractional probabilities [0.15, 0.85].
        fractional_mask = (prob_map >= 0.18) & (prob_map <= 0.82) & fg_candidate
        fractional_count = int(np.sum(fractional_mask))
        fractional_ratio = float(fractional_count / fg_px_count)

        # 2. Interior edge gradient vs border gradient
        # Glass objects typically feature sharp specular boundary edges with smooth gradient falloff inside
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        edge_magnitude = cv2.magnitude(sobel_x, sobel_y)

        fg_core = (trimap == 255)
        core_px_count = int(np.sum(fg_core))

        interior_edge_mean = 0.0
        if core_px_count > 100:
            interior_edge_mean = float(np.mean(edge_magnitude[fg_core]))

        # 3. Determine transparency classification
        if fractional_ratio > 0.28:
            t_type = "TRANSPARENT"
            is_trans = True
            conf = min(0.95, 0.60 + fractional_ratio * 0.4)
            reason = (
                f"High internal fractional probability ({fractional_ratio:.1%}) "
                f"indicating transparent glass, plastic, or liquid transmission"
            )
        elif fractional_ratio > 0.12 and interior_edge_mean > 15.0:
            t_type = "SEMI_TRANSPARENT"
            is_trans = True
            conf = 0.80
            reason = (
                f"Moderate fractional probability ({fractional_ratio:.1%}) with specular reflection "
                f"edges (gradient: {interior_edge_mean:.1f})"
            )
        else:
            t_type = "OPAQUE"
            is_trans = False
            conf = 0.90
            reason = "Subject core is predominantly solid opaque with sharp boundaries"

        return TransparencyResult(
            transparency_type=t_type,
            is_transparent=is_trans,
            fractional_ratio=fractional_ratio,
            confidence=conf,
            reasoning=reason,
        )
