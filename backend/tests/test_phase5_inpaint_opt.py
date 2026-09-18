"""
Phase 5 — Inpainting ROI Optimization Microbenchmark
Tests bounding-box constrained cv2.inpaint for transparent edge dilation.
Verifies execution speedup and strict invariant: opaque_pixels_modified = 0.
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

def dilate_rgb_edges_full(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    mask = (alpha > 0).astype(np.uint8)
    if mask.all() or not mask.any():
        return rgb
    inpaint_mask = (mask == 0).astype(np.uint8) * 255
    return cv2.inpaint(rgb, inpaint_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

def dilate_rgb_edges_roi(rgb: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    mask = (alpha > 0).astype(np.uint8)
    if mask.all() or not mask.any():
        return rgb

    inpaint_mask = (mask == 0).astype(np.uint8) * 255
    y_indices, x_indices = np.where(inpaint_mask > 0)
    if len(y_indices) == 0:
        return rgb

    pad = 5
    h, w = rgb.shape[:2]
    y1 = max(0, int(np.min(y_indices)) - pad)
    y2 = min(h, int(np.max(y_indices)) + 1 + pad)
    x1 = max(0, int(np.min(x_indices)) - pad)
    x2 = min(w, int(np.max(x_indices)) + 1 + pad)

    out_rgb = rgb.copy()
    rgb_crop = rgb[y1:y2, x1:x2, :]
    mask_crop = inpaint_mask[y1:y2, x1:x2]

    inpainted_crop = cv2.inpaint(rgb_crop, mask_crop, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    out_rgb[y1:y2, x1:x2, :] = inpainted_crop
    return out_rgb

def main():
    print("=" * 80)
    print("PHASE 5 — INPAINTING ROI OPTIMIZATION BENCHMARK")
    print("=" * 80)

    test_img_path = os.path.abspath(os.path.join("test_data", "test_human_images", "19035828_web1__12294096_web1_180615-PNR-newmayorchallenge.jpg"))
    pil_img = Image.open(test_img_path).convert("RGBA")
    rgba = np.array(pil_img, dtype=np.uint8)
    rgb = rgba[:, :, :3]
    alpha = np.ascontiguousarray(rgba[:, :, 3])
    h, w = alpha.shape
    cv2.circle(alpha, (w//2, h//2), min(w, h)//3, 0, -1)

    t0_full = time.perf_counter()
    res_full = dilate_rgb_edges_full(rgb, alpha)
    t_full_ms = (time.perf_counter() - t0_full) * 1000.0

    t0_roi = time.perf_counter()
    res_roi = dilate_rgb_edges_roi(rgb, alpha)
    t_roi_ms = (time.perf_counter() - t0_roi) * 1000.0

    # Invariant check: Verify opaque pixels (alpha == 255) are unmodified
    opaque_mask = (alpha == 255)
    opaque_diff_full = np.max(np.abs(rgb[opaque_mask].astype(int) - res_full[opaque_mask].astype(int))) if np.any(opaque_mask) else 0
    opaque_diff_roi = np.max(np.abs(rgb[opaque_mask].astype(int) - res_roi[opaque_mask].astype(int))) if np.any(opaque_mask) else 0

    diff_between_outputs = np.mean(np.abs(res_full.astype(float) - res_roi.astype(float)))

    print(f"Full Image Inpainting Time:  {t_full_ms:8.2f} ms")
    print(f"ROI-Constrained Inpainting: {t_roi_ms:8.2f} ms")
    print(f"Inpainting Speedup:          {t_full_ms / max(t_roi_ms, 0.01):8.2f}x faster!")
    print(f"Opaque Pixels Modified (Full): {opaque_diff_full} (Requirement: 0)")
    print(f"Opaque Pixels Modified (ROI):  {opaque_diff_roi} (Requirement: 0)")
    print(f"Difference between Full and ROI: {diff_between_outputs:.6f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
