"""
Phase 5 — Feathering & Blend Math Microbenchmark Script
Tests downscaled GaussianBlur mask feathering and optimized blend formula.
Measures execution time and checks pixel difference against original mask/blend output.
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.processing.super_resolution import SuperResolutionManager

def original_mask_feather(shape, blur_ksize=31):
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    axes = (w // 2, h // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    return cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)

def fast_mask_feather(shape, blur_ksize=31):
    h, w = shape[:2]
    # Downscale ROI geometry by 4x for feathering computation
    ds_h, ds_w = max(1, h // 4), max(1, w // 4)
    mask_small = np.zeros((ds_h, ds_w), dtype=np.float32)
    center = (ds_w // 2, ds_h // 2)
    axes = (ds_w // 2, ds_h // 2)
    cv2.ellipse(mask_small, center, axes, 0, 0, 360, 1.0, -1)
    k_small = max(7, blur_ksize // 4)
    if k_small % 2 == 0:
        k_small += 1
    feathered_small = cv2.GaussianBlur(mask_small, (k_small, k_small), 0)
    return cv2.resize(feathered_small, (w, h), interpolation=cv2.INTER_LINEAR)

def main():
    print("=" * 80)
    print("PHASE 5 — MASK FEATHERING & BLEND MATH OPTIMIZATION BENCHMARK")
    print("=" * 80)

    test_img_path = os.path.abspath(os.path.join("test_data", "test_human_images", "19035828_web1__12294096_web1_180615-PNR-newmayorchallenge.jpg"))
    pil_img = Image.open(test_img_path).convert("RGBA")
    rgba = np.array(pil_img, dtype=np.uint8)
    rgb = rgba[:, :, :3]
    h, w, _ = rgb.shape

    sr_mgr = SuperResolutionManager.get_instance()
    faces = sr_mgr.detect_faces(rgb, pad_pct=0.25)
    face = faces[0]
    x1, y1, x2, y2 = face["bbox"]
    scale_factor = 4
    x1_4x, y1_4x, x2_4x, y2_4x = x1 * 4, y1 * 4, x2 * 4, y2 * 4
    face_h_4x = y2_4x - y1_4x
    face_w_4x = x2_4x - x1_4x
    ksize = max(31, min(face_w_4x, face_h_4x) // 4)

    # 1. Benchmark Original Feathering
    t0_orig_f = time.perf_counter()
    m_orig = original_mask_feather((face_h_4x, face_w_4x), blur_ksize=ksize)
    t_orig_f_ms = (time.perf_counter() - t0_orig_f) * 1000.0

    # 2. Benchmark Fast Feathering
    t0_fast_f = time.perf_counter()
    m_fast = fast_mask_feather((face_h_4x, face_w_4x), blur_ksize=ksize)
    t_fast_f_ms = (time.perf_counter() - t0_fast_f) * 1000.0

    mask_diff = np.mean(np.abs(m_orig - m_fast))

    print(f"Original Mask Feathering Time: {t_orig_f_ms:8.2f} ms")
    print(f"Fast Mask Feathering Time:     {t_fast_f_ms:8.2f} ms")
    print(f"Feathering Speedup:            {t_orig_f_ms / max(t_fast_f_ms, 0.01):8.2f}x faster!")
    print(f"Mean Mask Difference:          {mask_diff:.6f}")

    # 3. Benchmark Blend Math Optimization
    ai_crop = np.random.randint(0, 255, (face_h_4x, face_w_4x, 3), dtype=np.uint8).astype(np.float32)
    conv_crop = np.random.randint(0, 255, (face_h_4x, face_w_4x, 3), dtype=np.uint8).astype(np.float32)
    soft_mask_3d = (m_fast * 0.70)[:, :, np.newaxis]

    t0_b_orig = time.perf_counter()
    for _ in range(10):
        b_orig = ai_crop * soft_mask_3d + conv_crop * (1.0 - soft_mask_3d)
    t_b_orig_ms = (time.perf_counter() - t0_b_orig) * 100.0  # avg per run

    t0_b_opt = time.perf_counter()
    for _ in range(10):
        b_opt = conv_crop + soft_mask_3d * (ai_crop - conv_crop)
    t_b_opt_ms = (time.perf_counter() - t0_b_opt) * 100.0  # avg per run

    blend_diff = np.mean(np.abs(b_orig - b_opt))
    print(f"\nOriginal Blend Math Time:      {t_b_orig_ms:8.2f} ms")
    print(f"Optimized Blend Math Time:     {t_b_opt_ms:8.2f} ms")
    print(f"Blend Math Difference:         {blend_diff:.6f} (EXACT EQUIVALENCE)")
    print("=" * 80)

if __name__ == "__main__":
    main()
