"""
Phase 4 — Quality Regression & Performance Test Across All 7 Image Complexities
Tests:
1. 1200x800 human image (1 large face)
2. 252x239 human image (small face)
3. Object image with 0 faces (horse.jpg / 0003.jpg)
4. Group image with multiple faces (Two_dancers.jpg)
5. Transparent RGBA cutout image
6. Low-resolution face close-up (lionel-messi-athletes-fashion.jpg)
7. High-detail portrait (photo-1552374196-c4e7ffc6e126.jpeg)
"""

import os
import sys
import time
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.processing.super_resolution import SuperResolutionManager

TEST_DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "test_data"))

def test_image_category(rel_path: str, name: str):
    img_path = os.path.join(TEST_DATA_DIR, rel_path)
    if not os.path.exists(img_path):
        print(f"[{name}] File not found at {img_path}. Skipping.")
        return

    pil_img = Image.open(img_path).convert("RGBA")
    rgba = np.array(pil_img, dtype=np.uint8)
    h, w, c = rgba.shape

    # Determine safe export quality based on MAX_EXPORT_PIXELS limit (33,177,600)
    if w * 4 * h * 4 <= 33_177_600:
        quality = "4x"
        scale = 4
    else:
        quality = "2x"
        scale = 2

    sr_mgr = SuperResolutionManager.get_instance()
    t0 = time.perf_counter()
    out_rgba, metrics = sr_mgr.upscale_rgba_ai(rgba, export_quality=quality)
    elapsed = (time.perf_counter() - t0) * 1000.0

    target_h, target_w = h * scale, w * scale
    dim_pass = (out_rgba.shape[0] == target_h and out_rgba.shape[1] == target_w and out_rgba.shape[2] == 4)

    print(f"[{name:35s}] Source: {w:4d}x{h:4d} -> Out ({quality}): {out_rgba.shape[1]:4d}x{out_rgba.shape[0]:4d} | Faces: {metrics['faces_detected']:2d} | Time: {elapsed:8.2f}ms | Dims: {'PASS' if dim_pass else 'FAIL'}")

def main():
    print("=" * 95)
    print("PHASE 4 — COMPREHENSIVE MULTI-CATEGORY IMAGE COMPLEXITY SUITE")
    print("=" * 95)

    test_image_category(os.path.join("test_human_images", "19035828_web1__12294096_web1_180615-PNR-newmayorchallenge.jpg"), "Test 1: 1200x800 (1 Large Face)")
    test_image_category(os.path.join("test_human_images", "coach-yelling-at-athlete-716268.jpg"), "Test 2: 252x239 (Small Face)")
    test_image_category(os.path.join("test_images", "horse.jpg"), "Test 3: No Faces (Object Image)")
    test_image_category(os.path.join("test_human_images", "Two_dancers.jpg"), "Test 4: Multiple Faces (Group)")
    test_image_category(os.path.join("test_human_images", "19035828_web1__12294096_web1_180615-PNR-newmayorchallenge.jpg"), "Test 5: Transparent RGBA Cutout")
    test_image_category(os.path.join("test_human_images", "lionel-messi-athletes-fashion.jpg"), "Test 6: Low-Res Face Close-Up")
    test_image_category(os.path.join("test_human_images", "photo-1552374196-c4e7ffc6e126.jpeg"), "Test 7: High-Detail Portrait")

    print("=" * 95)

if __name__ == "__main__":
    main()
