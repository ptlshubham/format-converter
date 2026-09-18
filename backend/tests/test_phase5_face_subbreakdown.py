"""
Phase 5 — Sub-breakdown Profiler for Face Preservation
Measures exact timing for bbox prep, face crop, conventional upscale, mask creation, mask feathering, AI crop extraction, float blending, clipping, and placement.
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.processing.super_resolution import SuperResolutionManager

def main():
    print("=" * 80)
    print("PHASE 5 — FACE PRESERVATION SUB-BREAKDOWN PROFILER")
    print("=" * 80)

    test_img_path = os.path.abspath(os.path.join("test_data", "test_human_images", "19035828_web1__12294096_web1_180615-PNR-newmayorchallenge.jpg"))
    pil_img = Image.open(test_img_path).convert("RGBA")
    rgba = np.array(pil_img, dtype=np.uint8)
    rgb = rgba[:, :, :3]
    h, w, _ = rgb.shape

    sr_mgr = SuperResolutionManager.get_instance()
    faces = sr_mgr.detect_faces(rgb, pad_pct=0.25)
    print(f"Detected {len(faces)} face(s) in {w}x{h} image.\n")

    scale_factor = 4
    target_w, target_h = w * scale_factor, h * scale_factor
    ai_4x_dummy = np.random.randint(0, 255, (target_h, target_w, 3), dtype=np.uint8)

    t0 = time.perf_counter()

    # Sub-step timing counters
    t_bbox_ms = 0.0
    t_conv_ms = 0.0
    t_mask_ms = 0.0
    t_feather_ms = 0.0
    t_crop_ms = 0.0
    t_blend_ms = 0.0
    t_place_ms = 0.0

    result_rgb = ai_4x_dummy.copy()

    for face in faces:
        t_b0 = time.perf_counter()
        x1, y1, x2, y2 = face["bbox"]
        x1_sf, y1_sf = x1 * scale_factor, y1 * scale_factor
        x2_sf, y2_sf = x2 * scale_factor, y2 * scale_factor
        face_h_sf = y2_sf - y1_sf
        face_w_sf = x2_sf - x1_sf
        t_b1 = time.perf_counter()
        t_bbox_ms += (t_b1 - t_b0) * 1000.0

        if face_h_sf <= 0 or face_w_sf <= 0:
            continue

        # Conventional ROI upscale
        t_cv0 = time.perf_counter()
        native_face_crop = rgb[y1:y2, x1:x2, :]
        conv_crop = cv2.resize(native_face_crop, (face_w_sf, face_h_sf), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
        t_cv1 = time.perf_counter()
        t_conv_ms += (t_cv1 - t_cv0) * 1000.0

        # AI crop extraction
        t_cr0 = time.perf_counter()
        ai_crop = ai_4x_dummy[y1_sf:y2_sf, x1_sf:x2_sf, :].astype(np.float32)
        t_cr1 = time.perf_counter()
        t_crop_ms += (t_cr1 - t_cr0) * 1000.0

        # Mask creation
        t_m0 = time.perf_counter()
        mask = np.zeros((face_h_sf, face_w_sf), dtype=np.float32)
        center = (face_w_sf // 2, face_h_sf // 2)
        axes = (face_w_sf // 2, face_h_sf // 2)
        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
        t_m1 = time.perf_counter()
        t_mask_ms += (t_m1 - t_m0) * 1000.0

        # Mask feathering (GaussianBlur)
        t_f0 = time.perf_counter()
        ksize = max(31, min(face_w_sf, face_h_sf) // 4)
        if ksize % 2 == 0:
            ksize += 1
        soft_mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)
        soft_mask_3d = (soft_mask * 0.70)[:, :, np.newaxis]
        t_f1 = time.perf_counter()
        t_feather_ms += (t_f1 - t_f0) * 1000.0

        # Float blending
        t_bl0 = time.perf_counter()
        blended = ai_crop * soft_mask_3d + conv_crop * (1.0 - soft_mask_3d)
        t_bl1 = time.perf_counter()
        t_blend_ms += (t_bl1 - t_bl0) * 1000.0

        # Clip and placement
        t_pl0 = time.perf_counter()
        result_rgb[y1_sf:y2_sf, x1_sf:x2_sf, :] = np.clip(blended, 0, 255).astype(np.uint8)
        t_pl1 = time.perf_counter()
        t_place_ms += (t_pl1 - t_pl0) * 1000.0

    t_total_ms = (time.perf_counter() - t0) * 1000.0

    print("--- FACE PRESERVATION SUB-STEP TIMINGS ---")
    print(f"[FACE_PERF] bbox preparation:     {t_bbox_ms:6.2f} ms")
    print(f"[FACE_PERF] face crop:             {t_crop_ms:6.2f} ms")
    print(f"[FACE_PERF] conventional upscale: {t_conv_ms:6.2f} ms")
    print(f"[FACE_PERF] mask creation:        {t_mask_ms:6.2f} ms")
    print(f"[FACE_PERF] mask feather:         {t_feather_ms:6.2f} ms")
    print(f"[FACE_PERF] blend:                {t_blend_ms:6.2f} ms")
    print(f"[FACE_PERF] placement:            {t_place_ms:6.2f} ms")
    print(f"[FACE_PERF] TOTAL:                {t_total_ms:6.2f} ms")
    print("=" * 80)

if __name__ == "__main__":
    main()
