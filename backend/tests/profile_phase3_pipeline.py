"""
Phase 3 Pipeline Profiling and Verification Script
Measures and prints detailed timing breakdowns for all major operations:
1. image loading & RGBA preprocessing
2. RGB edge dilation/inpainting
3. YuNet face detection
4. tile preparation & contextual padding
5. Real-ESRGAN DirectML inference
6. tile output cropping & reconstruction
7. HD 2x downsampling (Option B)
8. face preservation blending (0.70 weight)
9. alpha recombination
10. memory usage & total pipeline time
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.processing.super_resolution import SuperResolutionManager

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "diagnostic_results"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def measure_seams(img_4x: np.ndarray, core_size: int = 128) -> dict:
    h, w, _ = img_4x.shape
    stride_4x = core_size * 4
    x_lines = [x * stride_4x for x in range(1, w // stride_4x)]
    y_lines = [y * stride_4x for y in range(1, h // stride_4x)]

    x_deltas = []
    for gx in x_lines:
        if gx < w - 1:
            delta = np.mean(np.abs(img_4x[:, gx - 1, :].astype(float) - img_4x[:, gx, :].astype(float)))
            x_deltas.append(delta)

    y_deltas = []
    for gy in y_lines:
        if gy < h - 1:
            delta = np.mean(np.abs(img_4x[gy - 1, :, :].astype(float) - img_4x[gy, :, :].astype(float)))
            y_deltas.append(delta)

    return {
        "mean_x": float(np.mean(x_deltas)) if x_deltas else 0.0,
        "mean_y": float(np.mean(y_deltas)) if y_deltas else 0.0,
        "max_overall": float(max(np.max(x_deltas) if x_deltas else 0, np.max(y_deltas) if y_deltas else 0))
    }

def main():
    print("=" * 80)
    print("PHASE 3 — PIPELINE PROFILING AND QUALITY VERIFICATION")
    print("=" * 80)

    test_img_path = os.path.abspath(os.path.join("test_data", "test_human_images", "19035828_web1__12294096_web1_180615-PNR-newmayorchallenge.jpg"))
    pil_img = Image.open(test_img_path).convert("RGBA")
    rgba_arr = np.array(pil_img, dtype=np.uint8)
    h, w, _ = rgba_arr.shape
    print(f"Loaded Test Image: {test_img_path} ({w}x{h} RGBA)\n")

    sr_mgr = SuperResolutionManager.get_instance()

    # Benchmark 4x (Ultra HD) Export
    print("--- Running Ultra HD 4x Export Profile ---")
    res_4x, metrics_4x = sr_mgr.upscale_rgba_ai(rgba_arr, export_quality="4x")
    seams_4x = measure_seams(res_4x[:, :, :3], core_size=128)

    print("\n--- Running HD 2x Export Profile ---")
    res_2x, metrics_2x = sr_mgr.upscale_rgba_ai(rgba_arr, export_quality="2x")

    print("\n" + "=" * 80)
    print("TIMING BREAKDOWN REPORT (4x Ultra HD)")
    print("=" * 80)
    print(f"[PERF] Execution Provider:     {metrics_4x['provider']}")
    print(f"[PERF] Model Load / Session:   {metrics_4x['model_load_ms']} ms")
    print(f"[PERF] Tile Count:             {metrics_4x['tile_count']}")
    print(f"[PERF] preprocessing:          {metrics_4x['preprocessing_ms']} ms")
    print(f"[PERF] tile preparation:       {metrics_4x['tile_prep_ms']} ms")
    print(f"[PERF] Real-ESRGAN inference:  {metrics_4x['ai_inference_ms']} ms")
    print(f"[PERF] tile reconstruction:    {metrics_4x['tile_recon_ms']} ms")
    print(f"[PERF] face detection:         {metrics_4x['face_detection_ms']} ms")
    print(f"[PERF] face preservation:      {metrics_4x['face_blending_ms']} ms")
    print(f"[PERF] downsampling:           {metrics_4x['downsampling_ms']} ms")
    print(f"[PERF] alpha recombination:    {metrics_4x['alpha_recombination_ms']} ms")
    print(f"[PERF] TOTAL:                  {metrics_4x['total_export_ms']} ms")
    print(f"[PERF] Peak RSS:               {metrics_4x['rss_peak_mb']} MB")

    print("\n" + "=" * 80)
    print("TIMING BREAKDOWN REPORT (2x HD)")
    print("=" * 80)
    print(f"[PERF] Execution Provider:     {metrics_2x['provider']}")
    print(f"[PERF] face preservation:      {metrics_2x['face_blending_ms']} ms")
    print(f"[PERF] TOTAL:                  {metrics_2x['total_export_ms']} ms")

    print("\n" + "=" * 80)
    print("QUALITY & REGRESSION METRICS")
    print("=" * 80)
    print(f"4x Output Dimensions:  {res_4x.shape[1]}x{res_4x.shape[0]} (Expected: 4800x3200) -> {'PASS' if res_4x.shape[:2] == (3200, 4800) else 'FAIL'}")
    print(f"2x Output Dimensions:  {res_2x.shape[1]}x{res_2x.shape[0]} (Expected: 2400x1600) -> {'PASS' if res_2x.shape[:2] == (1600, 2400) else 'FAIL'}")
    print(f"X Seam Discontinuity:  {seams_4x['mean_x']:.4f}")
    print(f"Y Seam Discontinuity:  {seams_4x['mean_y']:.4f}")
    print(f"Max Seam Discontinuity:{seams_4x['max_overall']:.4f}")
    print(f"Faces Detected:        {metrics_4x['faces_detected']}")
    print(f"Alpha Preservation:    {'PASS' if res_4x.shape[2] == 4 and res_2x.shape[2] == 4 else 'FAIL'}")
    print("=" * 80)

if __name__ == "__main__":
    main()
