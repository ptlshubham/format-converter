"""
Phase 1 Tile Seams Verification & Benchmarking Script
Tests contextual tiled inference with tile_pad = 32 and core_size = 64
Measures boundary seam step-discontinuities before vs after.
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.processing.super_resolution import SuperResolutionManager

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "diagnostic_results"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_contextual_tiled_sr(
    img_rgb: np.ndarray,
    session: ort.InferenceSession,
    core_size: int = 64,
    tile_pad: int = 32
) -> np.ndarray:
    """
    Executes Real-ESRGAN x4plus inference with contextual tile padding.
    Input tile to ONNX session is always (core_size + 2*tile_pad) x (core_size + 2*tile_pad) = 128x128.
    Crops only the 4x core valid region to eliminate tile boundary seam artifacts.
    """
    h_orig, w_orig, _ = img_rgb.shape
    out_4x_h, out_4x_w = h_orig * 4, w_orig * 4
    output_4x_rgb = np.zeros((out_4x_h, out_4x_w, 3), dtype=np.uint8)

    # Pad image with reflect_101 border for contextual edge inference
    padded_img = cv2.copyMakeBorder(
        img_rgb, tile_pad, tile_pad, tile_pad, tile_pad, cv2.BORDER_REFLECT_101
    ).astype(np.float32) / 255.0

    tile_input_dim = core_size + 2 * tile_pad  # 64 + 64 = 128

    y_steps = (h_orig + core_size - 1) // core_size
    x_steps = (w_orig + core_size - 1) // core_size

    for y_idx in range(y_steps):
        for x_idx in range(x_steps):
            y1 = y_idx * core_size
            y2 = min(y1 + core_size, h_orig)
            x1 = x_idx * core_size
            x2 = min(x1 + core_size, w_orig)

            vh = y2 - y1
            vw = x2 - x1

            # In padded_img, original coordinate (y1, x1) is shifted by +tile_pad
            py1 = y1
            py2 = py1 + tile_input_dim
            px1 = x1
            px2 = px1 + tile_input_dim

            tile_in = padded_img[py1:py2, px1:px2, :]

            # Guarantee tile_in is exactly 128x128
            th, tw, _ = tile_in.shape
            if th != tile_input_dim or tw != tile_input_dim:
                tile_in = np.pad(
                    tile_in,
                    ((0, tile_input_dim - th), (0, tile_input_dim - tw), (0, 0)),
                    mode='edge'
                )

            t_nchw = np.transpose(tile_in, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
            o_nchw = session.run(['upscaled_image'], {'image': t_nchw})[0]
            o_tile = np.transpose(o_nchw[0], (1, 2, 0))
            o_tile_u8 = np.clip(o_tile * 255.0, 0, 255).astype(np.uint8)

            # Crop ONLY the valid 4x core region (ignoring the tile_pad * 4 border)
            crop_top = tile_pad * 4
            crop_left = tile_pad * 4
            crop_h = vh * 4
            crop_w = vw * 4

            o_core = o_tile_u8[crop_top : crop_top + crop_h, crop_left : crop_left + crop_w, :]

            # Place into destination output
            out_y1, out_y2 = y1 * 4, y2 * 4
            out_x1, out_x2 = x1 * 4, x2 * 4
            output_4x_rgb[out_y1:out_y2, out_x1:out_x2, :] = o_core

    return output_4x_rgb

def measure_seam_discontinuity(img_4x: np.ndarray, stride_4x: int) -> dict:
    """Measures mean RGB pixel delta across grid tile boundaries in 4x image."""
    h, w, _ = img_4x.shape
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
        "mean_x_seam_delta": float(np.mean(x_deltas)) if x_deltas else 0.0,
        "mean_y_seam_delta": float(np.mean(y_deltas)) if y_deltas else 0.0,
        "max_x_seam_delta": float(np.max(x_deltas)) if x_deltas else 0.0,
        "max_y_seam_delta": float(np.max(y_deltas)) if y_deltas else 0.0,
    }

def main():
    print("=" * 70)
    print("PHASE 1 — TILE SEAM DISCONTINUITY BENCHMARKING")
    print("=" * 70)

    test_img_path = os.path.abspath(os.path.join("test_data", "test_human_images", "19035828_web1__12294096_web1_180615-PNR-newmayorchallenge.jpg"))
    if not os.path.exists(test_img_path):
        test_img_path = os.path.abspath(os.path.join("Solo_Man_Festive_Background.png"))

    orig_pil = Image.open(test_img_path).convert("RGB")
    orig_rgb = np.array(orig_pil, dtype=np.uint8)
    h_orig, w_orig, _ = orig_rgb.shape
    print(f"Loaded Test Image: {test_img_path} ({w_orig}x{h_orig})")

    sr_mgr = SuperResolutionManager.get_instance()
    session = sr_mgr.get_session()

    # 1. Benchmark BEFORE (Old 128x128 tiles with tile_pad = 0)
    t0 = time.perf_counter()
    out_old_4x = np.zeros((h_orig * 4, w_orig * 4, 3), dtype=np.uint8)
    img_float = orig_rgb.astype(np.float32) / 255.0
    y_tiles = (h_orig + 128 - 1) // 128
    x_tiles = (w_orig + 128 - 1) // 128
    for y in range(y_tiles):
        for x in range(x_tiles):
            y1, y2 = y * 128, min((y + 1) * 128, h_orig)
            x1, x2 = x * 128, min((x + 1) * 128, w_orig)
            tile = img_float[y1:y2, x1:x2, :]
            th, tw, _ = tile.shape
            tp = np.pad(tile, ((0, 128 - th), (0, 128 - tw), (0, 0)), mode='edge')
            t_nchw = np.transpose(tp, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
            o_nchw = session.run(['upscaled_image'], {'image': t_nchw})[0]
            o_tile = np.transpose(o_nchw[0], (1, 2, 0))
            out_old_4x[y1*4:y2*4, x1*4:x2*4, :] = np.clip(o_tile[:th*4, :tw*4, :] * 255.0, 0, 255).astype(np.uint8)
    time_old = (time.perf_counter() - t0) * 1000.0
    seams_old = measure_seam_discontinuity(out_old_4x, stride_4x=512)

    print("\n--- BEFORE (tile_pad = 0, core_size = 128) ---")
    print(f"Execution Time: {time_old:.2f} ms")
    print(f"X Seam Discontinuity: {seams_old['mean_x_seam_delta']:.4f} (Max: {seams_old['max_x_seam_delta']:.4f})")
    print(f"Y Seam Discontinuity: {seams_old['mean_y_seam_delta']:.4f} (Max: {seams_old['max_y_seam_delta']:.4f})")

    # 2. Benchmark AFTER (New 128x128 tiles with core_size = 64, tile_pad = 32)
    t0 = time.perf_counter()
    out_new_4x = run_contextual_tiled_sr(orig_rgb, session, core_size=64, tile_pad=32)
    time_new = (time.perf_counter() - t0) * 1000.0
    seams_new = measure_seam_discontinuity(out_new_4x, stride_4x=256)

    print("\n--- AFTER (tile_pad = 32, core_size = 64) ---")
    print(f"Execution Time: {time_new:.2f} ms")
    print(f"X Seam Discontinuity: {seams_new['mean_x_seam_delta']:.4f} (Max: {seams_new['max_x_seam_delta']:.4f})")
    print(f"Y Seam Discontinuity: {seams_new['mean_y_seam_delta']:.4f} (Max: {seams_new['max_y_seam_delta']:.4f})")

    # Save comparison images
    cv2.imwrite(os.path.join(OUTPUT_DIR, "phase1_old_tiling_seams.png"), cv2.cvtColor(out_old_4x, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "phase1_new_tiling_seams.png"), cv2.cvtColor(out_new_4x, cv2.COLOR_RGB2BGR))
    
    print("\n[SUCCESS] Phase 1 Benchmarking Complete!")

if __name__ == "__main__":
    main()
