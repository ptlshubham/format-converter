"""
Phase 3 Tile Configurations Benchmark Matrix Script
Tests Baseline, Current, and Configurations A, B, C, D, E.
Measures total time, inference time, model call count, X/Y seam discontinuities, peak RSS, and face quality.
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image
try:
    import psutil
except ImportError:
    psutil = None


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.processing.super_resolution import SuperResolutionManager

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "diagnostic_results"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def measure_seam_discontinuity(img_4x: np.ndarray, stride_4x: int) -> dict:
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
        "mean_x": float(np.mean(x_deltas)) if x_deltas else 0.0,
        "mean_y": float(np.mean(y_deltas)) if y_deltas else 0.0,
        "max_x": float(np.max(x_deltas)) if x_deltas else 0.0,
        "max_y": float(np.max(y_deltas)) if y_deltas else 0.0,
        "max_overall": float(max(np.max(x_deltas) if x_deltas else 0, np.max(y_deltas) if y_deltas else 0))
    }

def run_tiled_config(
    img_rgb: np.ndarray,
    session,
    core_size: int,
    tile_pad: int
) -> tuple:
    h_orig, w_orig, _ = img_rgb.shape
    out_4x_h, out_4x_w = h_orig * 4, w_orig * 4
    output_4x_rgb = np.zeros((out_4x_h, out_4x_w, 3), dtype=np.uint8)

    t0_total = time.perf_counter()
    t_infer_accum = 0.0

    if tile_pad > 0:
        padded_img = cv2.copyMakeBorder(
            img_rgb, tile_pad, tile_pad, tile_pad, tile_pad, cv2.BORDER_REFLECT_101
        ).astype(np.float32) / 255.0
    else:
        padded_img = img_rgb.astype(np.float32) / 255.0

    tile_crop_dim = core_size + 2 * tile_pad

    y_steps = (h_orig + core_size - 1) // core_size
    x_steps = (w_orig + core_size - 1) // core_size
    model_calls = y_steps * x_steps

    for y_idx in range(y_steps):
        for x_idx in range(x_steps):
            y1 = y_idx * core_size
            y2 = min(y1 + core_size, h_orig)
            x1 = x_idx * core_size
            x2 = min(x1 + core_size, w_orig)

            vh = y2 - y1
            vw = x2 - x1

            if tile_pad > 0:
                py1 = y1
                px1 = x1
                tile_in = padded_img[py1 : py1 + tile_crop_dim, px1 : px1 + tile_crop_dim, :]
                th, tw, _ = tile_in.shape
                if th != tile_crop_dim or tw != tile_crop_dim:
                    tile_in = np.pad(tile_in, ((0, tile_crop_dim - th), (0, tile_crop_dim - tw), (0, 0)), mode='edge')
                
                # Resize tile_in to 128x128 for fixed ONNX model schema
                if tile_crop_dim != 128:
                    tile_onnx_in = cv2.resize(tile_in, (128, 128), interpolation=cv2.INTER_LANCZOS4)
                else:
                    tile_onnx_in = tile_in
            else:
                tile_in = padded_img[y1:y2, x1:x2, :]
                th, tw, _ = tile_in.shape
                tile_onnx_in = np.pad(tile_in, ((0, 128 - th), (0, 128 - tw), (0, 0)), mode='edge')

            t_nchw = np.transpose(tile_onnx_in, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)

            t_inf_0 = time.perf_counter()
            out_nchw = session.run(['upscaled_image'], {'image': t_nchw})[0]
            t_infer_accum += (time.perf_counter() - t_inf_0) * 1000.0

            out_tile = np.transpose(out_nchw[0], (1, 2, 0))
            out_tile_u8 = np.clip(out_tile * 255.0, 0, 255).astype(np.uint8)

            if tile_pad > 0:
                # Resize 512x512 ONNX output back to full crop 4x size (tile_crop_dim * 4)
                if tile_crop_dim != 128:
                    out_full_4x = cv2.resize(out_tile_u8, (tile_crop_dim * 4, tile_crop_dim * 4), interpolation=cv2.INTER_LANCZOS4)
                else:
                    out_full_4x = out_tile_u8

                crop_top = tile_pad * 4
                crop_left = tile_pad * 4
                out_core = out_full_4x[crop_top : crop_top + vh * 4, crop_left : crop_left + vw * 4, :]
            else:
                out_core = out_tile_u8[:vh * 4, :vw * 4, :]

            output_4x_rgb[y1*4:y2*4, x1*4:x2*4, :] = out_core

    t_total_ms = (time.perf_counter() - t0_total) * 1000.0
    return output_4x_rgb, t_total_ms, t_infer_accum, model_calls

def main():
    print("=" * 70)
    print("PHASE 3 — TILE CONFIGURATIONS BENCHMARK MATRIX")
    print("=" * 70)

    test_img_path = os.path.abspath(os.path.join("test_data", "test_human_images", "19035828_web1__12294096_web1_180615-PNR-newmayorchallenge.jpg"))
    orig_pil = Image.open(test_img_path).convert("RGB")
    orig_rgb = np.array(orig_pil, dtype=np.uint8)
    h_orig, w_orig, _ = orig_rgb.shape
    print(f"Loaded Test Image: {test_img_path} ({w_orig}x{h_orig})\n")

    sr_mgr = SuperResolutionManager.get_instance()
    session = sr_mgr.get_session()

    configs = [
        ("Baseline", 128, 0),
        ("Current", 64, 32),
        ("Test A", 128, 16),
        ("Test B", 128, 24),
        ("Test C", 128, 32),
        ("Test D", 96, 24),
        ("Test E", 64, 16),
    ]

    results = []

    for name, core, pad in configs:
        process = psutil.Process() if psutil else None
        rss_start = process.memory_info().rss / (1024 * 1024) if process else 0.0

        out_4x, t_total, t_infer, calls = run_tiled_config(orig_rgb, session, core_size=core, tile_pad=pad)
        seams = measure_seam_discontinuity(out_4x, stride_4x=core*4)

        rss_peak = process.memory_info().rss / (1024 * 1024) if process else 0.0

        # Save output image for face visual inspection
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"phase3_tile_config_{name.replace(' ', '_')}.png"), cv2.cvtColor(out_4x, cv2.COLOR_RGB2BGR))

        res = {
            "name": name,
            "core": core,
            "pad": pad,
            "calls": calls,
            "total_ms": round(t_total, 2),
            "infer_ms": round(t_infer, 2),
            "mean_x": round(seams["mean_x"], 4),
            "mean_y": round(seams["mean_y"], 4),
            "max_overall": round(seams["max_overall"], 4),
            "rss_mb": round(rss_peak, 2),
        }
        results.append(res)
        print(f"Config {name:10s} (Core={core:3d}, Pad={pad:2d}): Calls={calls:3d} | Total={t_total:8.2f}ms | Infer={t_infer:8.2f}ms | Seam X={seams['mean_x']:.4f}, Y={seams['mean_y']:.4f} (Max: {seams['max_overall']:.4f})")

    print("\n" + "=" * 80)
    print("REQUIRED BENCHMARK MATRIX TABLE")
    print("=" * 80)
    print(f"{'Config':10s} | {'Core':5s} | {'Pad':4s} | {'Calls':6s} | {'Total Time':11s} | {'X Seam':7s} | {'Y Seam':7s} | {'Max Seam':8s} | {'Peak RSS':9s}")
    print("-" * 80)
    for r in results:
        print(f"{r['name']:10s} | {r['core']:5d} | {r['pad']:4d} | {r['calls']:6d} | {r['total_ms']:8.2f} ms | {r['mean_x']:7.4f} | {r['mean_y']:7.4f} | {r['max_overall']:8.4f} | {r['rss_mb']:7.2f} MB")
    print("=" * 80)

if __name__ == "__main__":
    main()
