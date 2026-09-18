"""
Phase 4 Microbenchmarks — Interpolation Comparison
Benchmarks INTER_LANCZOS4, INTER_CUBIC, INTER_LINEAR, INTER_AREA for pre/post tile resizing.
"""

import os
import sys
import time
import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from backend.processing.super_resolution import SuperResolutionManager

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

def run_interpolation_benchmark(interp_pre, interp_post, name: str):
    test_img_path = os.path.abspath(os.path.join("test_data", "test_human_images", "19035828_web1__12294096_web1_180615-PNR-newmayorchallenge.jpg"))
    pil_img = Image.open(test_img_path).convert("RGBA")
    rgba_arr = np.array(pil_img, dtype=np.uint8)
    orig_h, orig_w, _ = rgba_arr.shape

    sr_mgr = SuperResolutionManager.get_instance()
    session = sr_mgr.get_session()

    rgb = rgba_arr[:, :, :3]
    alpha = rgba_arr[:, :, 3]
    dilated_rgb = sr_mgr.dilate_rgb_edges(rgb, alpha)

    core_size = 128
    tile_pad = 24
    tile_crop_dim = core_size + 2 * tile_pad  # 176

    padded_img = cv2.copyMakeBorder(
        dilated_rgb, tile_pad, tile_pad, tile_pad, tile_pad, cv2.BORDER_REFLECT_101
    ).astype(np.float32) / 255.0

    y_steps = (orig_h + core_size - 1) // core_size
    x_steps = (orig_w + core_size - 1) // core_size

    out_4x = np.zeros((orig_h * 4, orig_w * 4, 3), dtype=np.uint8)

    t0 = time.perf_counter()
    prep_accum = 0.0
    recon_accum = 0.0

    for y_idx in range(y_steps):
        for x_idx in range(x_steps):
            y1 = y_idx * core_size
            y2 = min(y1 + core_size, orig_h)
            x1 = x_idx * core_size
            x2 = min(x1 + core_size, orig_w)

            vh = y2 - y1
            vw = x2 - x1

            py1 = y1
            px1 = x1

            tp0 = time.perf_counter()
            tile_in = padded_img[py1 : py1 + tile_crop_dim, px1 : px1 + tile_crop_dim, :]
            th, tw, _ = tile_in.shape
            if th != tile_crop_dim or tw != tile_crop_dim:
                tile_in = np.pad(tile_in, ((0, tile_crop_dim - th), (0, tile_crop_dim - tw), (0, 0)), mode='edge')

            tile_onnx_in = cv2.resize(tile_in, (128, 128), interpolation=interp_pre)
            tile_nchw = np.transpose(tile_onnx_in, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
            prep_accum += (time.perf_counter() - tp0) * 1000.0

            out_nchw = session.run(['upscaled_image'], {'image': tile_nchw})[0]

            tr0 = time.perf_counter()
            out_tile = np.transpose(out_nchw[0], (1, 2, 0))
            out_tile_u8 = np.clip(out_tile * 255.0, 0, 255).astype(np.uint8)
            out_full_4x = cv2.resize(out_tile_u8, (tile_crop_dim * 4, tile_crop_dim * 4), interpolation=interp_post)
            crop_top = tile_pad * 4
            crop_left = tile_pad * 4
            out_core = out_full_4x[crop_top : crop_top + vh * 4, crop_left : crop_left + vw * 4, :]
            out_4x[y1*4:y2*4, x1*4:x2*4, :] = out_core
            recon_accum += (time.perf_counter() - tr0) * 1000.0

    total_ms = (time.perf_counter() - t0) * 1000.0
    seams = measure_seams(out_4x, core_size=128)

    print(f"[{name:20s}] Total={total_ms:8.2f}ms | Prep={prep_accum:6.2f}ms | Recon={recon_accum:6.2f}ms | X Seam={seams['mean_x']:.4f}, Y Seam={seams['mean_y']:.4f} (Max: {seams['max_overall']:.4f})")

def main():
    print("=" * 80)
    print("INTERPOLATION METHOD COMPARISON FOR PRE/POST TILE RESIZING")
    print("=" * 80)
    run_interpolation_benchmark(cv2.INTER_LANCZOS4, cv2.INTER_LANCZOS4, "LANCZOS4 / LANCZOS4")
    run_interpolation_benchmark(cv2.INTER_CUBIC, cv2.INTER_CUBIC, "CUBIC / CUBIC")
    run_interpolation_benchmark(cv2.INTER_AREA, cv2.INTER_LINEAR, "AREA / LINEAR")
    run_interpolation_benchmark(cv2.INTER_LINEAR, cv2.INTER_LINEAR, "LINEAR / LINEAR")
    print("=" * 80)

if __name__ == "__main__":
    main()
