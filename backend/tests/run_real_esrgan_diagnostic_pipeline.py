"""
Real-ESRGAN AI Export Pipeline Comprehensive Diagnostic Script
Executes Stages A-J, Checks 1-7, extracts facial crops, logs quantitative metrics,
and outputs exact root cause evidence.
"""

import os
import sys
import io
import time
import numpy as np
import cv2
from PIL import Image
import onnxruntime as ort

# Ensure backend modules can be imported
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from backend.segmentation.model_manager import ModelManager
from backend.background_removal_engine import BackgroundRemovalEngine
from backend.processing.super_resolution import SuperResolutionManager, TILE_SIZE
from backend.image_processing import export_rgba_bytes, composite_background

OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "diagnostic_results"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

def print_log(*args, **kwargs):
    kwargs['flush'] = True
    print(*args, **kwargs)

def crop_face(img: np.ndarray, bbox: tuple, scale: float = 1.0) -> np.ndarray:
    """Crops a face region given a 1x bounding box (x1, y1, x2, y2) scaled by `scale`."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    sx1 = int(round(x1 * scale))
    sy1 = int(round(y1 * scale))
    sx2 = int(round(x2 * scale))
    sy2 = int(round(y2 * scale))
    
    sx1 = max(0, min(sx1, w - 1))
    sy1 = max(0, min(sy1, h - 1))
    sx2 = max(sx1 + 1, min(sx2, w))
    sy2 = max(sy1 + 1, min(sy2, h))
    
    return img[sy1:sy2, sx1:sx2].copy()

def main():
    print_log("=" * 70)
    print_log("REAL-ESRGAN AI EXPORT PIPELINE DIAGNOSTIC INVESTIGATION")
    print_log("=" * 70)

    # 1. Load Test Image
    # Select test image with face
    test_img_path = os.path.abspath(os.path.join("test_data", "test_human_images", "coach-yelling-at-athlete-716268.jpg"))
    if not os.path.exists(test_img_path):
        # Fallback search
        test_dir = os.path.abspath(os.path.join("test_data", "test_human_images"))
        files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if not files:
            raise FileNotFoundError("No test image found in test_data/test_human_images")
        test_img_path = os.path.join(test_dir, files[0])

    print(f"\n[INFO] Loaded Test Image: {test_img_path}")
    with open(test_img_path, "rb") as f:
        img_bytes = f.read()

    orig_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    orig_np = np.array(orig_pil, dtype=np.uint8)
    h_orig, w_orig, _ = orig_np.shape
    print(f"[INFO] Image Resolution: {w_orig}x{h_orig}")

    # Select face region bounding box for low resolution test photo
    if "coach-yelling" in test_img_path.lower():
        face_bbox = (20, 10, 180, 180)
    elif "messi" in test_img_path.lower():
        face_bbox = (220, 20, 420, 240)
    else:
        # Default face bounding box (upper central area)
        face_bbox = (int(w_orig * 0.25), int(h_orig * 0.05), int(w_orig * 0.75), int(h_orig * 0.65))

    print(f"[INFO] Face Bounding Box (1x): {face_bbox} (Width: {face_bbox[2]-face_bbox[0]}px, Height: {face_bbox[3]-face_bbox[1]}px)")

    # Save 1x Original Face Crop
    orig_face_crop = crop_face(orig_np, face_bbox, 1.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_0_original_photo_face.png"), cv2.cvtColor(orig_face_crop, cv2.COLOR_RGB2BGR))

    # ==================================================
    # STEP 2: STAGE A NATIVE RGBA CUTOUT
    # ==================================================
    stage_a_rgba = np.dstack((orig_np, np.full((h_orig, w_orig), 255, dtype=np.uint8)))
    print_log(f"[STAGE A] Native RGBA Cutout Shape: {stage_a_rgba.shape}")

    # Save Stage A crop
    stage_a_crop = crop_face(stage_a_rgba, face_bbox, 1.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_A_native_rgba_face.png"), cv2.cvtColor(stage_a_crop, cv2.COLOR_RGBA2BGRA))

    # ==================================================
    # STEP 3: INTERCEPT STAGES B THROUGH I
    # ==================================================
    sr_mgr = SuperResolutionManager.get_instance()
    session = sr_mgr.get_session()

    # STAGE B: RGB extracted from RGBA
    stage_b_rgb = stage_a_rgba[:, :, :3].copy()
    stage_a_alpha = stage_a_rgba[:, :, 3].copy()
    stage_b_crop = crop_face(stage_b_rgb, face_bbox, 1.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_B_extracted_rgb_face.png"), cv2.cvtColor(stage_b_crop, cv2.COLOR_RGB2BGR))

    # STAGE C: RGB after transparent-edge color protection / inpainting
    stage_c_rgb = sr_mgr.dilate_rgb_edges(stage_b_rgb, stage_a_alpha)
    stage_c_crop = crop_face(stage_c_rgb, face_bbox, 1.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_C_inpainted_rgb_face.png"), cv2.cvtColor(stage_c_crop, cv2.COLOR_RGB2BGR))

    # STAGE D & E: Individual 128x128 input tile before Real-ESRGAN and 512x512 output tile
    fx1, fy1, fx2, fy2 = face_bbox
    tile_x_idx = fx1 // TILE_SIZE
    tile_y_idx = fy1 // TILE_SIZE
    
    td_y1 = tile_y_idx * TILE_SIZE
    td_y2 = min(td_y1 + TILE_SIZE, h_orig)
    td_x1 = tile_x_idx * TILE_SIZE
    td_x2 = min(td_x1 + TILE_SIZE, w_orig)
    
    img_float_c = stage_c_rgb.astype(np.float32) / 255.0
    tile_raw = img_float_c[td_y1:td_y2, td_x1:td_x2, :]
    th, tw, _ = tile_raw.shape
    pad_bottom = TILE_SIZE - th
    pad_right = TILE_SIZE - tw
    if pad_bottom > 0 or pad_right > 0:
        tile_padded = np.pad(tile_raw, ((0, pad_bottom), (0, pad_right), (0, 0)), mode='edge')
    else:
        tile_padded = tile_raw

    # STAGE D Tile (128x128 uint8)
    stage_d_tile_uint8 = np.clip(tile_padded * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_D_input_tile_128.png"), cv2.cvtColor(stage_d_tile_uint8, cv2.COLOR_RGB2BGR))

    # Run STAGE E Inference on D
    tile_nchw = np.transpose(tile_padded, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
    out_nchw_e = session.run(['upscaled_image'], {'image': tile_nchw})[0]
    out_tile_e = np.transpose(out_nchw_e[0], (1, 2, 0))
    stage_e_tile_uint8 = np.clip(out_tile_e * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_E_output_tile_512.png"), cv2.cvtColor(stage_e_tile_uint8, cv2.COLOR_RGB2BGR))

    # STAGE F & G: Fully reconstructed 4x RGB image before downsampling & tile stitching
    out_4x_h, out_4x_w = h_orig * 4, w_orig * 4
    output_4x_rgb = np.zeros((out_4x_h, out_4x_w, 3), dtype=np.uint8)

    y_tiles = (h_orig + TILE_SIZE - 1) // TILE_SIZE
    x_tiles = (w_orig + TILE_SIZE - 1) // TILE_SIZE

    for y_idx in range(y_tiles):
        for x_idx in range(x_tiles):
            y1 = y_idx * TILE_SIZE
            y2 = min(y1 + TILE_SIZE, h_orig)
            x1 = x_idx * TILE_SIZE
            x2 = min(x1 + TILE_SIZE, w_orig)

            tile = img_float_c[y1:y2, x1:x2, :]
            th, tw, _ = tile.shape

            pb = TILE_SIZE - th
            pr = TILE_SIZE - tw
            if pb > 0 or pr > 0:
                tp = np.pad(tile, ((0, pb), (0, pr), (0, 0)), mode='edge')
            else:
                tp = tile

            t_nchw = np.transpose(tp, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
            o_nchw = session.run(['upscaled_image'], {'image': t_nchw})[0]
            o_tile = np.transpose(o_nchw[0], (1, 2, 0))

            valid_h = th * 4
            valid_w = tw * 4
            o_crop = o_tile[:valid_h, :valid_w, :]
            o_crop_u8 = np.clip(o_crop * 255.0, 0, 255).astype(np.uint8)

            output_4x_rgb[y1*4:y2*4, x1*4:x2*4, :] = o_crop_u8

    stage_f_rgb = output_4x_rgb.copy()
    stage_g_rgb = output_4x_rgb.copy()

    stage_f_crop = crop_face(stage_f_rgb, face_bbox, 4.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_F_stitched_4x_rgb_face.png"), cv2.cvtColor(stage_f_crop, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_G_stitched_blended_4x_rgb_face.png"), cv2.cvtColor(stage_f_crop, cv2.COLOR_RGB2BGR))

    # STAGE H: 4x -> 2x downsampled RGB (Lanczos 0.5x)
    target_2x_w, target_2x_h = w_orig * 2, h_orig * 2
    stage_h_rgb = cv2.resize(stage_f_rgb, (target_2x_w, target_2x_h), interpolation=cv2.INTER_LANCZOS4)
    stage_h_crop = crop_face(stage_h_rgb, face_bbox, 2.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_H_downsampled_2x_rgb_face.png"), cv2.cvtColor(stage_h_crop, cv2.COLOR_RGB2BGR))

    # STAGE I: Final RGBA after alpha recombination (HD 2x and 4x)
    alpha_2x = cv2.resize(stage_a_alpha, (target_2x_w, target_2x_h), interpolation=cv2.INTER_LANCZOS4)
    alpha_2x = np.clip(alpha_2x, 0, 255).astype(np.uint8)
    stage_i_rgba_2x = np.dstack((stage_h_rgb, alpha_2x))

    alpha_4x = cv2.resize(stage_a_alpha, (w_orig * 4, h_orig * 4), interpolation=cv2.INTER_LANCZOS4)
    alpha_4x = np.clip(alpha_4x, 0, 255).astype(np.uint8)
    stage_i_rgba_4x = np.dstack((stage_f_rgb, alpha_4x))

    stage_i_crop_2x = crop_face(stage_i_rgba_2x, face_bbox, 2.0)
    stage_i_crop_4x = crop_face(stage_i_rgba_4x, face_bbox, 4.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_I_final_rgba_2x_face.png"), cv2.cvtColor(stage_i_crop_2x, cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_I_final_rgba_4x_face.png"), cv2.cvtColor(stage_i_crop_4x, cv2.COLOR_RGBA2BGRA))

    # STAGE J: Final PNG/WebP/JPG downloaded image
    png_bytes = export_rgba_bytes(stage_i_rgba_4x, "PNG")
    webp_bytes = export_rgba_bytes(stage_i_rgba_4x, "WEBP", quality=95)
    jpg_bytes = export_rgba_bytes(stage_i_rgba_4x, "JPG", quality=95)

    png_decoded = np.array(Image.open(io.BytesIO(png_bytes)).convert("RGBA"))
    webp_decoded = np.array(Image.open(io.BytesIO(webp_bytes)).convert("RGBA"))
    jpg_decoded = np.array(Image.open(io.BytesIO(jpg_bytes)).convert("RGB"))

    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_J_downloaded_png_face.png"), cv2.cvtColor(crop_face(png_decoded, face_bbox, 4.0), cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_J_downloaded_webp_face.png"), cv2.cvtColor(crop_face(webp_decoded, face_bbox, 4.0), cv2.COLOR_RGBA2BGRA))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "stage_J_downloaded_jpg_face.png"), cv2.cvtColor(crop_face(jpg_decoded, face_bbox, 4.0), cv2.COLOR_RGB2BGR))

    print_log("\n[SUCCESS] Intercepted and saved Stages A-J crops to diagnostic_results/\n")

    # ==================================================
    # EXECUTE CHECKS 1 THROUGH 7
    # ==================================================

    print_log("=" * 70)
    print_log("CHECK 1 — MODEL INPUT VERIFICATION")
    print_log("=" * 70)
    
    # 1.1 Input shape & schema
    model_inputs = [(i.name, i.shape, i.type) for i in session.get_inputs()]
    print_log(f"ONNX Model Input Schema: {model_inputs}")
    
    # 1.2 Channel Ordering (RGB vs BGR test)
    out_rgb_input = session.run(['upscaled_image'], {'image': tile_nchw})[0]
    
    tile_bgr_padded = tile_padded[:, :, ::-1] # Swap RGB -> BGR
    tile_bgr_nchw = np.transpose(tile_bgr_padded, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
    out_bgr_input = session.run(['upscaled_image'], {'image': tile_bgr_nchw})[0]

    out_rgb_u8 = np.clip(np.transpose(out_rgb_input[0], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    out_bgr_u8 = np.clip(np.transpose(out_bgr_input[0], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)

    cv2.imwrite(os.path.join(OUTPUT_DIR, "check1_inference_from_RGB_input.png"), cv2.cvtColor(out_rgb_u8, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "check1_inference_from_BGR_input.png"), cv2.cvtColor(out_bgr_u8, cv2.COLOR_BGR2RGB))

    diff_rgb_bgr = np.mean(np.abs(out_rgb_u8.astype(float) - out_bgr_u8[:, :, ::-1].astype(float)))
    print_log(f"Mean Pixel Difference between RGB input and BGR input inference: {diff_rgb_bgr:.2f}")
    
    # Check 1.3: Normalization range [0, 1] vs [-1, 1]
    print_log(f"Runtime Input Tile Min: {tile_nchw.min():.4f}, Max: {tile_nchw.max():.4f}, Mean: {tile_nchw.mean():.4f}")
    print_log(f"Runtime Output Tile Min: {out_nchw_e.min():.4f}, Max: {out_nchw_e.max():.4f}, Mean: {out_nchw_e.mean():.4f}")
    
    print_log("\n" + "=" * 70)
    print_log("CHECK 2 — TILE SIZE & CONTEXTUAL PROCESSING EXPERIMENT")
    print_log("=" * 70)
    
    # Method B: Overlapped Tiling Implementation
    tile_pad = 16
    overlap_4x_out = np.zeros((out_4x_h, out_4x_w, 3), dtype=np.uint8)
    
    padded_img_c = np.pad(stage_c_rgb, ((tile_pad, tile_pad), (tile_pad, tile_pad), (0, 0)), mode='edge').astype(np.float32) / 255.0

    for y_idx in range(y_tiles):
        for x_idx in range(x_tiles):
            y1 = y_idx * TILE_SIZE
            y2 = min(y1 + TILE_SIZE, h_orig)
            x1 = x_idx * TILE_SIZE
            x2 = min(x1 + TILE_SIZE, w_orig)

            # Expand tile boundaries by tile_pad
            py1 = y1
            py2 = y1 + TILE_SIZE + 2 * tile_pad
            px1 = x1
            px2 = x1 + TILE_SIZE + 2 * tile_pad

            sub_img = padded_img_c[py1:py2, px1:px2, :]
            sh, sw, _ = sub_img.shape

            # Resize sub_img to exact 128x128 for model input
            sub_128 = cv2.resize(sub_img, (TILE_SIZE, TILE_SIZE), interpolation=cv2.INTER_LANCZOS4)

            t_nchw = np.transpose(sub_128, (2, 0, 1))[np.newaxis, :, :, :].astype(np.float32)
            o_nchw = session.run(['upscaled_image'], {'image': t_nchw})[0]
            o_tile = np.transpose(o_nchw[0], (1, 2, 0))
            o_tile_u8 = np.clip(o_tile * 255.0, 0, 255).astype(np.uint8)

            # Resize output 512x512 back to full context size (sh*4, sw*4)
            o_full = cv2.resize(o_tile_u8, (sw * 4, sh * 4), interpolation=cv2.INTER_LANCZOS4)

            out_valid_h = (y2 - y1) * 4
            out_valid_w = (x2 - x1) * 4
            
            b_top = tile_pad * 4
            b_left = tile_pad * 4
            
            o_valid = o_full[b_top : b_top + out_valid_h, b_left : b_left + out_valid_w, :]
            overlap_4x_out[y1*4:y2*4, x1*4:x2*4, :] = o_valid

    overlap_face_crop = crop_face(overlap_4x_out, face_bbox, 4.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "check2_overlapped_tiling_face_4x.png"), cv2.cvtColor(overlap_face_crop, cv2.COLOR_RGB2BGR))

    tile_seam_diff = np.mean(np.abs(stage_f_crop.astype(float) - overlap_face_crop.astype(float)))
    print(f"Non-overlapped 128x128 vs Overlapped 128x128 Face Pixel Diff: {tile_seam_diff:.2f}")

    print("\n" + "=" * 70)
    print("CHECK 3 — TILE STITCHING & ALIGNMENT ANALYSIS")
    print("=" * 70)
    
    grid_lines_x = [x * 512 for x in range(1, x_tiles)]
    grid_lines_y = [y * 512 for y in range(1, y_tiles)]
    
    seam_discontinuities = []
    for gx in grid_lines_x:
        if gx < out_4x_w - 1:
            diff = np.mean(np.abs(stage_f_rgb[:, gx - 1, :].astype(float) - stage_f_rgb[:, gx, :].astype(float)))
            seam_discontinuities.append((f"X={gx}", round(diff, 2)))
    for gy in grid_lines_y:
        if gy < out_4x_h - 1:
            diff = np.mean(np.abs(stage_f_rgb[gy - 1, :, :].astype(float) - stage_f_rgb[gy, :, :].astype(float)))
            seam_discontinuities.append((f"Y={gy}", round(diff, 2)))
            
    print(f"Tile Seam Discontinuity Measurements (Mean RGB Delta at Boundary): {seam_discontinuities}")

    print("\n" + "=" * 70)
    print("CHECK 4 — RGB INPAINTING IMPACT ON OPAQUE FACIAL PIXELS")
    print("=" * 70)
    
    opaque_mask = (stage_a_alpha == 255)
    rgb_diff = np.abs(stage_b_rgb.astype(int) - stage_c_rgb.astype(int))
    opaque_diff = rgb_diff[opaque_mask]
    
    modified_opaque_px = np.count_nonzero(np.max(opaque_diff, axis=1) > 0)
    max_opaque_delta = np.max(opaque_diff) if opaque_diff.size > 0 else 0
    mean_opaque_delta = np.mean(opaque_diff) if opaque_diff.size > 0 else 0.0

    print(f"Total Opaque Pixels (alpha == 255): {np.count_nonzero(opaque_mask)}")
    print(f"Opaque Pixels Modified by cv2.inpaint: {modified_opaque_px}")
    print(f"Max Delta on Opaque Pixels: {max_opaque_delta}")
    print(f"Mean Delta on Opaque Pixels: {mean_opaque_delta:.4f}")
    
    modified_mask = (np.max(rgb_diff, axis=2) > 0)
    if np.any(modified_mask):
        ys, xs = np.where(modified_mask)
        inpaint_bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
        print(f"Inpainting Modification Bounding Box: {inpaint_bbox}")
    else:
        print("Inpainting Modification Bounding Box: None (No pixels modified)")

    print("\n" + "=" * 70)
    print("CHECK 5 — 4x -> 2x PIPELINE DEGRADATION ANALYSIS")
    print("=" * 70)
    
    conv_2x_rgb = cv2.resize(stage_b_rgb, (target_2x_w, target_2x_h), interpolation=cv2.INTER_LANCZOS4)
    conv_2x_crop = crop_face(conv_2x_rgb, face_bbox, 2.0)
    cv2.imwrite(os.path.join(OUTPUT_DIR, "check5_conventional_lanczos_2x_face.png"), cv2.cvtColor(conv_2x_crop, cv2.COLOR_RGB2BGR))

    ai_vs_conv_diff = np.mean(np.abs(stage_h_crop.astype(float) - conv_2x_crop.astype(float)))
    print(f"AI 2x (Real-ESRGAN 4x -> Lanczos 0.5x) vs Conventional Lanczos 2x Face Diff: {ai_vs_conv_diff:.2f}")

    print("\n" + "=" * 70)
    print("CHECK 6 — MODEL QUALITY (DIRECT UNTILED & UNPROCESSED INFERENCE)")
    print("=" * 70)
    
    raw_face_128 = cv2.resize(orig_face_crop, (128, 128), interpolation=cv2.INTER_LANCZOS4)
    raw_face_nchw = np.transpose(raw_face_128.astype(np.float32) / 255.0, (2, 0, 1))[np.newaxis, :, :, :]
    
    out_raw_face_nchw = session.run(['upscaled_image'], {'image': raw_face_nchw})[0]
    out_raw_face_512 = np.clip(np.transpose(out_raw_face_nchw[0], (1, 2, 0)) * 255.0, 0, 255).astype(np.uint8)
    
    cv2.imwrite(os.path.join(OUTPUT_DIR, "check6_raw_face_input_128.png"), cv2.cvtColor(raw_face_128, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "check6_raw_face_realesrgan_output_512.png"), cv2.cvtColor(out_raw_face_512, cv2.COLOR_RGB2BGR))
    print(f"[CHECK 6] Executed Direct Untiled Real-ESRGAN on Raw Face Crop.")

    print("\n" + "=" * 70)
    print("CHECK 7 — DOWNLOAD FORMAT ENCODING DIFFERENCES")
    print("=" * 70)
    
    png_crop = crop_face(png_decoded, face_bbox, 4.0)
    webp_crop = crop_face(webp_decoded, face_bbox, 4.0)
    jpg_crop = crop_face(jpg_decoded, face_bbox, 4.0)
    
    png_diff = np.mean(np.abs(stage_i_crop_4x.astype(float) - png_crop.astype(float)))
    webp_diff = np.mean(np.abs(stage_i_crop_4x.astype(float) - webp_crop.astype(float)))
    jpg_diff = np.mean(np.abs(stage_i_crop_4x[:, :, :3].astype(float) - jpg_crop.astype(float)))
    
    print(f"Pre-encoding RGBA vs Decoded PNG Face Diff: {png_diff:.4f}")
    print(f"Pre-encoding RGBA vs Decoded WebP Face Diff: {webp_diff:.4f}")
    print(f"Pre-encoding RGB vs Decoded JPG Face Diff: {jpg_diff:.4f}")

    print("\n" + "=" * 70)
    print("DIAGNOSTIC INVESTIGATION COMPLETE")
    print(f"All diagnostic crops saved in: {OUTPUT_DIR}")
    print("=" * 70)

if __name__ == "__main__":
    main()
