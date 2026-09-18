"""
Phase 2 Face Preservation & Blending Verification Script
Detects faces using YuNet, constructs soft feathered elliptical face masks,
blends Real-ESRGAN AI with Lanczos conventional upscale across weights 100%, 80%, 70%, 60%, 0%,
and compares 2x Export Option A vs Option B.
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

class FaceDetector:
    def __init__(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.model_path = os.path.join(base_dir, "models", "face_detector", "face_detection_yunet_2023mar.onnx")
        self.detector = None
        if os.path.exists(self.model_path):
            try:
                self.detector = cv2.FaceDetectorYN.create(self.model_path, "", (300, 300), score_threshold=0.4)
            except Exception as e:
                print(f"[WARN] Failed to load YuNet face detector: {e}")

    def detect_faces(self, img_rgb: np.ndarray, pad_pct: float = 0.25) -> list:
        h, w, _ = img_rgb.shape
        faces_out = []

        if self.detector is not None:
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            self.detector.setInputSize((w, h))
            _, faces = self.detector.detect(img_bgr)
            if faces is not None:
                for f in faces:
                    fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                    score = float(f[-1])
                    # Expand bounding box
                    pw = fw * pad_pct
                    ph = fh * pad_pct
                    ex1 = max(0, int(round(fx - pw)))
                    ey1 = max(0, int(round(fy - ph)))
                    ex2 = min(w, int(round(fx + fw + pw)))
                    ey2 = min(h, int(round(fy + fh + ph)))
                    faces_out.append({
                        "bbox": (ex1, ey1, ex2, ey2),
                        "orig_bbox": (int(fx), int(fy), int(fx+fw), int(fy+fh)),
                        "confidence": score
                    })

        # Fallback if YuNet not available or no faces found
        if not faces_out:
            # Fallback face box in upper-central region
            ex1 = int(w * 0.25)
            ey1 = int(h * 0.05)
            ex2 = int(w * 0.75)
            ey2 = int(h * 0.65)
            faces_out.append({
                "bbox": (ex1, ey1, ex2, ey2),
                "orig_bbox": (ex1, ey1, ex2, ey2),
                "confidence": 0.50
            })

        return faces_out

def generate_soft_elliptical_mask(shape: tuple, blur_ksize: int = 31) -> np.ndarray:
    """Generates a 2D float32 feathered elliptical mask (0.0 to 1.0)."""
    h, w = shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    center = (w // 2, h // 2)
    axes = (w // 2, h // 2)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)
    
    # Feather boundary using Gaussian blur
    if blur_ksize % 2 == 0:
        blur_ksize += 1
    blurred = cv2.GaussianBlur(mask, (blur_ksize, blur_ksize), 0)
    return blurred

def apply_face_preservation(
    native_rgb: np.ndarray,
    ai_4x_rgb: np.ndarray,
    faces: list,
    face_weight: float = 0.70
) -> np.ndarray:
    """
    Blends Real-ESRGAN AI 4x RGB with Lanczos conventional 4x upscale of native face crops
    using feathered elliptical face masks.
    """
    h_native, w_native, _ = native_rgb.shape
    target_h, target_w = h_native * 4, w_native * 4
    
    # Conventional 4x Lanczos upscale of native RGB
    conv_4x_rgb = cv2.resize(native_rgb, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
    
    result_rgb = ai_4x_rgb.copy()
    
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        
        # 4x coordinates
        x1_4x, y1_4x = x1 * 4, y1 * 4
        x2_4x, y2_4x = x2 * 4, y2 * 4
        
        face_h_4x = y2_4x - y1_4x
        face_w_4x = x2_4x - x1_4x
        
        if face_h_4x <= 0 or face_w_4x <= 0:
            continue
            
        ai_crop = ai_4x_rgb[y1_4x:y2_4x, x1_4x:x2_4x, :].astype(np.float32)
        conv_crop = conv_4x_rgb[y1_4x:y2_4x, x1_4x:x2_4x, :].astype(np.float32)
        
        # Soft elliptical mask
        soft_mask = generate_soft_elliptical_mask((face_h_4x, face_w_4x), blur_ksize=max(31, min(face_w_4x, face_h_4x) // 4))
        soft_mask_3d = (soft_mask * face_weight)[:, :, np.newaxis]
        
        # Blend: AI * (mask * weight) + Conventional * (1 - mask * weight)
        blended = ai_crop * soft_mask_3d + conv_crop * (1.0 - soft_mask_3d)
        blended_u8 = np.clip(blended, 0, 255).astype(np.uint8)
        
        result_rgb[y1_4x:y2_4x, x1_4x:x2_4x, :] = blended_u8

    return result_rgb

def main():
    print("=" * 70)
    print("PHASE 2 — FACE PRESERVATION & FACE-AWARE BLENDING BENCHMARK")
    print("=" * 70)

    test_img_path = os.path.abspath(os.path.join("test_data", "test_human_images", "19035828_web1__12294096_web1_180615-PNR-newmayorchallenge.jpg"))
    orig_pil = Image.open(test_img_path).convert("RGB")
    orig_rgb = np.array(orig_pil, dtype=np.uint8)
    h_orig, w_orig, _ = orig_rgb.shape

    # Detect faces
    detector = FaceDetector()
    faces = detector.detect_faces(orig_rgb, pad_pct=0.25)
    print(f"Detected Faces Count: {len(faces)}")
    for idx, f in enumerate(faces):
        print(f"  Face #{idx+1}: BBox {f['bbox']} Confidence: {f['confidence']:.4f}")

    sr_mgr = SuperResolutionManager.get_instance()
    session = sr_mgr.get_session()

    # Run Phase 1 Contextual Tiling SR
    t0 = time.perf_counter()
    from backend.tests.test_phase1_tile_seams import run_contextual_tiled_sr
    ai_4x_rgb = run_contextual_tiled_sr(orig_rgb, session, core_size=64, tile_pad=32)
    t_sr = (time.perf_counter() - t0) * 1000.0

    print(f"\nPhase 1 Real-ESRGAN 4x Inference Time: {t_sr:.2f} ms")

    # Benchmark Face Blending Weights
    weights = [1.00, 0.80, 0.70, 0.60, 0.00]
    print("\n--- BENCHMARKING FACE BLENDING WEIGHTS ---")
    for w in weights:
        t_b0 = time.perf_counter()
        blended_4x = apply_face_preservation(orig_rgb, ai_4x_rgb, faces, face_weight=w)
        t_b = (time.perf_counter() - t_b0) * 1000.0
        
        # Save face crops for visual inspection
        f0 = faces[0]["bbox"]
        crop_4x = blended_4x[f0[1]*4:f0[3]*4, f0[0]*4:f0[2]*4, :]
        fname = f"phase2_face_weight_{int(w*100)}.png"
        cv2.imwrite(os.path.join(OUTPUT_DIR, fname), cv2.cvtColor(crop_4x, cv2.COLOR_RGB2BGR))
        
        print(f"Weight AI={w*100:3.0f}% | Conv={(1-w)*100:3.0f}% -> Blend Time: {t_b:.2f} ms | Saved {fname}")

    # Benchmark 2x Export Option A vs Option B
    print("\n--- BENCHMARKING 2x EXPORT PIPELINE OPTIONS ---")
    # Option A: Face preservation BEFORE 2x downsample (at 4x)
    t_a0 = time.perf_counter()
    blended_4x_optA = apply_face_preservation(orig_rgb, ai_4x_rgb, faces, face_weight=0.70)
    optA_2x_rgb = cv2.resize(blended_4x_optA, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_LANCZOS4)
    t_optA = (time.perf_counter() - t_a0) * 1000.0

    # Option B: Face preservation AFTER 2x downsample (at 2x)
    t_b0 = time.perf_counter()
    raw_2x_rgb = cv2.resize(ai_4x_rgb, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_LANCZOS4)
    conv_2x_rgb = cv2.resize(orig_rgb, (w_orig * 2, h_orig * 2), interpolation=cv2.INTER_LANCZOS4)
    optB_2x_rgb = raw_2x_rgb.copy()
    for face in faces:
        x1, y1, x2, y2 = face["bbox"]
        x1_2x, y1_2x, x2_2x, y2_2x = x1*2, y1*2, x2*2, y2*2
        fh_2x, fw_2x = y2_2x - y1_2x, x2_2x - x1_2x
        ai_crop_2x = raw_2x_rgb[y1_2x:y2_2x, x1_2x:x2_2x, :].astype(np.float32)
        conv_crop_2x = conv_2x_rgb[y1_2x:y2_2x, x1_2x:x2_2x, :].astype(np.float32)
        mask_2x = generate_soft_elliptical_mask((fh_2x, fw_2x), blur_ksize=max(15, min(fw_2x, fh_2x) // 4))[:, :, np.newaxis] * 0.70
        blended_2x = ai_crop_2x * mask_2x + conv_crop_2x * (1.0 - mask_2x)
        optB_2x_rgb[y1_2x:y2_2x, x1_2x:x2_2x, :] = np.clip(blended_2x, 0, 255).astype(np.uint8)
    t_optB = (time.perf_counter() - t_b0) * 1000.0

    diff_optA_optB = np.mean(np.abs(optA_2x_rgb.astype(float) - optB_2x_rgb.astype(float)))
    print(f"Option A (Preserve at 4x then downsample to 2x) Time: {t_optA:.2f} ms")
    print(f"Option B (Downsample to 2x then preserve at 2x) Time: {t_optB:.2f} ms")
    print(f"Mean Pixel Difference between Option A and Option B: {diff_optA_optB:.4f}")

    cv2.imwrite(os.path.join(OUTPUT_DIR, "phase2_2x_OptionA.png"), cv2.cvtColor(optA_2x_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(OUTPUT_DIR, "phase2_2x_OptionB.png"), cv2.cvtColor(optB_2x_rgb, cv2.COLOR_RGB2BGR))

    print("\n[SUCCESS] Phase 2 Benchmarking Complete!")

if __name__ == "__main__":
    main()
