"""
Unified AI Background Removal Engine
Production-grade background removal using strictly ONE AI model:
- BiRefNet General FP16 ONNX (Universal Salient Object & Portrait Segmentation)

Complete Pipeline:
INPUT IMAGE
    ↓
BiRefNet General FP16 ONNX
    ↓
RAW FLOAT PROBABILITY [0.0, 1.0]
    ↓
MASK QUALITY ANALYSIS (Pure Classical CV)
    ↓
TRIMAP (0 = Background, 128 = Unknown, 255 = Foreground)
    ↓
EDGE / HAIR REFINEMENT (Transition Band Filtering)
    ↓
TRANSPARENCY ANALYSIS (Classical Specular & Contrast Detection)
    ↓
ALPHA REFINEMENT (Fractional Alpha for Glassware / Solid Core for Opaque)
    ↓
DE-FRINGE / HALO REMOVAL (Color Decontamination)
    ↓
FINAL FLOAT ALPHA -> UINT8 [0, 255]
    ↓
RGBA ASSEMBLY (Original Resolution & Orientation Strictly Preserved)
    ↓
PNG / WEBP / JPG
"""

import os
import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image

from .image_processing import validate_and_decode_image, get_process_memory_mb
from .segmentation.model_manager import ModelManager
from .processing.mask_quality import MaskQualityAnalyzer, MaskQualityMetrics
from .processing.trimap import generate_trimap, TrimapResult
from .processing.transparency_analyzer import TransparencyAnalyzer, TransparencyResult
from .processing.alpha_refinement import refine_alpha_matte, RefinedAlphaResult
from .processing.defringe import defringe_boundary
from .processing.alpha import assemble_rgba_result
from .processing.phase10_profiler import Phase10Profiler

logger = logging.getLogger("BackgroundRemoval.Engine")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(name)s [%(levelname)s]: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

DEBUG_BACKGROUND_REMOVAL = os.getenv("DEBUG_BACKGROUND_REMOVAL", "false").lower() in ("true", "1", "yes")
DEBUG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug", "background_removal")

_latest_debug_info: Dict[str, Any] = {}


def get_latest_debug_info() -> Dict[str, Any]:
    """
    Returns diagnostic file paths and timings for the most recent test run.
    """
    if not DEBUG_BACKGROUND_REMOVAL:
        return {"status": "disabled", "message": "DEBUG_BACKGROUND_REMOVAL is disabled."}
    if not _latest_debug_info:
        return {"status": "empty", "message": "No runs have been processed yet."}
    return {"status": "success", **_latest_debug_info}


class EngineProcessingResult:
    def __init__(
        self,
        rgba_image: np.ndarray,
        width: int,
        height: int,
        processing_time_ms: float,
        original_format: str,
        stage_timings: Dict[str, float],
        diagnostics: Dict[str, Any],
        is_fallback: bool = False,
        request_id: str = "",
    ):
        self.rgba_image = rgba_image
        self.width = width
        self.height = height
        self.processing_time_ms = processing_time_ms
        self.original_format = original_format
        self.stage_timings = stage_timings
        self.diagnostics = diagnostics
        self.is_fallback = is_fallback
        self.request_id = request_id


class BackgroundRemovalEngine:
    """
    Single unified production background-removal engine.
    Uses BiRefNet General FP16 ONNX exclusively across all image types.
    """
    _request_counter: int = 0

    @classmethod
    def _generate_request_id(cls) -> str:
        cls._request_counter += 1
        date_str = datetime.now().strftime("%Y%m%d")
        return f"BG-{date_str}-{cls._request_counter:03d}"

    @classmethod
    def process_image(
        cls,
        image_bytes: bytes,
        sensitivity: float = 10.0,
        edge_softness: float = 50.0,
        defringe_strength: float = 50.0,
        request_id: Optional[str] = None,
        source: str = "test",
        filename: Optional[str] = None,
    ) -> EngineProcessingResult:
        """
        Executes the single-model AI background removal pipeline with native resolution preservation.
        Accepts parameters as 0-100 (%) or 0.0-1.0 and normalizes them internally.
        """
        total_start = time.perf_counter()
        timings: Dict[str, float] = {}

        if not request_id:
            request_id = cls._generate_request_id()

        rss_start = get_process_memory_mb()
        logger.info(f"REQUEST START [Request ID: {request_id}] (Process RSS: {rss_start} MB)")

        # Normalize parameters from 0-100 to 0.0-1.0
        if sensitivity > 1.0:
            norm_sensitivity = float(sensitivity) / 100.0
            disp_sensitivity = int(round(sensitivity))
        else:
            norm_sensitivity = float(sensitivity)
            disp_sensitivity = int(round(sensitivity * 100.0))

        if edge_softness > 1.0:
            norm_edge_softness = float(edge_softness) / 100.0
            disp_softness = int(round(edge_softness))
        else:
            norm_edge_softness = float(edge_softness)
            disp_softness = int(round(edge_softness * 100.0))

        if defringe_strength > 1.0:
            norm_defringe = float(defringe_strength) / 100.0
            disp_defringe = int(round(defringe_strength))
        else:
            norm_defringe = float(defringe_strength)
            disp_defringe = int(round(defringe_strength * 100.0))

        norm_sensitivity = float(np.clip(norm_sensitivity, 0.0, 1.0))
        norm_edge_softness = float(np.clip(norm_edge_softness, 0.0, 1.0))
        norm_defringe = float(np.clip(norm_defringe, 0.0, 1.0))

        # 1. Safe Decode & Dimension Validation (EXIF orientation preserved)
        t0 = time.perf_counter()
        rgb_image, original_format, orig_w, orig_h = validate_and_decode_image(image_bytes)
        timings["decode_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

        # Immediate Request Start Logging in backend terminal
        is_reused_initial = ModelManager.get_instance().is_loaded
        prof = Phase10Profiler(
            request_id=request_id,
            endpoint="/api/background-remover/remove",
            operation="BACKGROUND_REMOVAL" if is_reused_initial else "BACKGROUND_REMOVAL (COLD)",
            orig_w=orig_w,
            orig_h=orig_h,
            format_str=original_format,
            scale="Native",
            output_format="PNG",
        )
        prof.t_start = total_start
        prof.record_stage("decode_ms", timings["decode_ms"])
        prof.log_request_start()

        try:
            model_mgr = ModelManager.get_instance()
            model = model_mgr.get_birefnet_general()
            # 2. Inference: BiRefNet General FP16 ONNX -> Raw Float32 Probability Map [0.0, 1.0]
            logger.info(f"INFERENCE START [Request ID: {request_id}]: BiRefNet General FP16 ONNX")
            t0 = time.perf_counter()
            prob_map = model.predict_probability_map(rgb_image)
            inference_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            timings["inference_ms"] = inference_time_ms
            rss_after_onnx = get_process_memory_mb()
            logger.info(f"INFERENCE COMPLETE [Request ID: {request_id}]: {inference_time_ms} ms (Process RSS: {rss_after_onnx} MB)")

            # 3. Mask Quality Analysis (Pure Classical CV)
            t0 = time.perf_counter()
            quality_metrics = MaskQualityAnalyzer.evaluate(prob_map)
            timings["mask_quality_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

            # 4. Trimap Generation (Definite FG, Transition, Definite BG)
            t0 = time.perf_counter()
            trimap_result = generate_trimap(prob_map=prob_map, sensitivity=norm_sensitivity)
            timings["trimap_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            del prob_map  # Stage cleanup: clean_prob_map retained in trimap_result
            rss_after_trimap = get_process_memory_mb()

            # 5. Transparency Analysis (Pure Classical: fractional probability, edge gradient, contrast)
            t0 = time.perf_counter()
            transparency_result = TransparencyAnalyzer.analyze(
                rgb_image=rgb_image,
                prob_map=trimap_result.clean_prob_map,
                trimap=trimap_result.trimap,
            )
            timings["transparency_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)

            # 6. Alpha Refinement (Continuous float probability preserved throughout)
            t0 = time.perf_counter()
            refined_alpha_res = refine_alpha_matte(
                rgb_image=rgb_image,
                prob_map=trimap_result.clean_prob_map,
                trimap=trimap_result.trimap,
                edge_softness=norm_edge_softness,
                transparency_result=transparency_result,
            )
            refined_alpha = refined_alpha_res.final_alpha
            timings["alpha_refine_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            del trimap_result  # Stage cleanup: trimap buffers released
            rss_after_alpha = get_process_memory_mb()

            # 7. Boundary De-Fringing / Halo Removal
            t0 = time.perf_counter()
            clean_rgb = defringe_boundary(
                rgb_image=rgb_image,
                alpha=refined_alpha,
                defringe_strength=norm_defringe,
            )
            timings["defringe_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            rss_after_defringe = get_process_memory_mb()

            # 8. RGBA Assembly & Native Dimension Verification
            t0 = time.perf_counter()
            rgba_image, alpha_metrics = assemble_rgba_result(clean_rgb, refined_alpha)
            timings["assembly_ms"] = round((time.perf_counter() - t0) * 1000.0, 2)
            del clean_rgb, refined_alpha  # Stage cleanup: assembled RGBA retained
            rss_after_assembly = get_process_memory_mb()

            sub_timings = getattr(model, "latest_timings", {})
            timings["preprocess_ms"] = sub_timings.get("preprocess_ms", 0.0)
            timings["onnx_inference_ms"] = sub_timings.get("onnx_inference_ms", inference_time_ms)
            timings["sigmoid_ms"] = sub_timings.get("sigmoid_ms", 0.0)
            timings["mask_resize_ms"] = sub_timings.get("mask_resize_ms", 0.0)
            
            # Check if model load happened on this request
            raw_load_ms = model_mgr.load_times.get("birefnet_general", 0.0)
            is_reused = getattr(model_mgr, "_birefnet_reused_this_req", True)
            effective_load_ms = 0.0 if is_reused else raw_load_ms
            timings["model_load_ms"] = effective_load_ms
            timings["warmup_ms"] = model_mgr.load_times.get("warmup", 0.0)

            total_elapsed_ms = round((time.perf_counter() - total_start) * 1000.0, 2)
            refinement_time_ms = round(
                timings["trimap_ms"]
                + timings["transparency_ms"]
                + timings["alpha_refine_ms"]
                + timings["defringe_ms"],
                2,
            )
            timings["refinement_ms"] = refinement_time_ms

            rss_cleanup = get_process_memory_mb()
            logger.info(f"CLEANUP COMPLETE [Request ID: {request_id}]: Intermediate pipeline arrays released (Process RSS: {rss_cleanup} MB)")

            prof.operation = "BACKGROUND_REMOVAL" if is_reused else "BACKGROUND_REMOVAL (COLD)"
            for k, v in timings.items():
                prof.record_stage(k, v)
            
            provider_name = model_mgr.providers[0] if model_mgr.providers else "CPUExecutionProvider"
            prof.model_info = {
                "name": model.model_name,
                "file": "model_fp16.onnx",
                "input_res": f"{model.input_resolution[0]} × {model.input_resolution[1]}",
                "precision": "FP16",
                "provider": provider_name,
                "cpu_threads": 4,
                "reused": "YES" if is_reused else "NO",
                "load_ms": effective_load_ms,
                "warmup_ms": timings.get("warmup_ms", 0.0),
            }

            prof.hardware_info = {
                "provider": provider_name,
                "cpu_threads": 4,
                "gpu_device": "DirectML GPU" if "Dml" in provider_name else "NOT AVAILABLE",
                "gpu_metrics": "AVAILABLE" if "Dml" in provider_name else "NOT AVAILABLE",
            }

            prof.inference_stats = {
                "calls": 1,
                "total_ms": inference_time_ms,
                "avg_ms": inference_time_ms,
                "min_ms": inference_time_ms,
                "max_ms": inference_time_ms,
            }

            # Alpha pixel counts
            final_alpha = rgba_image[:, :, 3]
            opaque_cnt = int(np.count_nonzero(final_alpha == 255))
            trans_cnt = int(np.count_nonzero(final_alpha == 0))
            semi_cnt = int(np.count_nonzero((final_alpha > 0) & (final_alpha < 255)))

            prof.mask_metrics = {
                "fg_pixels": opaque_cnt + semi_cnt,
                "bg_pixels": trans_cnt,
                "semi_pixels": semi_cnt,
            }
            prof.alpha_metrics = {
                "opaque_pixels": opaque_cnt,
                "transparent_pixels": trans_cnt,
                "semi_pixels": semi_cnt,
            }

            perf_report = prof.generate_and_log_report()

            # Optional Debug Files: strictly disabled for website uploads to preserve user privacy
            req_debug_dir = ""
            if DEBUG_BACKGROUND_REMOVAL and source.lower() != "website":
                try:
                    folder_type = "tests"
                    req_debug_dir = os.path.join(DEBUG_DIR, folder_type, f"request_{request_id}")
                    os.makedirs(req_debug_dir, exist_ok=True)
                    os.makedirs(DEBUG_DIR, exist_ok=True)

                    raw_prob_u8 = np.clip(np.round(prob_map * 255.0), 0, 255).astype(np.uint8)

                    for target_dir in (req_debug_dir, DEBUG_DIR):
                        Image.fromarray(rgb_image).save(os.path.join(target_dir, "input.png"), format="PNG")
                        Image.fromarray(raw_prob_u8, mode="L").save(os.path.join(target_dir, "raw_probability.png"), format="PNG")
                        Image.fromarray(refined_alpha_res.trimap_vis).save(os.path.join(target_dir, "trimap.png"), format="PNG")
                        Image.fromarray(refined_alpha_res.alpha_vis).save(os.path.join(target_dir, "refined_mask.png"), format="PNG")
                        Image.fromarray(refined_alpha, mode="L").save(os.path.join(target_dir, "final_alpha.png"), format="PNG")
                        Image.fromarray(rgba_image, mode="RGBA").save(os.path.join(target_dir, "final_cutout.png"), format="PNG")

                    global _latest_debug_info
                    _latest_debug_info = {
                        "request_id": request_id,
                        "debug_dir": os.path.abspath(req_debug_dir),
                        "input_path": os.path.abspath(os.path.join(req_debug_dir, "input.png")),
                        "raw_probability_path": os.path.abspath(os.path.join(req_debug_dir, "raw_probability.png")),
                        "trimap_path": os.path.abspath(os.path.join(req_debug_dir, "trimap.png")),
                        "refined_mask_path": os.path.abspath(os.path.join(req_debug_dir, "refined_mask.png")),
                        "final_alpha_path": os.path.abspath(os.path.join(req_debug_dir, "final_alpha.png")),
                        "final_cutout_path": os.path.abspath(os.path.join(req_debug_dir, "final_cutout.png")),
                        "model_name": model.model_name,
                        "transparency_type": transparency_result.transparency_type,
                        "device": model_mgr.device,
                        "inference_time": inference_time_ms,
                        "refinement_time": refinement_time_ms,
                        "total_processing_time": total_elapsed_ms,
                        "image_dims": f"{orig_w}x{orig_h}",
                    }
                    logger.info(f"[DEBUG] Generated intermediate diagnostic files in: {req_debug_dir}")
                except Exception as dbg_err:
                    logger.error(f"Failed to generate debug images: {dbg_err}")

            diagnostics = {
                "request_id": request_id,
                "engine": "BackgroundRemovalEngine",
                "engine_type": "BiRefNet General FP16 ONNX",
                "model_name": model.model_name,
                "transparency_type": transparency_result.transparency_type,
                "is_transparent": transparency_result.is_transparent,
                "device": model_mgr.device,
                "sensitivity": disp_sensitivity,
                "edge_softness": disp_softness,
                "defringe_strength": disp_defringe,
                "image_width": orig_w,
                "image_height": orig_h,
                "total_pixels": orig_w * orig_h,
                "processing_time_ms": total_elapsed_ms,
                "mask_quality": quality_metrics.to_dict(),
                "alpha_metrics": alpha_metrics,
                "debug_folder": req_debug_dir if (DEBUG_BACKGROUND_REMOVAL and req_debug_dir) else None,
            }

            return EngineProcessingResult(
                rgba_image=rgba_image,
                width=orig_w,
                height=orig_h,
                processing_time_ms=total_elapsed_ms,
                original_format=original_format,
                stage_timings=timings,
                diagnostics=diagnostics,
                is_fallback=False,
                request_id=request_id,
            )

        except Exception as e:
            logger.error(f"Background removal failed: {str(e)}", exc_info=True)
            raise e
