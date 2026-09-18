"""
FastAPI Routes for Background Remover
Endpoints:
- POST /api/background-remover/remove: Process image and return transparent base64 image + metadata + diagnostics
- POST /api/background-remover/diagnostics: Development diagnostic endpoint returning pipeline stages + metrics
- POST /api/background-remover/composite: Composite image with custom background and return exported base64
- GET /api/background-remover/health: Health check & engine metadata
- GET /api/background-remover/status: System status & device info
- GET /api/background-remover/debug/latest: Latest diagnostic file paths
"""

import io
import gc
import time
import base64
import asyncio
from typing import Optional
import numpy as np
from PIL import Image
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ...background_removal_engine import BackgroundRemovalEngine, get_latest_debug_info
from ...image_processing import export_rgba_bytes, composite_background, get_process_memory_mb
from ...segmentation.model_manager import ModelManager, ManagerState
from ...processing.super_resolution import SuperResolutionManager
from ...processing.phase10_profiler import LiveTerminalProfiler

router = APIRouter(prefix="/api/background-remover", tags=["Background Remover"])


class CompositeRequest(BaseModel):
    image_base64: str
    bg_type: str = "transparent"  # 'transparent' | 'solid' | 'gradient'
    color1: Optional[str] = "#FFFFFF"
    color2: Optional[str] = "#000000"
    gradient_direction: Optional[str] = "to-bottom"
    export_format: Optional[str] = "PNG"  # PNG, WEBP, JPG
    quality: Optional[int] = 90


class ExportHDRequest(BaseModel):
    image_base64: str
    export_quality: str = "hd2x"  # 'hd2x' | 'hd4x'
    export_format: Optional[str] = "PNG"  # PNG, WEBP, JPG
    bg_type: str = "transparent"  # 'transparent' | 'solid' | 'gradient'
    color1: Optional[str] = "#FFFFFF"
    color2: Optional[str] = "#000000"
    gradient_direction: Optional[str] = "to-bottom"
    quality: Optional[int] = 95


@router.get("/health")
def health_check():
    mgr = ModelManager.get_instance()
    return {
        "status": "online",
        "engine": "BackgroundRemovalEngine (BiRefNet General FP16 ONNX)",
        "model": "BiRefNet General FP16 (ONNX)",
        "device": mgr.device,
        "privacy": "In-memory processing with immediate cleanup (no server storage)",
    }


@router.get("/status")
def get_system_status():
    mgr = ModelManager.get_instance()
    return {
        "engine": "BackgroundRemovalEngine",
        "architecture": "BiRefNet General FP16 ONNX",
        "device": mgr.device,
        "providers": mgr.providers,
        "state": mgr.state.value,
        "model_loaded": mgr.is_loaded,
        "inference_busy": mgr.is_inference_busy,
        "process_memory_mb": get_process_memory_mb(),
        "idle_timeout_seconds": mgr.idle_timeout_seconds,
        "idle_timer_active": mgr.idle_timer_active,
        "idle_remaining_seconds": mgr.idle_remaining_seconds,
        "cached_models": {
            "birefnet_general": mgr._birefnet_general is not None,
        },
        "load_times_ms": mgr._load_times,
    }


@router.get("/debug/latest")
def get_debug_latest_endpoint():
    return get_latest_debug_info()


@router.post("/remove")
async def remove_background(
    file: UploadFile = File(...),
    sensitivity: float = Form(10.0),
    edge_softness: float = Form(50.0),
    defringe_strength: float = Form(50.0),
    diagnostics: bool = Form(False),
):
    """
    Accepts an uploaded image file, processes it purely in memory via BackgroundRemovalEngine
    on a dedicated worker thread, enforcing strictly ONE inference at a time.
    """
    allowed_types = ["image/jpeg", "image/png", "image/webp", "image/avif", "image/bmp", "image/tiff"]
    if file.content_type and file.content_type.lower() not in allowed_types and not file.filename.lower().endswith(
        ('.jpg', '.jpeg', '.png', '.webp', '.avif', '.bmp', '.tiff')
    ):
        raise HTTPException(status_code=400, detail="Unsupported image format. Please upload JPG, PNG, WEBP, or AVIF.")

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(image_bytes) > 60 * 1024 * 1024:  # 60MB max
        raise HTTPException(status_code=400, detail="Image size exceeds safe maximum limit of 60MB.")

    mgr = ModelManager.get_instance()

    # Rule 1: Thread-safe single-inference check
    if not mgr.try_acquire_inference():
        raise HTTPException(
            status_code=429,
            detail="Background removal is currently processing another image. Please wait.",
        )

    # Cancel any active idle timer since new request has started
    mgr.cancel_idle_timer()

    try:
        # Rule 2 & 3: Ensure model is loaded lazily with single-flight protection
        mgr.get_birefnet_general()

        # Rule 5: Run CPU-bound inference in worker thread to keep FastAPI event loop responsive
        result = await asyncio.to_thread(
            BackgroundRemovalEngine.process_image,
            image_bytes=image_bytes,
            sensitivity=float(sensitivity),
            edge_softness=float(edge_softness),
            defringe_strength=float(defringe_strength),
            source="website",
            filename=file.filename,
        )

        t_enc = time.perf_counter()
        png_bytes = export_rgba_bytes(result.rgba_image, export_format="PNG")
        base64_data = base64.b64encode(png_bytes).decode("utf-8")
        encoding_ms = round((time.perf_counter() - t_enc) * 1000.0, 2)
        result.stage_timings["encoding_ms"] = encoding_ms

        data_uri = f"data:image/png;base64,{base64_data}"

        # Clean up response buffers
        del png_bytes
        result.rgba_image = None

        response_payload = {
            "success": True,
            "requestId": result.request_id,
            "engine": "BackgroundRemovalEngine",
            "model": result.diagnostics.get("model_name", "BiRefNet General (FP16 ONNX)"),
            "device": result.diagnostics.get("device", "CPU"),
            "width": result.width,
            "height": result.height,
            "processingTimeMs": result.processing_time_ms,
            "originalFormat": result.original_format,
            "resultDataUri": data_uri,
            "stageTimings": result.stage_timings,
            "diagnostics": result.diagnostics,
            "privacyNotice": "Your image was processed strictly in memory and was not saved to disk.",
        }

        return JSONResponse(response_payload)

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Background removal failed: {str(e)}")
    finally:
        # Rules 7, 13, 15: Always release lock, run cleanup, and start idle timer
        mgr.set_state(ManagerState.CLEANING_UP)
        try:
            del image_bytes
        except Exception:
            pass
        gc.collect()
        mgr.release_inference()
        mgr.schedule_idle_unload()


@router.post("/diagnostics")
async def run_diagnostics_endpoint(
    file: UploadFile = File(...),
    sensitivity: float = Form(10.0),
    edge_softness: float = Form(50.0),
    defringe_strength: float = Form(50.0),
):
    """
    Dedicated Development Diagnostic endpoint using BackgroundRemovalEngine with single-inference protection.
    """
    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    mgr = ModelManager.get_instance()
    if not mgr.try_acquire_inference():
        raise HTTPException(
            status_code=429,
            detail="Background removal is currently processing another image. Please wait.",
        )

    mgr.cancel_idle_timer()

    try:
        mgr.get_birefnet_general()
        result = await asyncio.to_thread(
            BackgroundRemovalEngine.process_image,
            image_bytes=image_bytes,
            sensitivity=float(sensitivity),
            edge_softness=float(edge_softness),
            defringe_strength=float(defringe_strength),
            source="website",
            filename=file.filename,
        )

        png_bytes = export_rgba_bytes(result.rgba_image, export_format="PNG")
        base64_data = base64.b64encode(png_bytes).decode("utf-8")
        data_uri = f"data:image/png;base64,{base64_data}"
        del png_bytes
        result.rgba_image = None

        return JSONResponse({
            "success": True,
            "requestId": result.request_id,
            "engine": "BackgroundRemovalEngine",
            "model": result.diagnostics.get("model_name", "BiRefNet General (FP16 ONNX)"),
            "device": result.diagnostics.get("device", "CPU"),
            "width": result.width,
            "height": result.height,
            "processingTimeMs": result.processing_time_ms,
            "stageTimings": result.stage_timings,
            "diagnostics": result.diagnostics,
            "resultDataUri": data_uri,
        })
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Diagnostic pipeline failed: {str(e)}")
    finally:
        mgr.set_state(ManagerState.CLEANING_UP)
        try:
            del image_bytes
        except Exception:
            pass
        gc.collect()
        mgr.release_inference()
        mgr.schedule_idle_unload()


@router.post("/composite")
async def composite_endpoint(request: CompositeRequest):
    """
    Composites a transparent cutout base64 image onto a custom background (solid, gradient, or transparent)
    and returns the exported base64 image.
    """
    try:
        t0 = time.perf_counter()
        data = request.image_base64
        if "," in data:
            data = data.split(",", 1)[1]
        raw_bytes = base64.b64decode(data)
        t_b64 = round((time.perf_counter() - t0) * 1000.0, 2)

        t0 = time.perf_counter()
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
        rgba_array = np.array(pil_img, dtype=np.uint8)
        orig_h, orig_w = rgba_array.shape[:2]
        t_dec = round((time.perf_counter() - t0) * 1000.0, 2)

        t0 = time.perf_counter()
        composited_rgba = composite_background(
            rgba_image=rgba_array,
            bg_type=request.bg_type,
            color1=request.color1 or "#FFFFFF",
            color2=request.color2 or "#000000",
            gradient_direction=request.gradient_direction or "to-bottom",
        )
        t_comp = round((time.perf_counter() - t0) * 1000.0, 2)

        fmt = (request.export_format or "PNG").upper()
        t0 = time.perf_counter()
        out_bytes = export_rgba_bytes(
            composited_rgba,
            export_format=fmt,
            quality=request.quality or 90,
        )
        t_enc = round((time.perf_counter() - t0) * 1000.0, 2)

        out_b64 = base64.b64encode(out_bytes).decode("utf-8")
        mime = "image/jpeg" if fmt in ("JPG", "JPEG") else ("image/webp" if fmt == "WEBP" else "image/png")
        out_kb = round(len(out_bytes) / 1024.0, 2)

        req_id = f"CMP-{time.strftime('%Y%m%d')}-{int(time.time()*1000)%1000:03d}"
        prof = LiveTerminalProfiler(
            request_id=req_id,
            endpoint="/api/background-remover/composite",
            operation="EXPORT_CUTOUT",
            orig_w=orig_w,
            orig_h=orig_h,
            format_str="RGBA",
            scale="Native",
            output_format=fmt,
        )
        prof.export_info = {
            "export_type": "Native Cutout",
            "input_dim": f"{orig_w} × {orig_h}",
            "output_dim": f"{orig_w} × {orig_h}",
            "format": fmt,
            "encoding_ms": t_enc,
            "output_size_str": f"{out_kb} KB",
        }
        prof.record_stage("b64_decode_ms", t_b64)
        prof.record_stage("decode_ms", t_dec)
        prof.record_stage("composite_ms", t_comp)
        prof.record_stage("encoding_ms", t_enc)
        perf_data = prof.generate_and_log_report()

        return JSONResponse({
            "success": True,
            "resultDataUri": f"data:{mime};base64,{out_b64}",
            "format": fmt,
            "width": composited_rgba.shape[1],
            "height": composited_rgba.shape[0],
            "perfMetrics": {
                "totalMs": perf_data.get("total_time_ms", 0.0),
                "encodingMs": t_enc,
                "compositeMs": t_comp,
                "inputWidth": orig_w,
                "inputHeight": orig_h,
                "outputWidth": composited_rgba.shape[1],
                "outputHeight": composited_rgba.shape[0],
            },
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Compositing failed: {str(e)}")


@router.post("/export-hd")
async def export_hd_endpoint(request: ExportHDRequest):
    """
    Executes AI Super-Resolution export (HD 2x or Ultra HD 4x) using Real-ESRGAN x4plus.
    Protected by global HEAVY_AI_LOCK. Automatically unloads BiRefNet before running Real-ESRGAN.
    Returns enhanced base64 result + performance and memory metrics.
    """
    try:
        data = request.image_base64
        if "," in data:
            data = data.split(",", 1)[1]
        raw_bytes = base64.b64decode(data)
        pil_img = Image.open(io.BytesIO(raw_bytes)).convert("RGBA")
        rgba_array = np.array(pil_img, dtype=np.uint8)

        sr_mgr = SuperResolutionManager.get_instance()

        # Execute AI Super Resolution on worker thread
        ai_rgba, perf_metrics = await asyncio.to_thread(
            sr_mgr.upscale_rgba_ai,
            rgba_array=rgba_array,
            export_quality=request.export_quality,
        )

        # Composite background if solid or gradient selected
        composited_rgba = composite_background(
            rgba_image=ai_rgba,
            bg_type=request.bg_type,
            color1=request.color1 or "#FFFFFF",
            color2=request.color2 or "#000000",
            gradient_direction=request.gradient_direction or "to-bottom",
        )

        fmt = (request.export_format or "PNG").upper()
        out_bytes = export_rgba_bytes(
            composited_rgba,
            export_format=fmt,
            quality=request.quality or 95,
        )
        out_b64 = base64.b64encode(out_bytes).decode("utf-8")
        mime = "image/jpeg" if fmt in ("JPG", "JPEG") else ("image/webp" if fmt == "WEBP" else "image/png")

        resp = JSONResponse({
            "success": True,
            "resultDataUri": f"data:{mime};base64,{out_b64}",
            "format": fmt,
            "quality": request.export_quality,
            "width": composited_rgba.shape[1],
            "height": composited_rgba.shape[0],
            "perfMetrics": perf_metrics,
        })

        del raw_bytes, pil_img, rgba_array, ai_rgba, composited_rgba, out_bytes, out_b64
        gc.collect()

        return resp
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"AI Super Resolution export failed: {str(e)}")

