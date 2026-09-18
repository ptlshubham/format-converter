"""
Phase 10B Live Terminal Performance Monitor & Profiler
Generates comprehensive human-readable terminal performance reports and persists structured JSON metrics.
"""

import os
import sys
import json
import time
import psutil
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np

# Ensure Windows terminal handles UTF-8 emojis and bar graphics smoothly
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logger = logging.getLogger("BackgroundRemoval.Profiler")

PERFORMANCE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tests", "diagnostic_results", "performance"
)


class LiveTerminalProfiler:
    """
    Unified Live Terminal Performance Monitor for all request types:
    - BACKGROUND_REMOVAL / RE_PROCESS
    - EXPORT_CUTOUT (Composite)
    - EXPORT_HD_2X / EXPORT_HD_4X (Super Resolution)
    """

    def __init__(
        self,
        request_id: str,
        endpoint: str,
        operation: str,
        orig_w: int,
        orig_h: int,
        format_str: str = "PNG",
        scale: str = "Native",
        output_format: str = "PNG",
    ):
        self.request_id = request_id
        self.endpoint = endpoint
        self.operation = operation
        self.orig_w = orig_w
        self.orig_h = orig_h
        self.orig_pixels = orig_w * orig_h
        self.format_str = format_str.upper()
        self.scale = scale
        self.output_format = output_format.upper()
        
        self.started_at = datetime.now().strftime("%H:%M:%S")
        self.t_start = time.perf_counter()
        self.timings: Dict[str, float] = {}
        
        # Memory tracking
        process = psutil.Process() if psutil else None
        self.rss_before = process.memory_info().rss / (1024 * 1024) if process else 0.0
        self.rss_peak = self.rss_before
        self.rss_after = self.rss_before
        
        # Model & Session Details
        self.model_info: Dict[str, Any] = {
            "name": "BiRefNet General",
            "file": "model_fp16.onnx",
            "input_res": "1024 × 1024",
            "precision": "FP16",
            "provider": "CPUExecutionProvider",
            "cpu_threads": 4,
            "reused": "NO",
            "load_ms": 0.0,
            "warmup_ms": 0.0,
        }
        
        self.inference_stats: Dict[str, Any] = {
            "calls": 1,
            "total_ms": 0.0,
            "avg_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "median_ms": 0.0,
            "std_dev_ms": 0.0,
        }
        
        self.hardware_info: Dict[str, Any] = {
            "provider": "CPUExecutionProvider",
            "cpu_threads": 4,
            "gpu_device": "NOT AVAILABLE",
            "gpu_metrics": "NOT AVAILABLE",
        }
        
        self.image_info: Dict[str, Any] = {
            "channels": "RGBA",
            "model_input": "1024 × 1024",
            "output_w": orig_w,
            "output_h": orig_h,
        }
        
        self.mask_metrics: Dict[str, Any] = {
            "fg_pixels": 0,
            "bg_pixels": 0,
            "semi_pixels": 0,
        }
        
        self.alpha_metrics: Dict[str, Any] = {
            "opaque_pixels": 0,
            "transparent_pixels": 0,
            "semi_pixels": 0,
        }
        
        self.face_info: Dict[str, Any] = {
            "detection_ms": 0.0,
            "faces_found": 0,
            "status": "SKIPPED",
        }
        
        self.export_info: Dict[str, Any] = {
            "export_type": scale,
            "input_dim": f"{orig_w} × {orig_h}",
            "output_dim": f"{orig_w} × {orig_h}",
            "format": self.output_format,
            "encoding_ms": 0.0,
            "output_size_str": "N/A",
            "output_bytes": 0,
        }
        
        self.safety_info: Dict[str, Any] = {
            "max_pixels": 33_177_600,
            "requested_pixels": self.orig_pixels,
            "status": "PASS",
        }
        
        self.lock_info: Dict[str, Any] = {
            "status": "RELEASED",
            "wait_ms": 0.0,
            "hold_ms": 0.0,
        }

    def update_rss(self):
        if psutil:
            curr_rss = psutil.Process().memory_info().rss / (1024 * 1024)
            self.rss_peak = max(self.rss_peak, curr_rss)
            self.rss_after = curr_rss

    def record_stage(self, stage_name: str, duration_ms: float):
        self.timings[stage_name] = round(duration_ms, 2)
        self.update_rss()

    def _safe_print(self, text: str):
        try:
            line_str = str(text)
        except Exception:
            line_str = repr(text)

        # 1. Output directly to sys.stderr (primary console stream used by Uvicorn on Windows)
        try:
            sys.stderr.write(line_str + "\n")
            sys.stderr.flush()
        except Exception:
            try:
                sys.stderr.write(line_str.encode("ascii", "replace").decode("ascii") + "\n")
                sys.stderr.flush()
            except Exception:
                pass

        # 2. Output directly to sys.stdout
        try:
            sys.stdout.write(line_str + "\n")
            sys.stdout.flush()
        except Exception:
            pass

    def log_request_start(self):
        """Prints the Request Start Log immediately when the request is received."""
        self._start_logged = True
        lines = []
        lines.append("\n" + "=" * 60)
        lines.append(f"🚀 {self.operation} REQUEST")
        lines.append("=" * 60)
        lines.append(f"Request ID       : {self.request_id}")
        lines.append(f"Started At       : {self.started_at}")
        lines.append(f"Endpoint         : {self.endpoint}")
        lines.append(f"Operation        : {self.operation}")
        lines.append(f"Input Format     : {self.format_str}")
        lines.append(f"Input Size       : {self.orig_w} × {self.orig_h}")
        lines.append(f"Input Pixels     : {self.orig_pixels:,}")
        lines.append(f"Input Channels   : {self.image_info.get('channels', 'RGBA')}")
        lines.append(f"Requested Scale  : {self.scale}")
        lines.append(f"Output Format    : {self.output_format}")
        lines.append("=" * 60)

        if "COMPOSITE" not in self.operation:
            lines.append("\n---")
            lines.append("🤖 MODEL")
            lines.append(f"Model Name       : {self.model_info.get('name', 'BiRefNet General')}")
            lines.append(f"Model File       : {self.model_info.get('file', 'model_fp16.onnx')}")
            lines.append(f"Model Input      : {self.model_info.get('input_res', '1024 × 1024')}")
            lines.append(f"Model Precision  : {self.model_info.get('precision', 'FP16')}")
            lines.append(f"Execution Provider: {self.model_info.get('provider', 'CPUExecutionProvider')}")
            lines.append(f"CPU Threads      : {self.model_info.get('cpu_threads', 4)}")
            lines.append(f"Session Status   : {'REUSED' if self.model_info.get('reused') == 'YES' else 'LOADED'}")
            lines.append("----------------------------------")
            lines.append("\n[PROCESSING] Processing image in memory... (Inference in progress, please wait)\n")

        for l in lines:
            self._safe_print(l)

    def generate_and_log_report(self) -> Dict[str, Any]:
        t_end = time.perf_counter()
        total_ms = (t_end - self.t_start) * 1000.0
        total_sec = total_ms / 1000.0
        self.update_rss()
        rss_delta = round(self.rss_after - self.rss_before, 2)

        # Build Stage Timing Breakdown depending on operation
        if "EXPORT_HD" in self.operation:
            stages_ordered = [
                ("Image Decode", self.timings.get("decode_ms", 0.0)),
                ("Image Preprocessing", self.timings.get("preprocess_ms", 0.0)),
                ("Face Detection", self.timings.get("face_detection_ms", 0.0)),
                ("Tile Preparation", self.timings.get("tile_prep_ms", 0.0)),
                ("Model Session Load", self.timings.get("model_load_ms", 0.0)),
                ("Model Inference", self.timings.get("ai_inference_ms", self.timings.get("inference_ms", 0.0))),
                ("Tile Reconstruction", self.timings.get("tile_recon_ms", 0.0)),
                ("Downsampling (2x)", self.timings.get("downsampling_ms", 0.0)),
                ("Face Preservation", self.timings.get("face_blending_ms", 0.0)),
                ("Alpha Recombination", self.timings.get("alpha_recombination_ms", 0.0)),
                ("Output Encoding", self.timings.get("encoding_ms", 0.0)),
                ("Response Preparation", self.timings.get("response_prep_ms", 1.0)),
            ]
        elif "COMPOSITE" in self.operation or "EXPORT_CUTOUT" in self.operation:
            stages_ordered = [
                ("Base64 Decode", self.timings.get("b64_decode_ms", 0.0)),
                ("Image Decoding", self.timings.get("decode_ms", 0.0)),
                ("Background Compositing", self.timings.get("composite_ms", 0.0)),
                ("Output Encoding", self.timings.get("encoding_ms", 0.0)),
                ("Response Preparation", self.timings.get("response_prep_ms", 1.0)),
            ]
        else:
            stages_ordered = [
                ("Image Decode", self.timings.get("decode_ms", 0.0)),
                ("Image Validation", self.timings.get("validation_ms", 0.1)),
                ("Image Preprocessing", self.timings.get("preprocess_ms", 0.0)),
                ("Model Session Load", self.timings.get("model_load_ms", 0.0)),
                ("Model Warmup", self.timings.get("warmup_ms", 0.0)),
                ("Model Inference", self.timings.get("onnx_inference_ms", self.timings.get("inference_ms", 0.0))),
                ("Sigmoid Activation", self.timings.get("sigmoid_ms", 0.0)),
                ("Mask Un-letterbox / Resize", self.timings.get("mask_resize_ms", 0.0)),
                ("Mask Quality Evaluation", self.timings.get("mask_quality_ms", 0.0)),
                ("Trimap Generation", self.timings.get("trimap_ms", 0.0)),
                ("Transparency Analysis", self.timings.get("transparency_ms", 0.0)),
                ("Alpha Matte Refinement", self.timings.get("alpha_refine_ms", 0.0)),
                ("De-Fringing / Halo Removal", self.timings.get("defringe_ms", 0.0)),
                ("RGBA Assembly", self.timings.get("assembly_ms", 0.0)),
                ("Output Encoding", self.timings.get("encoding_ms", 0.0)),
                ("Response Preparation", self.timings.get("response_prep_ms", 1.0)),
            ]

        # Calculate Stage Percentages
        pipeline_table = []
        total_stage_sum = sum(dur for _, dur in stages_ordered)
        effective_total = max(total_ms, total_stage_sum)

        for name, dur in stages_ordered:
            dur_sec = dur / 1000.0
            pct = (dur / effective_total * 100.0) if effective_total > 0 else 0.0
            pipeline_table.append((name, dur, dur_sec, pct))

        # Top Bottlenecks
        sorted_bottlenecks = sorted(pipeline_table, key=lambda x: x[1], reverse=True)

        # Output Terminal Report
        lines = []
        if not getattr(self, "_start_logged", False):
            lines.append("\n" + "=" * 60)
            lines.append(f"🚀 {self.operation} REQUEST")
            lines.append("=" * 60)
            lines.append(f"Request ID       : {self.request_id}")
            lines.append(f"Started At       : {self.started_at}")
            lines.append(f"Endpoint         : {self.endpoint}")
            lines.append(f"Operation        : {self.operation}")
            lines.append(f"Input Format     : {self.format_str}")
            lines.append(f"Input Size       : {self.orig_w} × {self.orig_h}")
            lines.append(f"Input Pixels     : {self.orig_pixels:,}")
            lines.append(f"Input Channels   : {self.image_info.get('channels', 'RGBA')}")
            lines.append(f"Requested Scale  : {self.scale}")
            lines.append(f"Output Format    : {self.output_format}")
            lines.append("=" * 60)

            # Model Info (if applicable)
            if "COMPOSITE" not in self.operation:
                lines.append("\n---")
                lines.append("🤖 MODEL")
                lines.append(f"Model Name       : {self.model_info.get('name', 'BiRefNet General')}")
                lines.append(f"Model File       : {self.model_info.get('file', 'model_fp16.onnx')}")
                lines.append(f"Model Input      : {self.model_info.get('input_res', '1024 × 1024')}")
                lines.append(f"Model Precision  : {self.model_info.get('precision', 'FP16')}")
                lines.append(f"Execution Provider: {self.model_info.get('provider', 'CPUExecutionProvider')}")
                lines.append(f"CPU Threads      : {self.model_info.get('cpu_threads', 4)}")
                lines.append(f"Session Status   : {'REUSED' if self.model_info.get('reused') == 'YES' else 'LOADED'}")
                lines.append("----------------------------------")

        # Model Session Details
        if "COMPOSITE" not in self.operation:
            lines.append("\n---")
            lines.append("⏱️ MODEL SESSION")
            load_ms = self.model_info.get('load_ms', 0.0)
            lines.append(f"Session Load Time : {load_ms:.2f} ms")
            lines.append(f"Session Load Time : {load_ms / 1000.0:.3f} sec")
            lines.append(f"Session Reused    : {self.model_info.get('reused', 'NO')}")
            lines.append(f"Warmup Time       : {self.model_info.get('warmup_ms', 0.0):.2f} ms")

        # Pipeline Timing Table
        lines.append("\n---")
        lines.append("📊 PIPELINE TIMING\n")
        for name, dur, dur_sec, pct in pipeline_table:
            lines.append(f"{name:<28}: {dur:8.2f} ms  ({dur_sec:6.3f} sec)  ({pct:6.2f}%)")
        lines.append("\n---")
        lines.append(f"TOTAL PROCESSING TIME        : {total_ms:8.2f} ms")
        lines.append(f"TOTAL PROCESSING TIME        : {total_sec:6.3f} sec")
        lines.append("----------------------------------------------")

        # Time Distribution
        lines.append("\n---")
        lines.append("📈 TIME DISTRIBUTION\n")
        for name, dur, dur_sec, pct in sorted_bottlenecks:
            if pct > 0.01:
                lines.append(f"{name:<24}: {pct:6.2f}%")
        lines.append("TOTAL                   : 100.00%")

        # Performance Graph Bar
        lines.append("\n---")
        lines.append("⚡ PERFORMANCE GRAPH\n")
        max_bars = 20
        for name, dur, dur_sec, pct in pipeline_table:
            if dur > 0.1:
                bars = max(1, int(round(dur / effective_total * max_bars))) if effective_total > 0 else 1
                bar_str = "█" * bars
                short_name = name.replace("Model ", "").replace("Image ", "").replace("Refinement", "Ref")[:14]
                lines.append(f"{short_name:<15} |{bar_str:<20}| {int(round(dur))} ms")

        # Inference Statistics
        if "COMPOSITE" not in self.operation:
            lines.append("\n---")
            lines.append("🧠 INFERENCE")
            calls = self.inference_stats.get("calls", 1)
            inf_tot = self.inference_stats.get("total_ms", 0.0)
            lines.append(f"Inference Calls : {calls}")
            lines.append(f"Total Inference : {inf_tot:.2f} ms")
            lines.append(f"Inference Time  : {inf_tot / 1000.0:.3f} sec")
            if calls > 1:
                lines.append(f"Average         : {self.inference_stats.get('avg_ms', 0.0):.2f} ms")
                lines.append(f"Minimum         : {self.inference_stats.get('min_ms', 0.0):.2f} ms")
                lines.append(f"Maximum         : {self.inference_stats.get('max_ms', 0.0):.2f} ms")

        # Memory Section
        lines.append("\n---")
        lines.append("💾 MEMORY")
        lines.append(f"RSS Before Request : {self.rss_before:.2f} MB")
        lines.append(f"RSS Peak           : {self.rss_peak:.2f} MB")
        lines.append(f"RSS After Request  : {self.rss_after:.2f} MB")
        lines.append(f"RSS Delta          : {rss_delta:+.2f} MB")

        # Hardware Runtime
        lines.append("\n---")
        lines.append("🖥️ RUNTIME")
        lines.append(f"Execution Provider : {self.hardware_info.get('provider', 'CPUExecutionProvider')}")
        lines.append(f"CPU Threads        : {self.hardware_info.get('cpu_threads', 4)}")
        lines.append(f"GPU Device         : {self.hardware_info.get('gpu_device', 'NOT AVAILABLE')}")
        lines.append(f"GPU Metrics        : {self.hardware_info.get('gpu_metrics', 'NOT AVAILABLE')}")

        # Image & Mask Info
        lines.append("\n---")
        lines.append("🖼️ IMAGE")
        lines.append(f"Input: {self.orig_w} × {self.orig_h}")
        lines.append(f"Pixels: {self.orig_pixels:,}")
        lines.append(f"Aspect Ratio: {round(self.orig_w / max(1, self.orig_h), 4):.4f}")
        lines.append(f"Channels: {self.image_info.get('channels', 'RGBA')}")
        lines.append(f"Model Input: {self.image_info.get('model_input', '1024 × 1024')}")
        lines.append(f"Output: {self.image_info.get('output_w', self.orig_w)} × {self.image_info.get('output_h', self.orig_h)}")
        
        if self.mask_metrics.get("fg_pixels", 0) > 0:
            lines.append("\nMASK")
            lines.append(f"Foreground Pixels       : {self.mask_metrics.get('fg_pixels', 0):,}")
            lines.append(f"Transparent Pixels      : {self.mask_metrics.get('bg_pixels', 0):,}")
            lines.append(f"Semi-transparent Pixels : {self.mask_metrics.get('semi_pixels', 0):,}")

        # Face Detection Info
        lines.append("\n---")
        lines.append("👤 FACE DETECTION")
        lines.append(f"Detection Time : {self.face_info.get('detection_ms', 0.0):.2f} ms")
        lines.append(f"Faces Found    : {self.face_info.get('faces_found', 0)}")
        lines.append(f"Face Processing: {self.face_info.get('status', 'SKIPPED')}")

        # Export Information (if export operation)
        if "EXPORT" in self.operation or "COMPOSITE" in self.operation:
            lines.append("\n---")
            lines.append("📤 EXPORT")
            lines.append(f"Export Type : {self.export_info.get('export_type', self.scale)}")
            lines.append(f"Input       : {self.export_info.get('input_dim', f'{self.orig_w} × {self.orig_h}')}")
            lines.append(f"Output      : {self.export_info.get('output_dim', f'{self.orig_w} × {self.orig_h}')}")
            lines.append(f"Format      : {self.export_info.get('format', self.output_format)}")
            lines.append(f"Encoding Time: {self.export_info.get('encoding_ms', 0.0):.2f} ms")
            lines.append(f"Output Size : {self.export_info.get('output_size_str', 'N/A')}")

        # Safety Guard
        lines.append("\n---")
        lines.append("🛡️ SAFETY")
        lines.append(f"MAX_EXPORT_PIXELS : {self.safety_info.get('max_pixels', 33177600):,}")
        lines.append(f"Requested Pixels  : {self.safety_info.get('requested_pixels', self.orig_pixels):,}")
        lines.append(f"Safety Status     : {self.safety_info.get('status', 'PASS')}")

        # AI Lock Info
        if "EXPORT_HD" in self.operation:
            lines.append("\n---")
            lines.append("🔒 AI LOCK")
            lines.append(f"Lock Status   : {self.lock_info.get('status', 'RELEASED')}")
            lines.append(f"Lock Wait Time: {self.lock_info.get('wait_ms', 0.0):.2f} ms")
            lines.append(f"Lock Hold Time: {self.lock_info.get('hold_ms', 0.0):.2f} ms")

        # Request End Summary
        lines.append("\n" + "=" * 60)
        lines.append("✅ REQUEST COMPLETED")
        lines.append("=" * 60)
        lines.append(f"Request ID       : {self.request_id}")
        lines.append(f"Status           : SUCCESS")
        lines.append(f"Input            : {self.orig_w} × {self.orig_h}")
        lines.append(f"Output           : {self.image_info.get('output_w', self.orig_w)} × {self.image_info.get('output_h', self.orig_h)}")
        lines.append(f"Operation        : {self.operation}")
        lines.append(f"Total Time       : {total_ms:.2f} ms ({total_sec:.3f} sec)")
        lines.append(f"Model Inference  : {self.inference_stats.get('total_ms', 0.0):.2f} ms")
        lines.append(f"Session Load     : {self.model_info.get('load_ms', 0.0):.2f} ms")
        post_proc_ms = total_ms - self.inference_stats.get('total_ms', 0.0) - self.model_info.get('load_ms', 0.0)
        lines.append(f"Post Processing  : {max(0.0, post_proc_ms):.2f} ms")
        lines.append(f"Faces            : {self.face_info.get('faces_found', 0)}")
        lines.append(f"Provider         : {self.hardware_info.get('provider', 'CPUExecutionProvider')}")
        lines.append("=" * 60)

        # Performance Indicator Level
        lines.append("\n" + "=" * 60)
        lines.append("⚡ PERFORMANCE STATUS")
        lines.append("=" * 60)
        lines.append(f"Total: {total_ms:,.2f} ms")

        dominant_stage = sorted_bottlenecks[0][0] if sorted_bottlenecks else "Unknown"
        dominant_time = sorted_bottlenecks[0][1] if sorted_bottlenecks else 0.0
        dominant_pct = sorted_bottlenecks[0][3] if sorted_bottlenecks else 0.0

        if total_ms > 10000.0 or dominant_pct > 70.0:
            perf_status = "🔴 INFERENCE BOTTLENECK"
        elif total_ms > 2000.0:
            perf_status = "🟡 MODERATE PROCESSING COST"
        else:
            perf_status = "🟢 OPTIMAL EXECUTION SPEED"

        lines.append(f"Status        : {perf_status}")
        lines.append(f"Dominant Stage: {dominant_stage}")
        lines.append(f"Dominant Time : {dominant_time:.2f} ms")
        lines.append(f"Percentage    : {dominant_pct:.2f}%")
        lines.append("=" * 60 + "\n")

        # Print to terminal
        for l in lines:
            self._safe_print(l)

        # Save Performance JSON
        os.makedirs(PERFORMANCE_DIR, exist_ok=True)
        json_path = os.path.join(PERFORMANCE_DIR, f"{self.request_id}.json")
        perf_data = {
            "request_id": self.request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endpoint": self.endpoint,
            "operation": self.operation,
            "input_dimensions": f"{self.orig_w}x{self.orig_h}",
            "output_dimensions": f"{self.image_info.get('output_w', self.orig_w)}x{self.image_info.get('output_h', self.orig_h)}",
            "input_format": self.format_str,
            "output_format": self.output_format,
            "scale": self.scale,
            "model_info": self.model_info,
            "provider": self.hardware_info.get("provider", "CPUExecutionProvider"),
            "session_reused": self.model_info.get("reused") == "YES",
            "session_load_ms": self.model_info.get("load_ms", 0.0),
            "warmup_ms": self.model_info.get("warmup_ms", 0.0),
            "stage_timings_ms": self.timings,
            "stage_percentages": {name: pct for name, _, _, pct in pipeline_table},
            "inference_statistics": self.inference_stats,
            "memory_statistics": {
                "rss_before_mb": self.rss_before,
                "rss_peak_mb": self.rss_peak,
                "rss_after_mb": self.rss_after,
                "rss_delta_mb": rss_delta,
            },
            "face_statistics": self.face_info,
            "safety_status": self.safety_info.get("status", "PASS"),
            "lock_statistics": self.lock_info,
            "total_time_ms": total_ms,
            "total_time_sec": total_sec,
            "performance_status": perf_status,
            "status": "SUCCESS",
        }

        with open(json_path, "w") as f:
            json.dump(perf_data, f, indent=2)

        return perf_data

    def generate_and_log_error_report(
        self, failed_stage: str, error_type: str, safe_message: str
    ) -> Dict[str, Any]:
        t_end = time.perf_counter()
        elapsed_ms = (t_end - self.t_start) * 1000.0

        lines = []
        lines.append("\n" + "=" * 60)
        lines.append("❌ REQUEST FAILED")
        lines.append("=" * 60)
        lines.append(f"Request ID   : {self.request_id}")
        lines.append(f"Endpoint     : {self.endpoint}")
        lines.append(f"Operation    : {self.operation}")
        lines.append(f"Status       : FAILED")
        lines.append(f"Failed Stage : {failed_stage}")
        lines.append(f"Error Type   : {error_type}")
        lines.append(f"Error        : {safe_message}")
        lines.append(f"Elapsed Time : {elapsed_ms:.2f} ms ({elapsed_ms / 1000.0:.3f} sec)")
        lines.append("=" * 60 + "\n")

        for l in lines:
            self._safe_print(l)

        os.makedirs(PERFORMANCE_DIR, exist_ok=True)
        json_path = os.path.join(PERFORMANCE_DIR, f"{self.request_id}.json")
        err_data = {
            "request_id": self.request_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "endpoint": self.endpoint,
            "operation": self.operation,
            "status": "FAILED",
            "failed_stage": failed_stage,
            "error_type": error_type,
            "error_message": safe_message,
            "elapsed_time_ms": elapsed_ms,
        }

        with open(json_path, "w") as f:
            json.dump(err_data, f, indent=2)

        return err_data

# Alias for backward compatibility
Phase10Profiler = LiveTerminalProfiler
