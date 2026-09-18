"""
Phase 11 — BiRefNet Performance & Execution Provider Benchmark Suite
Thorough diagnostic and benchmarking tool for BiRefNet Background Removal pipeline.

MEASURE FIRST. CHANGE SECOND.
Strictly read-only diagnostic benchmark script.
"""

import os
import sys
import gc
import time
import struct
import subprocess
import numpy as np
import cv2
from PIL import Image
from typing import Dict, Any, List, Tuple, Optional

import onnxruntime as ort
import psutil

# Suppress ORT verbose logs
if hasattr(ort, "set_default_logger_severity"):
    ort.set_default_logger_severity(3)


def get_process_rss_mb() -> float:
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss / (1024.0 * 1024.0), 2)


def get_gpu_vram_info() -> Dict[str, Any]:
    try:
        res = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,name,driver_version", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, check=True)
        line = res.stdout.strip().split('\n')[0]
        parts = [p.strip() for p in line.split(',')]
        used_mb = float(parts[0])
        total_mb = float(parts[1])
        gpu_name = parts[2]
        driver_ver = parts[3]
        return {
            "gpu_name": gpu_name,
            "driver_version": driver_ver,
            "vram_used_mb": used_mb,
            "vram_total_mb": total_mb,
            "vram_free_mb": total_mb - used_mb,
            "status": "ok"
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "gpu_name": "NVIDIA GeForce RTX 2050 (4 GB)", "vram_total_mb": 4096.0}


def preprocess_image(rgb_image: np.ndarray, target_size: int = 1024) -> Tuple[np.ndarray, Tuple[int, int, int, int, float]]:
    orig_h, orig_w = rgb_image.shape[:2]
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    interp = cv2.INTER_AREA if (new_w < orig_w or new_h < orig_h) else cv2.INTER_LINEAR
    resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=interp)

    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    padded_img = np.full((target_size, target_size, 3), (124, 116, 104), dtype=np.uint8)
    padded_img[pad_top : pad_top + new_h, pad_left : pad_left + new_w, :] = resized

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    inv_std_255 = (1.0 / (255.0 * std)).astype(np.float32)
    mean_over_std = (mean / std).astype(np.float32)

    norm_img = padded_img.astype(np.float32) * inv_std_255 - mean_over_std
    tensor = np.ascontiguousarray(np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...], dtype=np.float32)

    meta = (pad_top, pad_left, new_h, new_w, scale)
    return tensor, meta


def postprocess_logits(raw_output: np.ndarray, orig_shape: Tuple[int, int], meta: Tuple[int, int, int, int, float]) -> np.ndarray:
    orig_h, orig_w = orig_shape
    pad_top, pad_left, new_h, new_w, _ = meta

    logits = raw_output[0, 0]  # Shape (1024, 1024)
    clipped_logits = np.clip(logits, -50.0, 50.0)
    prob_1024 = 1.0 / (1.0 + np.exp(-clipped_logits))

    prob_cropped = prob_1024[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    prob_orig = cv2.resize(prob_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    prob_final = np.clip(prob_orig, 0.0, 1.0, out=prob_orig).astype(np.float32)
    return prob_final


class BiRefNetBenchmark:
    def __init__(self):
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.model_path = os.path.join(self.base_dir, "backend", "models", "segmentation", "birefnet-general", "model_fp16.onnx")
        self.test_image_path = os.path.join(self.base_dir, "test_data", "test_images", "1234.png")

        assert os.path.exists(self.model_path), f"BiRefNet model file not found at '{self.model_path}'"
        assert os.path.exists(self.test_image_path), f"Test image file not found at '{self.test_image_path}'"

        # Load RGB image into memory
        pil_img = Image.open(self.test_image_path).convert("RGB")
        self.rgb_img = np.array(pil_img)
        self.orig_h, self.orig_w = self.rgb_img.shape[:2]

        self.baseline_mask: Optional[np.ndarray] = None
        self.results: List[Dict[str, Any]] = []

    def print_section(self, title: str):
        print("\n" + "=" * 60)
        print(title)
        print("=" * 60)

    def print_terminal_report(
        self,
        request_id: str,
        image_name: str,
        provider: str,
        threads: int,
        graph_opt: str,
        load_ms: float,
        pre_ms: float,
        onnx_ms: float,
        post_ms: float,
        cleanup_ms: float,
        total_ms: float,
        rss_before: float,
        rss_peak: float,
        rss_after: float,
        gpu_name: str,
        vram_info: str,
        quality_res: str,
        actual_provider: str,
    ):
        print("\n============================================================")
        print("BIRERNET PHASE 11 BENCHMARK")
        print("============================================================")
        print(f"REQUEST ID:         {request_id}")
        print(f"IMAGE:              {image_name} ({self.orig_w}x{self.orig_h})")
        print(f"MODEL:              BiRefNet General FP16 ONNX")
        print(f"INPUT SIZE:         1024 x 1024")
        print(f"DTYPE:              float32")
        print(f"PROVIDER:           {provider} (Active: {actual_provider})")
        print(f"CPU THREADS:        {threads}")
        print(f"GRAPH OPTIMIZATION: {graph_opt}")
        print("-" * 60)
        print(f"MODEL LOAD:         {load_ms:8.2f} ms ({load_ms/1000.0:6.2f} s) [{load_ms/total_ms*100:5.1f}%]")
        print(f"PREPROCESS:         {pre_ms:8.2f} ms ({pre_ms/1000.0:6.2f} s) [{pre_ms/total_ms*100:5.1f}%]")
        print(f"INFERENCE:          {onnx_ms:8.2f} ms ({onnx_ms/1000.0:6.2f} s) [{onnx_ms/total_ms*100:5.1f}%]")
        print(f"POSTPROCESS:        {post_ms:8.2f} ms ({post_ms/1000.0:6.2f} s) [{post_ms/total_ms*100:5.1f}%]")
        print(f"CLEANUP:            {cleanup_ms:8.2f} ms ({cleanup_ms/1000.0:6.2f} s) [{cleanup_ms/total_ms*100:5.1f}%]")
        print(f"TOTAL:              {total_ms:8.2f} ms ({total_ms/1000.0:6.2f} s) [100.0%]")
        print("-" * 60)
        print(f"RSS BEFORE:         {rss_before:6.2f} MB")
        print(f"RSS PEAK:           {rss_peak:6.2f} MB (Delta: +{round(rss_peak - rss_before, 2)} MB)")
        print(f"RSS AFTER:          {rss_after:6.2f} MB")
        print(f"GPU:                {gpu_name}")
        print(f"GPU VRAM:           {vram_info}")
        print(f"QUALITY RESULT:     {quality_res}")
        print("============================================================\n")

    def run_single_benchmark(
        self,
        test_name: str,
        provider_name: str,
        providers_list: List[str],
        threads: int = 4,
        opt_level: ort.GraphOptimizationLevel = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        opt_name: str = "ORT_ENABLE_BASIC",
        existing_session: Optional[ort.InferenceSession] = None,
        request_id: str = "REQ-001"
    ) -> Tuple[Dict[str, Any], Optional[ort.InferenceSession], np.ndarray]:

        print(f"--> Starting test: {test_name} (Provider: {provider_name}, Threads: {threads}, Opt: {opt_name})...", flush=True)
        gc.collect()
        rss_before = get_process_rss_mb()
        t_start_total = time.perf_counter()

        # Step 1: Session creation / Model load
        t0_load = time.perf_counter()
        if existing_session is not None:
            session = existing_session
            load_ms = 0.0
        else:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            opts.graph_optimization_level = opt_level
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1

            session = ort.InferenceSession(self.model_path, sess_options=opts, providers=providers_list)
            load_ms = round((time.perf_counter() - t0_load) * 1000.0, 2)

        actual_providers = session.get_providers()
        actual_provider_str = actual_providers[0] if actual_providers else "Unknown"

        rss_after_load = get_process_rss_mb()

        # Step 2: Preprocess
        t0_pre = time.perf_counter()
        tensor, meta = preprocess_image(self.rgb_img, target_size=1024)
        pre_ms = round((time.perf_counter() - t0_pre) * 1000.0, 2)

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Step 3: ONNX Inference
        print(f"    Running ONNX inference for {test_name}...", flush=True)
        t0_onnx = time.perf_counter()
        raw_outputs = session.run([output_name], {input_name: tensor})
        onnx_ms = round((time.perf_counter() - t0_onnx) * 1000.0, 2)
        rss_peak = get_process_rss_mb()

        # Step 4: Postprocess
        t0_post = time.perf_counter()
        final_mask = postprocess_logits(raw_outputs[0], (self.orig_h, self.orig_w), meta)
        post_ms = round((time.perf_counter() - t0_post) * 1000.0, 2)

        # Step 5: Cleanup
        t0_clean = time.perf_counter()
        del tensor, raw_outputs
        gc.collect()
        cleanup_ms = round((time.perf_counter() - t0_clean) * 1000.0, 2)
        rss_after = get_process_rss_mb()

        total_ms = round((time.perf_counter() - t_start_total) * 1000.0, 2)

        # Quality Check against baseline
        if self.baseline_mask is None:
            self.baseline_mask = final_mask
            quality_res = "Baseline (Reference)"
        else:
            mae = float(np.mean(np.abs(self.baseline_mask - final_mask)))
            max_diff = float(np.max(np.abs(self.baseline_mask - final_mask)))
            if mae < 1e-4:
                quality_res = f"Equivalent (MAE: {mae:.6f})"
            elif mae < 1e-2:
                quality_res = f"Minor difference (MAE: {mae:.4f})"
            elif mae < 0.1:
                quality_res = f"Noticeable difference (MAE: {mae:.3f})"
            else:
                quality_res = f"Failed (MAE: {mae:.3f}, MaxDiff: {max_diff:.2f})"

        gpu_info = get_gpu_vram_info()
        vram_str = f"{gpu_info['vram_used_mb']} MB / {gpu_info['vram_total_mb']} MB" if gpu_info['status'] == 'ok' else "N/A"

        self.print_terminal_report(
            request_id=request_id,
            image_name="1234.png",
            provider=provider_name,
            threads=threads,
            graph_opt=opt_name,
            load_ms=load_ms,
            pre_ms=pre_ms,
            onnx_ms=onnx_ms,
            post_ms=post_ms,
            cleanup_ms=cleanup_ms,
            total_ms=total_ms,
            rss_before=rss_before,
            rss_peak=rss_peak,
            rss_after=rss_after,
            gpu_name=gpu_info.get("gpu_name", "NVIDIA GeForce RTX 2050"),
            vram_info=vram_str,
            quality_res=quality_res,
            actual_provider=actual_provider_str,
        )

        res_entry = {
            "test_name": test_name,
            "provider": provider_name,
            "actual_provider": actual_provider_str,
            "threads": threads,
            "optimization": opt_name,
            "load_ms": load_ms,
            "pre_ms": pre_ms,
            "inference_ms": onnx_ms,
            "post_ms": post_ms,
            "total_ms": total_ms,
            "peak_rss_mb": rss_peak,
            "final_rss_mb": rss_after,
            "quality": quality_res,
        }

        return res_entry, session, final_mask

    def run_all(self):
        print("\n============================================================")
        print("PHASE 11 — BIRERNET PERFORMANCE INVESTIGATION BENCHMARK")
        print("============================================================\n")

        # ---------------------------------------------------------------------
        # 1. Environment Check
        # ---------------------------------------------------------------------
        self.print_section("1. ENVIRONMENT DIAGNOSIS")
        print(f"Python Version:         {sys.version.split()[0]}")
        print(f"ONNX Runtime Version:   {ort.__version__}")
        print(f"Available Providers:    {ort.get_available_providers()}")
        gpu_info = get_gpu_vram_info()
        print(f"GPU Name:               {gpu_info.get('gpu_name', 'N/A')}")
        print(f"Driver Version:         {gpu_info.get('driver_version', 'N/A')}")
        print(f"VRAM Total:             {gpu_info.get('vram_total_mb', 0)} MB")

        # ---------------------------------------------------------------------
        # 2. Cold vs Warm Session Test (CPU Baseline)
        # ---------------------------------------------------------------------
        self.print_section("2. COLD VS WARM SESSION TEST (CPU 4 Threads)")

        res_cold, session_cpu, _ = self.run_single_benchmark(
            test_name="Baseline (Cold)",
            provider_name="CPUExecutionProvider",
            providers_list=["CPUExecutionProvider"],
            threads=4,
            opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            opt_name="ORT_ENABLE_BASIC",
            request_id="REQ-COLD-001"
        )
        self.results.append(res_cold)

        res_warm, _, _ = self.run_single_benchmark(
            test_name="Baseline (Warm)",
            provider_name="CPUExecutionProvider",
            providers_list=["CPUExecutionProvider"],
            threads=4,
            opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            opt_name="ORT_ENABLE_BASIC",
            existing_session=session_cpu,
            request_id="REQ-WARM-002"
        )
        self.results.append(res_warm)

        del session_cpu
        gc.collect()

        # ---------------------------------------------------------------------
        # 3. CPU Thread Count Benchmark (1, 2, 4, 8 threads)
        # ---------------------------------------------------------------------
        self.print_section("3. CPU THREAD COUNT BENCHMARK")
        for th in [1, 2, 4, 8]:
            res_th, sess, _ = self.run_single_benchmark(
                test_name=f"CPU ({th} Threads)",
                provider_name="CPUExecutionProvider",
                providers_list=["CPUExecutionProvider"],
                threads=th,
                opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                opt_name="ORT_ENABLE_BASIC",
                request_id=f"REQ-CPU-TH{th}"
            )
            self.results.append(res_th)
            del sess
            gc.collect()

        # ---------------------------------------------------------------------
        # 4. ONNX Graph Optimization Benchmark
        # ---------------------------------------------------------------------
        self.print_section("4. ONNX GRAPH OPTIMIZATION BENCHMARK")
        opts_map = [
            ("Disable", ort.GraphOptimizationLevel.ORT_DISABLE_ALL, "ORT_DISABLE_ALL"),
            ("Basic", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC, "ORT_ENABLE_BASIC"),
            ("Extended", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, "ORT_ENABLE_EXTENDED"),
            ("All", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, "ORT_ENABLE_ALL"),
        ]
        for name, level, opt_str in opts_map:
            res_opt, sess, _ = self.run_single_benchmark(
                test_name=f"Graph ({name})",
                provider_name="CPUExecutionProvider",
                providers_list=["CPUExecutionProvider"],
                threads=4,
                opt_level=level,
                opt_name=opt_str,
                request_id=f"REQ-OPT-{name.upper()}"
            )
            self.results.append(res_opt)
            del sess
            gc.collect()

        # ---------------------------------------------------------------------
        # 5. Execution Provider Test (DirectML & CUDA)
        # ---------------------------------------------------------------------
        self.print_section("5. GPU EXECUTION PROVIDER BENCHMARK (DirectML & CUDA)")
        available_p = ort.get_available_providers()

        if "DmlExecutionProvider" in available_p:
            print("Testing DmlExecutionProvider (DirectML)...")
            try:
                res_dml_cold, sess_dml, _ = self.run_single_benchmark(
                    test_name="GPU (DirectML Cold)",
                    provider_name="DmlExecutionProvider",
                    providers_list=["DmlExecutionProvider", "CPUExecutionProvider"],
                    threads=4,
                    opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                    opt_name="ORT_ENABLE_BASIC",
                    request_id="REQ-DML-COLD"
                )
                self.results.append(res_dml_cold)

                res_dml_warm, _, _ = self.run_single_benchmark(
                    test_name="GPU (DirectML Warm)",
                    provider_name="DmlExecutionProvider",
                    providers_list=["DmlExecutionProvider", "CPUExecutionProvider"],
                    threads=4,
                    opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                    opt_name="ORT_ENABLE_BASIC",
                    existing_session=sess_dml,
                    request_id="REQ-DML-WARM"
                )
                self.results.append(res_dml_warm)

                del sess_dml
                gc.collect()
            except Exception as e:
                print(f"[ERROR] DirectML execution failed: {e}")
        else:
            print("DmlExecutionProvider is NOT available in this environment.")

        if "CUDAExecutionProvider" in available_p:
            print("Testing CUDAExecutionProvider...")
            try:
                res_cuda_cold, sess_cuda, _ = self.run_single_benchmark(
                    test_name="GPU (CUDA Cold)",
                    provider_name="CUDAExecutionProvider",
                    providers_list=["CUDAExecutionProvider", "CPUExecutionProvider"],
                    threads=4,
                    opt_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
                    opt_name="ORT_ENABLE_BASIC",
                    request_id="REQ-CUDA-COLD"
                )
                self.results.append(res_cuda_cold)
                del sess_cuda
                gc.collect()
            except Exception as e:
                print(f"[ERROR] CUDA execution failed: {e}")
        else:
            print("CUDAExecutionProvider is NOT available in installed ONNX Runtime package.")

        # ---------------------------------------------------------------------
        # 6. Final Benchmark Summary Table (Section 14 Format)
        # ---------------------------------------------------------------------
        self.print_section("6. FINAL BENCHMARK SUMMARY TABLE")
        print(f"{'Test':<22} | {'Provider':<20} | {'Threads':<7} | {'Optimization':<18} | {'Load (ms)':<9} | {'Inference (ms)':<14} | {'Total (s)':<9} | {'Peak RSS':<9} | {'Quality'}")
        print("-" * 135)
        for r in self.results:
            tot_s = f"{r['total_ms']/1000.0:6.2f}s"
            peak_mb = f"{r['peak_rss_mb']:6.1f}MB"
            print(f"{r['test_name']:<22} | {r['actual_provider']:<20} | {r['threads']:<7} | {r['optimization']:<18} | {r['load_ms']:<9.1f} | {r['inference_ms']:<14.1f} | {tot_s:<9} | {peak_mb:<9} | {r['quality']}")

        print("\n============================================================")
        print("BENCHMARK INVESTIGATION COMPLETE")
        print("============================================================\n")


if __name__ == "__main__":
    bm = BiRefNetBenchmark()
    bm.run_all()
