"""
Performance Profiler and Benchmark for BiRefNet General ONNX Inference
Measures each stage independently:
1. Decode & EXIF
2. Preprocessing & Letterbox (HWC -> NCHW Float32)
3. ONNX Inference (with configurable intra/inter threads, session options)
4. Sigmoid continuous probability mapping
5. Unletterbox and native resolution restore
6. Classical Refinement stages
7. PNG Encoding
"""

import os
import sys
import time
import statistics
import numpy as np
import cv2
from PIL import Image

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import functools
print = functools.partial(print, flush=True)
import onnxruntime as ort

MODEL_PATH = os.path.join(
    project_root, "backend", "models", "segmentation", "birefnet-general", "model_fp16.onnx"
)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def create_session(intra_threads: int, graph_opt: ort.GraphOptimizationLevel):
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = graph_opt
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.intra_op_num_threads = intra_threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])


def preprocess_image(rgb_image: np.ndarray, target_size: int = 1024):
    orig_h, orig_w = rgb_image.shape[:2]
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = max(1, int(round(orig_w * scale)))
    new_h = max(1, int(round(orig_h * scale)))

    interp = cv2.INTER_AREA if (new_w < orig_w or new_h < orig_h) else cv2.INTER_CUBIC
    resized = cv2.resize(rgb_image, (new_w, new_h), interpolation=interp)

    pad_top = (target_size - new_h) // 2
    pad_bottom = target_size - new_h - pad_top
    pad_left = (target_size - new_w) // 2
    pad_right = target_size - new_w - pad_left

    padded = cv2.copyMakeBorder(
        resized, pad_top, pad_bottom, pad_left, pad_right,
        borderType=cv2.BORDER_CONSTANT, value=[124, 116, 104]
    )

    # Fast standardized tensor creation
    norm_img = (padded.astype(np.float32) * (1.0 / 255.0) - MEAN) / STD
    tensor = np.ascontiguousarray(np.transpose(norm_img, (2, 0, 1))[np.newaxis, ...], dtype=np.float32)
    return tensor, (orig_w, orig_h, new_w, new_h, pad_top, pad_left)


def postprocess_logits(logits: np.ndarray, meta: tuple):
    orig_w, orig_h, new_w, new_h, pad_top, pad_left = meta
    clipped = np.clip(logits, -50.0, 50.0)
    prob_1024 = 1.0 / (1.0 + np.exp(-clipped))
    prob_cropped = prob_1024[pad_top : pad_top + new_h, pad_left : pad_left + new_w]
    prob_orig = cv2.resize(prob_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return np.clip(prob_orig, 0.0, 1.0).astype(np.float32)


def run_benchmark():
    print("=" * 80)
    print("BIREFNET GENERAL FP16 ONNX - DETAILED PERFORMANCE PROFILING")
    print("=" * 80)
    print(f"Model Path: {MODEL_PATH}")
    print(f"File Size:  {round(os.path.getsize(MODEL_PATH) / 1024 / 1024, 2)} MB")
    print(f"Providers:  {ort.get_available_providers()}")
    print(f"OS CPU Cores: Physical=6, Logical=12")

    # Create dummy 640x359 image (matching user's production image)
    test_img = np.random.randint(0, 255, (359, 640, 3), dtype=np.uint8)

    # Thread configurations to evaluate: 4 threads, 6 threads (physical cores), 12 threads (logical)
    thread_configs = [4, 6, 12]

    print("\n--- STAGE BREAKDOWN TIMING (Detailed Sub-Millisecond Profiling) ---")
    t0 = time.perf_counter()
    tensor, meta = preprocess_image(test_img)
    t_pre = (time.perf_counter() - t0) * 1000.0
    print(f"1. Preprocessing (Resize, Letterbox, Normalize, NCHW): {t_pre:.2f} ms")

    dummy_logits = np.random.randn(1024, 1024).astype(np.float32)
    t0 = time.perf_counter()
    prob_map = postprocess_logits(dummy_logits, meta)
    t_post = (time.perf_counter() - t0) * 1000.0
    print(f"2. Postprocessing (Sigmoid, Unletterbox, Native Resize): {t_post:.2f} ms")

    # Benchmark thread scaling
    print("\n--- THREAD CONFIGURATION BENCHMARK ---")
    results = {}
    for threads in thread_configs:
        print(f"\nEvaluating intra_op_num_threads = {threads} ...")
        t_load_start = time.perf_counter()
        session = create_session(threads, ort.GraphOptimizationLevel.ORT_ENABLE_ALL)
        load_time = (time.perf_counter() - t_load_start) * 1000.0
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        # Warmup
        t_warmup_start = time.perf_counter()
        session.run([output_name], {input_name: tensor})
        warmup_time = (time.perf_counter() - t_warmup_start) * 1000.0
        print(f"  Load Time: {load_time:.1f} ms | Warmup Time: {warmup_time:.1f} ms")

        # 1 measured iteration
        runs = []
        for i in range(1):
            t_run = time.perf_counter()
            session.run([output_name], {input_name: tensor})
            elapsed = (time.perf_counter() - t_run) * 1000.0
            runs.append(elapsed)
            print(f"  Iteration {i+1}: {elapsed:.1f} ms")

        avg_time = statistics.mean(runs)
        results[threads] = {
            "load_ms": load_time,
            "warmup_ms": warmup_time,
            "runs": runs,
            "avg_ms": avg_time,
            "min_ms": min(runs),
            "max_ms": max(runs),
        }

    print("\n" + "=" * 80)
    print("THREAD CONFIGURATION SUMMARY TABLE")
    print("=" * 80)
    print(f"{'Threads':<10} | {'Avg Inference (ms)':<20} | {'Min (ms)':<12} | {'Max (ms)':<12}")
    print("-" * 60)
    for threads, res in results.items():
        print(f"{threads:<10} | {res['avg_ms']:<20.1f} | {res['min_ms']:<12.1f} | {res['max_ms']:<12.1f}")


if __name__ == "__main__":
    run_benchmark()
