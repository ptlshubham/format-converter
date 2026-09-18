"""
Phase 6 Final Microbenchmark: Compare Phase 5 Baseline vs Phase 6 Optimized ONNX Inference.
"""

import os
import sys
import time
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "super_resolution", "realesrgan_x4plus.onnx"
)

def benchmark_phase5_vs_phase6():
    print("=" * 80)
    print("PHASE 5 BASELINE VS PHASE 6 OPTIMIZED INFERENCE BENCHMARK")
    print("=" * 80)

    # Config A: Phase 5 Baseline
    opts_a = ort.SessionOptions()
    opts_a.log_severity_level = 3
    opts_a.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess_a = ort.InferenceSession(MODEL_PATH, sess_options=opts_a, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    in_name_a = sess_a.get_inputs()[0].name
    out_name_a = sess_a.get_outputs()[0].name

    raw_tiles = [np.random.rand(128, 128, 3).astype(np.float32) for _ in range(70)]

    # Warmup
    sess_a.run([out_name_a], {in_name_a: np.transpose(raw_tiles[0], (2, 0, 1))[np.newaxis, :, :, :]})

    times_a = []
    for tile in raw_tiles:
        arr = np.transpose(tile, (2, 0, 1))[np.newaxis, :, :, :]
        t0 = time.perf_counter()
        sess_a.run([out_name_a], {in_name_a: arr})
        times_a.append((time.perf_counter() - t0) * 1000.0)

    # Config B: Phase 6 Optimized
    opts_b = ort.SessionOptions()
    opts_b.log_severity_level = 3
    opts_b.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_b = ort.InferenceSession(MODEL_PATH, sess_options=opts_b, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
    in_name_b = sess_b.get_inputs()[0].name
    out_name_b = sess_b.get_outputs()[0].name

    buf = np.empty((1, 3, 128, 128), dtype=np.float32)

    # Warmup
    buf[0, 0, :, :] = raw_tiles[0][:, :, 0]
    buf[0, 1, :, :] = raw_tiles[0][:, :, 1]
    buf[0, 2, :, :] = raw_tiles[0][:, :, 2]
    sess_b.run([out_name_b], {in_name_b: buf})

    times_b = []
    for tile in raw_tiles:
        buf[0, 0, :, :] = tile[:, :, 0]
        buf[0, 1, :, :] = tile[:, :, 1]
        buf[0, 2, :, :] = tile[:, :, 2]
        t0 = time.perf_counter()
        sess_b.run([out_name_b], {in_name_b: buf})
        times_b.append((time.perf_counter() - t0) * 1000.0)

    print(f"Phase 5 Baseline (ORT_ENABLE_BASIC + non-contiguous): Total = {sum(times_a):8.2f} ms | Mean = {np.mean(times_a):6.2f} ms | P95 = {np.percentile(times_a, 95):6.2f} ms")
    print(f"Phase 6 Optimized (ORT_ENABLE_EXTENDED + prealloc buffer): Total = {sum(times_b):8.2f} ms | Mean = {np.mean(times_b):6.2f} ms | P95 = {np.percentile(times_b, 95):6.2f} ms")
    print(f"Inference Speedup: {sum(times_a) - sum(times_b):.2f} ms ({((sum(times_a) - sum(times_b)) / sum(times_a)) * 100:.2f}% faster)")

if __name__ == "__main__":
    benchmark_phase5_vs_phase6()
