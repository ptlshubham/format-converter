"""
Test ONNX Runtime Session Options and DirectML Provider Options for Real-ESRGAN.
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

def test_options():
    print("=" * 80)
    print("TESTING SESSION AND PROVIDER OPTIONS")
    print("=" * 80)

    configs = [
        ("Base (EXTENDED)", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, None, True),
        ("ENABLE_ALL", ort.GraphOptimizationLevel.ORT_ENABLE_ALL, None, True),
        ("EXTENDED + MemPattern True", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, True, True),
        ("EXTENDED + MemPattern False", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, False, True),
        ("EXTENDED + CPU Arena False", ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED, True, False),
    ]

    tile_buf = np.random.rand(1, 3, 128, 128).astype(np.float32)

    for name, opt_level, mem_pattern, cpu_arena in configs:
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = opt_level
        if mem_pattern is not None:
            opts.enable_mem_pattern = mem_pattern
        opts.enable_cpu_mem_arena = cpu_arena

        providers = [
            ("DmlExecutionProvider", {"device_id": 0}),
            "CPUExecutionProvider"
        ]

        try:
            t_init0 = time.perf_counter()
            session = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=providers)
            t_init = (time.perf_counter() - t_init0) * 1000.0

            input_name = session.get_inputs()[0].name
            output_name = session.get_outputs()[0].name

            # Warmup
            session.run([output_name], {input_name: tile_buf})

            # 70 calls
            times = []
            for _ in range(70):
                t0 = time.perf_counter()
                session.run([output_name], {input_name: tile_buf})
                times.append((time.perf_counter() - t0) * 1000.0)

            print(f"Config: {name:30s} | Init: {t_init:6.2f} ms | Total 70: {sum(times):8.2f} ms | Mean Tile: {np.mean(times):6.2f} ms")
        except Exception as e:
            print(f"Config: {name:30s} FAILED: {e}")

if __name__ == "__main__":
    test_options()
