"""
Test SessionOptions settings for BiRefNet ONNX on CPU execution.
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
    "models", "segmentation", "birefnet-general", "model_fp16.onnx"
)

def test_configs():
    print("=" * 80)
    print("TESTING BIREFNET SESSION OPTIONS ON CPU")
    print("=" * 80)

    dummy_input = np.random.rand(1, 3, 1024, 1024).astype(np.float32)

    configs = [
        ("DISABLE_ALL", ort.GraphOptimizationLevel.ORT_DISABLE_ALL),
        ("BASIC", ort.GraphOptimizationLevel.ORT_ENABLE_BASIC),
    ]

    for name, opt_level in configs:
        print(f"\n--- Testing Config: {name} ---")
        try:
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            opts.graph_optimization_level = opt_level
            opts.intra_op_num_threads = 4
            opts.inter_op_num_threads = 1
            opts.enable_cpu_mem_arena = False
            opts.enable_mem_pattern = False

            t0 = time.perf_counter()
            sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
            dt_load = (time.perf_counter() - t0) * 1000.0
            print(f"[{name}] Session Loaded in {dt_load:.2f} ms")

            in_name = sess.get_inputs()[0].name
            out_name = sess.get_outputs()[0].name

            t0 = time.perf_counter()
            res = sess.run([out_name], {in_name: dummy_input})[0]
            dt_run = (time.perf_counter() - t0) * 1000.0
            print(f"[{name}] SUCCESS: Output shape {res.shape} in {dt_run:.2f} ms ({dt_run/1000.0:.2f} s)")
        except Exception as e:
            print(f"[{name}] FAILED: {e}")

if __name__ == "__main__":
    test_configs()
