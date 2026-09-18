"""
Test thread counts (1, 2, 4) and memory arena settings for BiRefNet CPU execution.
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

def test_thread_options():
    print("=" * 80)
    print("TESTING BIREFNET CPU THREAD COUNTS AND MEMORY SETTINGS")
    print("=" * 80)

    dummy_input = np.random.rand(1, 3, 1024, 1024).astype(np.float32)

    for threads in [1, 2, 4]:
        for arena in [False, True]:
            print(f"\n--- Testing Threads: {threads}, MemArena: {arena} ---")
            try:
                opts = ort.SessionOptions()
                opts.log_severity_level = 3
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                opts.intra_op_num_threads = threads
                opts.inter_op_num_threads = 1
                opts.enable_cpu_mem_arena = arena
                opts.enable_mem_pattern = arena

                t0 = time.perf_counter()
                sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
                dt_load = (time.perf_counter() - t0) * 1000.0
                print(f"[Threads: {threads}, Arena: {arena}] Loaded in {dt_load:.2f} ms")

                in_name = sess.get_inputs()[0].name
                out_name = sess.get_outputs()[0].name

                t0 = time.perf_counter()
                res = sess.run([out_name], {in_name: dummy_input})[0]
                dt_run = (time.perf_counter() - t0) * 1000.0
                print(f"[Threads: {threads}, Arena: {arena}] SUCCESS! Inference time: {dt_run:.2f} ms ({dt_run/1000.0:.2f} s)")
                return  # Stop after first successful configuration!
            except Exception as e:
                print(f"[Threads: {threads}, Arena: {arena}] FAILED: {e}")

if __name__ == "__main__":
    test_thread_options()
