"""
Test BiRefNet ONNX execution with mmap memory configuration.
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

def test_mmap():
    print("=" * 80)
    print("TESTING BIREFNET WITH MMAP SESSION CONFIG")
    print("=" * 80)

    dummy_input = np.random.rand(1, 3, 1024, 1024).astype(np.float32)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.add_session_config_entry("session.use_mmap", "1")

    # Try DmlExecutionProvider
    try:
        print("Testing DmlExecutionProvider with mmap...")
        t0 = time.perf_counter()
        sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["DmlExecutionProvider", "CPUExecutionProvider"])
        dt_load = (time.perf_counter() - t0) * 1000.0
        print(f"Session Loaded in {dt_load:.2f} ms")

        in_name = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        t0 = time.perf_counter()
        res = sess.run([out_name], {in_name: dummy_input})[0]
        dt_run = (time.perf_counter() - t0) * 1000.0
        print(f"DirectML Inference SUCCESS: Output shape {res.shape} in {dt_run:.2f} ms")
    except Exception as e:
        print(f"DirectML FAILED: {e}")

    # Try CPUExecutionProvider
    try:
        print("\nTesting CPUExecutionProvider with mmap...")
        t0 = time.perf_counter()
        sess_cpu = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
        dt_load = (time.perf_counter() - t0) * 1000.0
        print(f"Session Loaded in {dt_load:.2f} ms")

        in_name = sess_cpu.get_inputs()[0].name
        out_name = sess_cpu.get_outputs()[0].name

        t0 = time.perf_counter()
        res = sess_cpu.run([out_name], {in_name: dummy_input})[0]
        dt_run = (time.perf_counter() - t0) * 1000.0
        print(f"CPU Inference SUCCESS: Output shape {res.shape} in {dt_run:.2f} ms")
    except Exception as e:
        print(f"CPU FAILED: {e}")

if __name__ == "__main__":
    test_mmap()
