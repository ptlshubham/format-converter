"""
Test passing float16 input tensor to BiRefNet model_fp16.onnx.
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

def test_fp16_input():
    print("=" * 80)
    print("TESTING FP16 INPUT TENSOR TO BIREFNET")
    print("=" * 80)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.enable_cpu_mem_arena = True

    t0 = time.perf_counter()
    sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    dt_load = (time.perf_counter() - t0) * 1000.0
    print(f"Session Loaded in {dt_load:.2f} ms")

    inp = sess.get_inputs()[0]
    outp = sess.get_outputs()[0]
    print(f"ONNX Model Input Name: {inp.name}, Type: {inp.type}, Shape: {inp.shape}")
    print(f"ONNX Model Output Name: {outp.name}, Type: {outp.type}, Shape: {outp.shape}")

    # Pass float16 input matching input type tensor(float16)
    if "float16" in inp.type:
        dummy_input = np.random.rand(1, 3, 1024, 1024).astype(np.float16)
        print("\nPassing float16 input tensor...")
    else:
        dummy_input = np.random.rand(1, 3, 1024, 1024).astype(np.float32)
        print("\nPassing float32 input tensor...")

    t0 = time.perf_counter()
    res = sess.run([outp.name], {inp.name: dummy_input})[0]
    dt_run = (time.perf_counter() - t0) * 1000.0
    print(f"BiRefNet Inference SUCCESS! Output shape: {res.shape}, dtype: {res.dtype} in {dt_run:.2f} ms ({dt_run/1000.0:.2f} s)")

if __name__ == "__main__":
    test_fp16_input()
