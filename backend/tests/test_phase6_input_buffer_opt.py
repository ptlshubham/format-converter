"""
Test memory buffer contiguous layout optimization for ONNX DirectML inputs.
Compares:
1. Non-contiguous transpose array view (current Phase 5)
2. np.ascontiguousarray float32
3. Pre-allocated C-contiguous float32 NCHW buffer (in-place copy)
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

def test_input_memory_layout():
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    
    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=opts,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    # Generate 70 dummy 128x128 HWC float32 tiles
    raw_tiles = [np.random.rand(128, 128, 3).astype(np.float32) for _ in range(70)]
    
    # 1. Non-contiguous transpose
    times_non_contig = []
    for tile in raw_tiles:
        arr = np.transpose(tile, (2, 0, 1))[np.newaxis, :, :, :]
        t0 = time.perf_counter()
        session.run([output_name], {input_name: arr})
        times_non_contig.append((time.perf_counter() - t0) * 1000.0)
        
    # 2. Contiguous array
    times_contig = []
    for tile in raw_tiles:
        arr = np.ascontiguousarray(np.transpose(tile, (2, 0, 1))[np.newaxis, :, :, :], dtype=np.float32)
        t0 = time.perf_counter()
        session.run([output_name], {input_name: arr})
        times_contig.append((time.perf_counter() - t0) * 1000.0)

    # 3. Pre-allocated buffer in-place copy
    buf = np.empty((1, 3, 128, 128), dtype=np.float32)
    times_prealloc = []
    for tile in raw_tiles:
        buf[0, 0, :, :] = tile[:, :, 0]
        buf[0, 1, :, :] = tile[:, :, 1]
        buf[0, 2, :, :] = tile[:, :, 2]
        t0 = time.perf_counter()
        session.run([output_name], {input_name: buf})
        times_prealloc.append((time.perf_counter() - t0) * 1000.0)

    print(f"Non-contiguous mean inference: {np.mean(times_non_contig):.2f} ms | Total: {sum(times_non_contig):.2f} ms")
    print(f"Contiguous array mean inference: {np.mean(times_contig):.2f} ms | Total: {sum(times_contig):.2f} ms")
    print(f"Pre-allocated buffer mean inference: {np.mean(times_prealloc):.2f} ms | Total: {sum(times_prealloc):.2f} ms")

if __name__ == "__main__":
    test_input_memory_layout()
