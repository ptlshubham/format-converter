"""
Phase 6: Real-ESRGAN ONNX Inference Optimization & Diagnostic Suite
Investigates:
1. ONNX Model Metadata (inputs, outputs, static vs dynamic dims, ops)
2. Dynamic Batching Support ([N, 3, 128, 128])
3. Graph Optimization Levels (ORT_ENABLE_BASIC vs ORT_ENABLE_EXTENDED vs ORT_ENABLE_ALL)
4. Session Options & DirectML Provider Options
5. Warm-up vs Steady-State Inference Timing
6. Tile Memory Layout & Input Preparation Overhead
7. Full 70-tile Performance & Quality Benchmarking
"""

import os
import sys
import time
import numpy as np
import cv2

try:
    import onnxruntime as ort
except ImportError:
    ort = None

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "super_resolution", "realesrgan_x4plus.onnx"
)

def inspect_model_metadata():
    print("=" * 80)
    print("TASK 5: INSPECT ONNX MODEL METADATA")
    print("=" * 80)
    
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        return
        
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    
    print(f"Model Path: {MODEL_PATH}")
    
    # Inputs
    inputs = session.get_inputs()
    print("\n[INPUTS]")
    for inp in inputs:
        print(f"  Name: {inp.name}")
        print(f"  Shape: {inp.shape}")
        print(f"  Type: {inp.type}")
        
    # Outputs
    outputs = session.get_outputs()
    print("\n[OUTPUTS]")
    for out in outputs:
        print(f"  Name: {out.name}")
        print(f"  Shape: {out.shape}")
        print(f"  Type: {out.type}")
        
    # Provider
    print(f"\nAvailable Providers in ORT: {ort.get_available_providers()}")


def test_batching():
    print("\n" + "=" * 80)
    print("TASK 6: BATCHING INVESTIGATION")
    print("=" * 80)
    
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    for batch_size in [1, 2, 4, 8]:
        dummy_input = np.zeros((batch_size, 3, 128, 128), dtype=np.float32)
        try:
            t0 = time.perf_counter()
            res = session.run([output_name], {input_name: dummy_input})[0]
            dt = (time.perf_counter() - t0) * 1000.0
            print(f"Batch size {batch_size}: SUCCESS -> Output shape {res.shape} in {dt:.2f} ms")
        except Exception as e:
            print(f"Batch size {batch_size}: FAILED -> {e}")


def test_graph_optimization_levels():
    print("\n" + "=" * 80)
    print("TASK 17: ONNX GRAPH OPTIMIZATION LEVELS")
    print("=" * 80)
    
    opt_levels = {
        "ORT_DISABLE_ALL": ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "ORT_ENABLE_BASIC": ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "ORT_ENABLE_EXTENDED": ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "ORT_ENABLE_ALL": ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    
    dummy_input = np.random.rand(1, 3, 128, 128).astype(np.float32)
    
    for name, level in opt_levels.items():
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        opts.graph_optimization_level = level
        
        t_load0 = time.perf_counter()
        session = ort.InferenceSession(
            MODEL_PATH,
            sess_options=opts,
            providers=["DmlExecutionProvider", "CPUExecutionProvider"]
        )
        t_load = (time.perf_counter() - t_load0) * 1000.0
        
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        
        # Warmup (1 call)
        session.run([output_name], {input_name: dummy_input})
        
        # Benchmark 10 calls
        times = []
        for _ in range(10):
            t0 = time.perf_counter()
            session.run([output_name], {input_name: dummy_input})
            times.append((time.perf_counter() - t0) * 1000.0)
            
        mean_time = np.mean(times)
        print(f"Level: {name:20s} | Session Init: {t_load:7.2f} ms | Mean Tile (10 calls): {mean_time:6.2f} ms")


def test_warmup_and_steady_state():
    print("\n" + "=" * 80)
    print("TASK 10 & 11: WARM-UP VS STEADY-STATE INFERENCE")
    print("=" * 80)
    
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    session = ort.InferenceSession(
        MODEL_PATH,
        sess_options=opts,
        providers=["DmlExecutionProvider", "CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    dummy_input = np.random.rand(1, 3, 128, 128).astype(np.float32)
    
    print(f"Active Provider: {session.get_providers()[0]}")
    
    runs = 4
    for r in range(1, runs + 1):
        times = []
        for i in range(70):
            t0 = time.perf_counter()
            session.run([output_name], {input_name: dummy_input})
            times.append((time.perf_counter() - t0) * 1000.0)
            
        t_first = times[0]
        t_rem_mean = np.mean(times[1:])
        t_total = sum(times)
        print(f"Run {r}: Total 70-tile = {t_total:8.2f} ms | First tile = {t_first:6.2f} ms | Remaining mean = {t_rem_mean:6.2f} ms")


if __name__ == "__main__":
    inspect_model_metadata()
    test_batching()
    test_graph_optimization_levels()
    test_warmup_and_steady_state()
