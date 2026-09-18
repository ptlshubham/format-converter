"""
Thorough BiRefNet FP16 to FP32 Conversion
Uses onnx shape inference to correctly resolve ALL intermediate tensor types.
"""

import os
import sys
import time
import numpy as np

MODEL_DIR = r"C:\Users\ompat\OneDrive\Desktop\JPG to WEBP\backend\models\segmentation\birefnet-general"
FP16_PATH = os.path.join(MODEL_DIR, "model_fp16.onnx")
FP32_PATH = os.path.join(MODEL_DIR, "model_fp32.onnx")


def convert_fp16_to_fp32():
    import onnx
    from onnx import numpy_helper, TensorProto, helper

    print("Loading FP16 model: %s" % FP16_PATH)
    print("  Size: %.1f MB" % (os.path.getsize(FP16_PATH) / (1024*1024)))

    t0 = time.perf_counter()
    model = onnx.load(FP16_PATH)
    print("  Loaded in %.0f ms" % ((time.perf_counter() - t0) * 1000))

    # 1. Convert all FP16 initializers (weights/biases) to FP32
    init_count = 0
    for init in model.graph.initializer:
        if init.data_type == TensorProto.FLOAT16:
            arr = numpy_helper.to_array(init).astype(np.float32)
            new_t = numpy_helper.from_array(arr, name=init.name)
            init.CopyFrom(new_t)
            init_count += 1
    print("  Converted %d initializer tensors" % init_count)

    # 2. Convert all graph inputs from FP16 to FP32
    for inp in model.graph.input:
        if inp.type.tensor_type.elem_type == TensorProto.FLOAT16:
            inp.type.tensor_type.elem_type = TensorProto.FLOAT

    # 3. Convert all graph outputs from FP16 to FP32
    for out in model.graph.output:
        if out.type.tensor_type.elem_type == TensorProto.FLOAT16:
            out.type.tensor_type.elem_type = TensorProto.FLOAT

    # 4. Remove ALL value_info (intermediate type annotations)
    #    Shape inference will regenerate them correctly as FP32
    while len(model.graph.value_info) > 0:
        model.graph.value_info.pop()
    print("  Cleared all intermediate value_info entries")

    # 5. Convert Cast nodes that cast TO float16 -> cast to float32 instead
    #    Also remove redundant Cast nodes (float32 -> float32)
    cast_fixed = 0
    for node in model.graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT16:
                    attr.i = TensorProto.FLOAT
                    cast_fixed += 1
    print("  Fixed %d Cast-to-FP16 nodes" % cast_fixed)

    # 6. Fix any Constant nodes that produce FP16 tensors
    const_fixed = 0
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.t.data_type == TensorProto.FLOAT16:
                    arr = numpy_helper.to_array(attr.t).astype(np.float32)
                    new_t = numpy_helper.from_array(arr)
                    attr.t.CopyFrom(new_t)
                    const_fixed += 1
    print("  Fixed %d Constant FP16 nodes" % const_fixed)

    # 7. Run shape inference to regenerate all intermediate types as FP32
    print("  Running shape inference (this may take a moment)...")
    t0 = time.perf_counter()
    try:
        model = onnx.shape_inference.infer_shapes(model, check_type=False, strict_mode=False)
        print("  Shape inference complete in %.0f ms" % ((time.perf_counter() - t0) * 1000))
    except Exception as e:
        print("  Shape inference warning (proceeding anyway): %s" % str(e)[:200])

    # 8. Save the FP32 model
    print("  Saving FP32 model...")
    t0 = time.perf_counter()
    onnx.save(model, FP32_PATH)
    print("  Saved to: %s" % FP32_PATH)
    print("  FP32 size: %.1f MB" % (os.path.getsize(FP32_PATH) / (1024*1024)))
    print("  Saved in %.0f ms" % ((time.perf_counter() - t0) * 1000))


def benchmark_model(model_path, label, threads, num_iterations=2):
    import onnxruntime as ort

    print("")
    print("=" * 60)
    print("BENCHMARK: %s" % label)
    print("  Model: %s (%.1f MB)" % (os.path.basename(model_path), os.path.getsize(model_path)/(1024*1024)))
    print("  Threads: %d" % threads)
    print("=" * 60)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.enable_cpu_mem_arena = False
    opts.enable_mem_pattern = False
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1

    print("  Loading session...")
    t0 = time.perf_counter()
    session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    load_ms = (time.perf_counter() - t0) * 1000
    print("  Session loaded in %.0f ms" % load_ms)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    dummy = np.zeros((1, 3, 1024, 1024), dtype=np.float32)

    print("  Running warmup pass...")
    t0 = time.perf_counter()
    session.run([output_name], {input_name: dummy})
    warmup_ms = (time.perf_counter() - t0) * 1000
    print("  Warmup: %.0f ms" % warmup_ms)

    times = []
    for i in range(num_iterations):
        t0 = time.perf_counter()
        session.run([output_name], {input_name: dummy})
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        print("  Iteration %d: %.0f ms" % (i+1, elapsed))

    avg = sum(times) / len(times)
    print("  AVERAGE: %.0f ms" % avg)

    del session
    return {"load_ms": load_ms, "warmup_ms": warmup_ms, "avg_ms": avg, "min_ms": min(times), "max_ms": max(times)}


def main():
    threads = os.cpu_count() or 12
    print("CPU: AMD Ryzen 5 7535HS | Threads: %d" % threads)

    # Step 1: Convert
    if not os.path.exists(FP32_PATH):
        print("\n--- CONVERTING FP16 -> FP32 ---")
        convert_fp16_to_fp32()
    else:
        print("FP32 model already exists (%.1f MB)" % (os.path.getsize(FP32_PATH)/(1024*1024)))

    # Step 2: Benchmark FP32
    fp32 = benchmark_model(FP32_PATH, "BiRefNet General FP32 (native CPU)", threads)

    # Step 3: Benchmark FP16
    fp16 = benchmark_model(FP16_PATH, "BiRefNet General FP16 (emulated)", threads)

    # Comparison
    print("\n" + "=" * 60)
    print("FINAL COMPARISON")
    print("=" * 60)
    print("  %-15s %10s %10s %10s" % ("", "FP16", "FP32", "Speedup"))
    for label, k in [("Load", "load_ms"), ("Warmup", "warmup_ms"), ("Inference", "avg_ms")]:
        ratio = fp16[k] / max(fp32[k], 1)
        print("  %-15s %9.0fms %9.0fms %9.1fx" % (label, fp16[k], fp32[k], ratio))

    if fp32["avg_ms"] < fp16["avg_ms"]:
        pct = ((fp16["avg_ms"] - fp32["avg_ms"]) / fp16["avg_ms"]) * 100
        print("\n  >>> FP32 is %.0f%% FASTER on this CPU <<<" % pct)
    print("=" * 60)


if __name__ == "__main__":
    main()
