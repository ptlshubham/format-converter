"""
Isolated Operator-Level Profiling Script for BiRefNet General FP16 ONNX
Must only be executed when NO OTHER ONNX session is running.
Runs 1 inference pass with ONNX Runtime profiling enabled, extracts the slowest operators,
cleans up the temporary trace JSON file, and exits.
"""

import os
import sys
import glob
import json
import time
import numpy as np

# Ensure root directory is on sys.path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

import onnxruntime as ort

MODEL_PATH = os.path.join(PROJECT_DIR, "backend", "models", "segmentation", "birefnet-general", "model_fp16.onnx")


def run_operator_profile():
    print("[OPERATOR PROFILE] Starting isolated ONNX operator profiling...")
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        sys.exit(1)

    trace_prefix = os.path.join(PROJECT_DIR, "backend", "tests", "ort_profile_trace")

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.intra_op_num_threads = 4
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    opts.enable_cpu_mem_arena = True
    opts.enable_mem_pattern = True
    opts.enable_profiling = True
    opts.profile_file_prefix = trace_prefix

    print("[OPERATOR PROFILE] Creating profiling session (CPU, 4 threads)...")
    sess = ort.InferenceSession(MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"])

    input_name = sess.get_inputs()[0].name
    output_name = sess.get_outputs()[0].name

    # Create dummy standard 1024x1024 float32 input tensor
    dummy_input = np.zeros((1, 3, 1024, 1024), dtype=np.float32)

    print("[OPERATOR PROFILE] Running inference with operator trace...")
    t0 = time.perf_counter()
    _ = sess.run([output_name], {input_name: dummy_input})
    inference_ms = (time.perf_counter() - t0) * 1000.0
    print(f"[OPERATOR PROFILE] Inference completed in {inference_ms:.2f} ms")

    profile_file = sess.end_profiling()
    print(f"[OPERATOR PROFILE] Trace saved to: {profile_file}")

    # Parse trace JSON
    op_durations = {}
    op_counts = {}
    node_durations = []

    try:
        with open(profile_file, "r", encoding="utf-8") as f:
            events = json.load(f)

        for ev in events:
            if ev.get("cat") == "Node":
                dur_us = ev.get("dur", 0)
                name = ev.get("name", "")
                args = ev.get("args", {})
                op_type = args.get("op_name", name.split("_")[0] if "_" in name else name)

                op_durations[op_type] = op_durations.get(op_type, 0) + dur_us
                op_counts[op_type] = op_counts.get(op_type, 0) + 1
                node_durations.append({
                    "name": name,
                    "op_type": op_type,
                    "dur_ms": dur_us / 1000.0,
                })

        # Sort aggregated op types
        sorted_ops = sorted(op_durations.items(), key=lambda x: x[1], reverse=True)
        total_op_dur_us = sum(op_durations.values()) or 1.0

        # Sort top nodes
        sorted_nodes = sorted(node_durations, key=lambda x: x["dur_ms"], reverse=True)[:15]

        result_summary = {
            "total_inference_ms": round(inference_ms, 2),
            "total_operator_time_ms": round(total_op_dur_us / 1000.0, 2),
            "top_operator_types": [
                {
                    "op_type": op,
                    "total_ms": round(dur / 1000.0, 2),
                    "pct": round((dur / total_op_dur_us) * 100.0, 2),
                    "count": op_counts.get(op, 0),
                }
                for op, dur in sorted_ops[:10]
            ],
            "top_slowest_nodes": sorted_nodes,
        }

        # Save parsed summary
        summary_path = os.path.join(PROJECT_DIR, "backend", "tests", "operator_profile_summary.json")
        with open(summary_path, "w", encoding="utf-8") as sf:
            json.dump(result_summary, sf, indent=2)

        print("[OPERATOR PROFILE] Top Operator Types by Time:")
        for item in result_summary["top_operator_types"]:
            print(f"  {item['op_type']:<15} {item['total_ms']:>10.2f} ms ({item['pct']:>5.1f}%) | count: {item['count']}")

    finally:
        # Delete trace file to prevent leaving large files
        if os.path.exists(profile_file):
            try:
                os.remove(profile_file)
                print(f"[OPERATOR PROFILE] Cleaned up temporary trace file: {profile_file}")
            except Exception as e:
                print(f"Warning: could not delete trace file: {e}")


if __name__ == "__main__":
    run_operator_profile()
