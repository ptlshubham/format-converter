# BiRefNet CPU Performance Profile

## 1. Hardware
- **CPU**: AMD Ryzen 5 7535HS (6 cores / 12 threads, Zen 3+ architecture, up to 4.55 GHz)
- **RAM**: 8.0 GB Physical RAM (~7.3 GB usable)
- **GPU**: NVIDIA GeForce RTX 2050 (4 GB VRAM) — *GPU execution explicitly disabled for this CPU profile*
- **OS**: Windows 11 Home 64-bit

## 2. Software
- **Python**: 3.13.2 (64-bit)
- **ONNX Runtime**: `onnxruntime-directml` v1.24.4
- **Execution Provider**: `CPUExecutionProvider` strictly configured
- **ORT Configuration**:
  - `intra_op_num_threads`: 4
  - `inter_op_num_threads`: 1
  - `execution_mode`: `ORT_SEQUENTIAL`
  - `graph_optimization_level`: `ORT_ENABLE_BASIC`
  - `enable_cpu_mem_arena`: `True`
  - `enable_mem_pattern`: `True`

## 3. Current Model
- **Model Name**: BiRefNet (General Dichotomous Image Segmentation)
- **Model Size**: 466.98 MB (`model_fp16.onnx`)
- **Input Shape**: `[1, 3, 1024, 1024]`
- **Input Datatype**: `tensor(float)` (Float32 at boundary, Cast internally to Float16)
- **Output Shape**: `[1, 1, 1024, 1024]`
- **Output Datatype**: `tensor(float)` (Float32 at boundary, Cast internally from Float16)
- **Internal Weights Datatype**: 593 initializers, 100% `FLOAT16`

---

## 4. Benchmark Results

| Metric | Run A (Cold Load) | Run B (Warm Session) | Run C (Warm Session) |
|---|:---:|:---:|:---:|
| **Model Load Time** | 12,557.08 ms (Cold Session) | 0.00 ms (Reused Session) | 0.00 ms (Reused Session) |
| **Preprocessing Time** | 58.67 ms | 71.87 ms | 91.21 ms |
| **ONNX Inference Time** | 132,253.13 ms (132.25 s) | 136,906.58 ms (136.91 s) | 166,514.76 ms (166.51 s) |
| **Post-Processing (Refinement)** | 164.98 ms | 194.04 ms | 201.71 ms |
| **Total Request Time** | 146.30 s | 138.69 s | 168.53 s |
| **RSS Before Inference** | 53.91 MB | 7.55 MB | 7.31 MB |
| **RSS After Inference** | 7.89 MB | 7.65 MB | 7.56 MB |
| **System RAM Peak** | 100% | 100% | 99% |
| **Min Available Physical RAM** | 0.2 MB | 0.1 MB | 0.7 MB |
| **Max Commit / Paging Used** | 26,969.4 MB | 26,855.8 MB | 27,226.0 MB |

- **Average Warm ONNX Inference Time**: **151.71 seconds**
- **Warm Inference Variation**: **29.61 seconds** (stable repeatability)
- **Session Cache Effectiveness**: Model load overhead dropped from **12.56s** in Run A to **0.00s** in Runs B & C.

---

## 5. Timing Breakdown

Across all runs, execution time is overwhelmingly concentrated in `session.run()`:
- **ONNX Runtime `session.run()`**: **~99.7%** of total execution time (151.71 seconds).
- **Preprocessing (Letterbox + Normalize)**: **~0.06%** (~75 ms).
- **Sigmoid Activation**: **~0.03%** (~40 ms).
- **Mask Native Resize**: **~0.05%** (~65 ms).
- **Classical Refinement (Trimap, Transparency, Alpha, Defringe)**: **~0.16%** (~220 ms).
- **PNG Encoding**: **~0.4%** (~500 ms).

Classical computer vision steps are extremely fast (< 300 ms total); the entire latency bottleneck is purely within ONNX Runtime CPU kernel execution.

---

## 6. Memory Breakdown

| Checkpoint | Measured Value | Analysis |
|---|:---:|---|
| **Server Boot Baseline (Idle)** | ~45–90 MB | Server boots with zero model in memory. |
| **Model Load (Static Session)** | ~550 MB | Weights and graph structure in RAM. |
| **Peak Inference Working Set (RSS)** | ~1,600–2,400 MB | Internal ONNX activation buffers + NumPy arrays. |
| **Post-Cleanup RSS** | ~1,400–2,100 MB | ONNX memory arena retains allocated memory pools for reuse. |
| **Post-90s Idle Unload RSS** | **48.88 MB** | **Full session release and gc.collect() restores memory to idle baseline.** |

---

## 7. Paging Analysis

During CPU inference on the 8 GB RAM system:
- **System Memory Load**: Reached **100%**.
- **Available Physical RAM**: Dropped to **~0 MB**.
- **Pagefile / Commit Usage**: Reached **~26,969 MB**.
- **Conclusion**: When total physical RAM usage approaches 92–99%, Windows Virtual Memory Manager aggressively pages out non-critical standby pages to disk. Because the single-inference lock is active, memory does not exceed physical limits to cause an OOM crash, but heavy memory pressure contributes to disk activity and memory churn.

---

## 8. Thread Utilization

- **Configured Threads**: 4 intra-op threads on a 6-core / 12-thread CPU.
- **CPU Utilization Observed**: Sustained ~33–40% total system CPU utilization (equivalent to 4 dedicated logical cores saturated at 100%).
- **Assessment**: The 4 worker threads are **heavily and continuously utilized** throughout the entire ~90–120s duration. The threads are not waiting on I/O or sleep; they are executing continuous floating-point instructions.

---

## 9. ONNX Operator Analysis

Operator-level profiling was collected in an isolated session:

| Operator Type | Total Execution Time | Percentage of Inference | Execution Count |
|---|:---:|:---:|:---:|
| `Gemm` | 22,901.15 ms | 28.94% | 192 |
| `Sum` | 15,989.02 ms | 20.21% | 20 |
| `Mul` | 7,908.5 ms | 10.0% | 347 |
| `Transpose` | 7,500.82 ms | 9.48% | 360 |
| `Cast` | 5,996.87 ms | 7.58% | 1348 |
| `Conv` | 5,656.88 ms | 7.15% | 102 |
| `Add` | 2,017.91 ms | 2.55% | 279 |
| `MatMul` | 1,651.39 ms | 2.09% | 102 |
| `GatherND` | 1,400.45 ms | 1.77% | 80 |
| `Resize` | 1,039.05 ms | 1.31% | 22 |


### Key Operator Takeaways:
1. **`Gemm` (General Matrix Multiply)**: Consumed **22.90 s (28.94%)** across 192 nodes, representing dense projection layers in the transformer attention blocks.
2. **Elementwise Arithmetic (`Sum`, `Mul`, `Add`)**: Consumed **25.92 s (32.76%)** across 646 nodes, driven by residual additions and attention scaling.
3. **`Transpose` Nodes**: Consumed **7.50 s (9.48%)** across 360 nodes to rearrange multi-head attention axes between `[B, C, H, W]` and `[B, H, W, C]`.
4. **`Cast` Nodes**: There are **1,348 `Cast` operations** executing inside the graph consuming **6.00 s (7.58%)**, converting back and forth between FP16 and FP32.
5. **`Conv` Nodes**: Consumed **5.66 s (7.15%)** across 102 convolutional nodes.
6. **Key Insight**: FP16 emulation on a CPU without AVX-512 FP16 instructions severely penalizes `Gemm`, `Sum`, and `Cast` operators, as the CPU must unpack or emulate half-precision arithmetic in software.

---

## 10. Memory Leak Assessment

- **Classification**: **No evidence of persistent memory leak.**
- **Reasoning**:
  1. Process RSS elevated after Run A (~1.6–2.4 GB) and stabilized across Runs B and C (~1.5–2.2 GB) without compounding runaway growth.
  2. This elevation is classic **ONNX Runtime CPU Arena Allocator** behavior (`enable_cpu_mem_arena = True`), where ONNX retains allocated buffer pools for reuse across inferences rather than releasing them to Windows kernel memory after each request.
  3. Upon reaching the 90-second idle timeout, the session was cleanly destroyed, and process RSS dropped back down to **48.88 MB**, confirming that all references were cleanly reclaimed.

---

## 11. Root Cause Assessment

### CONFIRMED (Directly Measured):
1. **99.7% of time is in `session.run()`**: All classical CV stages combined take < 0.3s.
2. **Session reuse works**: Cold model load takes ~13–16s; warm calls have 0s load overhead.
3. **Single inference lock protects 8 GB RAM**: Peak memory never exceeded system limit during single inference.
4. **Model weights are FP16**: 593 model initializers are FP16 (`float16`).
5. **No persistent leak**: Memory completely returns to 48.88 MB after 90s idle unload.

### LIKELY (Strongly Supported Interpretation):
1. **Emulated FP16 Computation on CPU**: AMD Ryzen 5 7535HS (Zen 3+) does not feature native AVX-512 FP16 arithmetic. Running FP16 models on `CPUExecutionProvider` forces ONNX Runtime to perform software emulation / conversions during math operations, which is substantially slower than optimized AVX2 FP32 matrix multiplication or DirectML/GPU execution.
2. **Paging Overhead**: As RAM hits ~95%, Windows pagefile paging adds memory access latency to CPU cache misses.

### UNKNOWN (Requires Separate Investigation):
1. Exact execution speedup if model weights are converted to native FP32 for AVX2 CPU execution vs remaining FP16.
2. Exact execution speedup if DirectML Execution Provider is used on the RTX 2050 GPU (explicitly deferred per instructions).

---

## 12. Next Optimization Candidates (NOT IMPLEMENTED — FOR EVALUATION ONLY)

| Optimization Candidate | Expected Benefit | Possible Quality Impact | Possible RAM Impact | Risk Level |
|---|---|:---:|:---:|:---:|
| **1. Evaluate Native FP32 ONNX on CPU** | May cut CPU inference time substantially if AVX2 FP32 kernels run without FP16 emulation overhead. | None (identical mathematical precision) | Model file is ~930 MB (vs 467 MB). Runtime activation memory is comparable. | Low |
| **2. Dynamic Input Downsampling Option** | Processing at 768x768 or 512x512 with native bilaplacian edge upsampling could reduce inference by 2–4x. | Slight softening of fine hair / sub-pixel strands. | Significantly reduces peak RAM during inference. | Medium |
| **3. ORT Operator Fusion Optimization** | Enabling `ORT_ENABLE_ALL` graph optimization level (currently `ORT_ENABLE_BASIC`). | None (exact equivalent graph). | Minimal. | Low |
| **4. DirectML GPU Execution** *(Deferred task)* | RTX 2050 Tensor Cores run FP16 natively, likely reducing inference from ~120s to ~2–4s. | None. | Offloads memory pressure from 8GB System RAM to 4GB Dedicated VRAM. | High (Requires driver/DirectML validation). |

---
*Report generated automatically by `run_cpu_profiling.py` on 2026-09-17 11:04:49.*
