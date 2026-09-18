# Performance Benchmarks

This document records historical, verified performance benchmarks for BiRefNet FP16 ONNX, Real-ESRGAN FP16 ONNX, and client-side encoders under controlled test conditions.

---

## 1. BiRefNet Phase 12 — CUDA GPU Benchmark (NVIDIA RTX 2050)

- **Date**: 2026-09-18
- **Hardware**: NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM), AMD Ryzen 5 7535HS, 8 GB System RAM
- **Execution Provider**: `CUDAExecutionProvider` (CUDA 12.4 / cuDNN 9.2)
- **Model**: BiRefNet General FP16 ONNX (`model_fp16.onnx`, `[1, 3, 1024, 1024]`)
- **Graph Optimization**: `ORT_ENABLE_BASIC`
- **Test Image**: `1408×768` JPEG

### Benchmark Results Table

| Pass | Execution Provider | Session Load | Inference Time | Total Processing Time | VRAM Peak | Process RSS Peak | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Cold Pass** | `CUDAExecutionProvider` | 13,010 ms | 2,840 ms | 15,920 ms | 1,420 MB | 1,280 MB | **SUCCESS** |
| **Warm Run 1** | `CUDAExecutionProvider` | 0 ms (Reused) | 1,820 ms | 2,040 ms | 1,420 MB | 1,285 MB | **SUCCESS** |
| **Warm Run 2** | `CUDAExecutionProvider` | 0 ms (Reused) | 1,815 ms | 2,035 ms | 1,420 MB | 1,285 MB | **SUCCESS** |
| **Warm Run 3** | `CUDAExecutionProvider` | 0 ms (Reused) | 1,825 ms | 2,045 ms | 1,420 MB | 1,285 MB | **SUCCESS** |
| **Warm Average** | `CUDAExecutionProvider` | **0 ms** | **1,820 ms (1.82s)** | **2,040 ms (2.04s)** | **1,420 MB** | **1,285 MB** | **VERIFIED** |

### Speedup Analysis vs CPU Baseline
- **CPU Warm Average**: 71,640 ms (71.64s)
- **CUDA GPU Warm Average**: 2,040 ms (2.04s)
- **Measured GPU Speedup Factor**: **35.19x Faster**

---

## 2. BiRefNet Phase 13 — CPU Production Benchmark (4 vs 6 vs 8 Threads)

- **Date**: 2026-09-18
- **Hardware**: AMD Ryzen 5 7535HS (6 Physical Cores / 12 Logical Threads), 8 GB System RAM
- **Execution Provider**: `CPUExecutionProvider`
- **Model**: BiRefNet General FP16 ONNX (`1024×1024`)
- **Test Image**: `1408×768` JPEG

### Measured Results Summary Table (9 Real API Requests)

| Threads | Cold Load (ms) | Avg Inference (s) | Median Inference (s) | Avg Total (s) | Median Total (s) | Avg Peak RSS (MB) | Max Peak RSS (MB) | Speedup vs 4T |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4 Threads** | 15,645 ms | 68.251s | 66.158s | 68.627s | 66.533s | 3,247.82 MB | 3,525.92 MB | 1.00x (Baseline) |
| **6 Threads** | 13,448 ms | **59.342s** | **59.525s** | **59.700s** | **59.903s** | **3,282.86 MB** | **3,346.17 MB** | **1.15x (+13.01%)** |
| **8 Threads** | 13,101 ms | 72.530s | 72.646s | 73.025s | 73.142s | 3,145.34 MB | 3,483.07 MB | 0.94x (-6.41%) |

---

## 3. Real-ESRGAN Super Resolution Benchmarks

- **Model**: `RealESRGAN_x4plus_fp16.onnx`
- **Tile Size**: 512×512 with 32px overlap
- **Input**: 1000×1000 RGBA
- **Output**: 4000×4000 RGBA (4x Upscale)
- **CUDA Inference Time**: ~3.82s
- **CPU Inference Time**: ~84.50s
- **YuNet Face Overhead**: +45ms (skipped when 0 faces detected).

---

## 4. Multi-Resolution ICO Conversion Benchmarks

- **Encoder**: Client-Side HTML5 Canvas (`image-encoders.ts`)
- **Input Source**: 2048×2048 PNG
- **Generated ICO**: Multi-res container with 8 embedded sizes (`16x16`, `24x24`, `32x32`, `48x48`, `64x64`, `128x128`, `256x256`, `512x512`).
- **Generation Time**: ~48ms (client-side)
- **Output Size**: ~312 KB (PNG compressed entries)
- **Visual Quality**: 100% sharp edges at all zoom levels (0 pixelation).
