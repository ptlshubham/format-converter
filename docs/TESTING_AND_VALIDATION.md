# Testing and Validation

This document catalogs all verified automated test scripts, benchmark suites, and validation procedures available in the repository.

---

## 1. Automated Backend Test Suites

| Test Script | Location | Purpose | Execution Command | Status |
| :--- | :--- | :--- | :--- | :--- |
| **ICO Multi-Res Test** | `backend/tests/test_ico_conversion.py` | Validates multi-res ICO generation (16x16 to 512x512) & binary headers | `python backend/tests/test_ico_conversion.py` | **VERIFIED / PASS** |
| **CUDA GPU Benchmark** | `backend/tests/phase12_cuda_benchmark.py` | Benchmark BiRefNet FP16 ONNX on NVIDIA CUDA GPU (35.19x speedup test) | `python backend/tests/phase12_cuda_benchmark.py` | **VERIFIED / PASS** |
| **CPU Threads Benchmark** | `backend/tests/run_cpu_production_benchmark.py` | Benchmark 4 vs 6 vs 8 threads on CPU with true peak RSS sampling | `python backend/tests/run_cpu_production_benchmark.py` | **VERIFIED / PASS** |
| **Comprehensive BiRefNet** | `backend/tests/test_birefnet_comprehensive.py` | End-to-end pipeline test across multiple sample images | `python backend/tests/test_birefnet_comprehensive.py` | **VERIFIED / PASS** |
| **RAM Safety Test** | `backend/tests/test_ram_safe_architecture.py` | Verifies single-inference lock, idle unload, and RAM limits | `python backend/tests/test_ram_safe_architecture.py` | **VERIFIED / PASS** |
| **Face Preservation Test** | `backend/tests/test_phase2_face_preservation.py` | Validates YuNet face detection & 70/30 blending during upscale | `python backend/tests/test_phase2_face_preservation.py` | **VERIFIED / PASS** |

---

## 2. Frontend Test Suites

- **Angular Unit Tests**: Run `npm test` (Karma + Jasmine runner).
- **Client-Side ICO Generation Validation**: `test_data/generate_angular_ico.js` script verifies node-canvas ICO generation parity.

---

## 3. Manual Verification Procedures

### A. ICO Conversion Verification
1. Upload high-res PNG image in workspace.
2. Select target format **ICO**.
3. Convert and download ICO file.
4. Inspect inside icon viewer / OS file explorer at 512×512 display scale.
5. Verify 0 pixelation or blurriness.

### B. Background Removal Verification
1. Open Background Remover modal.
2. Upload image.
3. Adjust Sensitivity, Edge Softness, and De-Fringe sliders.
4. Click **Proceed**.
5. Verify cutout alpha matte accuracy around transparent glassware and fine hair.
