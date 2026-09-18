# Engine Freeze Rules & Protection Policy

This document defines strict protection guidelines for critical AI models, core vision pipelines, and backend concurrency controls.

---

## 1. Protected System Modules (STRICT NO-TOUCH ZONE)

Do NOT modify, delete, replace, or reconfigure any of the following components without explicit written user approval and empirical benchmark validation:

### A. Background Removal Engine & Models
- **BiRefNet Model File**: `backend/models/segmentation/birefnet-general/model_fp16.onnx`
- **BiRefNet Adapter**: `backend/segmentation/model_adapter.py`
- **Model Manager**: `backend/segmentation/model_manager.py`
- **Background Removal Engine**: `backend/background_removal_engine.py`
- **Classical CV Post-Processing**:
  - `backend/processing/mask_quality.py`
  - `backend/processing/trimap.py`
  - `backend/processing/transparency_analyzer.py`
  - `backend/processing/alpha_refinement.py`
  - `backend/processing/defringe.py`
  - `backend/processing/alpha.py`

### B. Super-Resolution Engine & Models
- **Real-ESRGAN Models**:
  - `backend/models/super_resolution/RealESRGAN_x4plus_fp16.onnx`
  - `backend/models/super_resolution/RealESRGAN_x2plus_fp16.onnx`
- **Super-Resolution Engine**: `backend/processing/super_resolution.py`
- **YuNet Face Detector**: `backend/models/face_detection/face_detection_yunet_2023mar.onnx`

### C. Client-Side Image Encoders
- `src/app/services/image-encoders.ts` (Specifically multi-resolution ICO generator logic).

---

## 2. Mandatory Rules for Maintenance & Modifications

1. **No Silent Engine Changes**: Never swap BiRefNet or Real-ESRGAN for lightweight alternatives (e.g. U2Net, MediaPipe, RMBG) under the assumption of "speeding up" processing.
2. **No Model Weight Modification**: Do not re-export, quantize, or modify the FP16 ONNX model files.
3. **Preserve Single-Inference Lock (`HEAVY_AI_LOCK`)**: Never remove or bypass single-inference concurrency locks.
4. **Preserve Lazy Loading & 90s Idle Unload**: Do not remove idle model unloading mechanisms.
5. **No Loss of Resolution**: Native input dimensions (`orig_w × orig_h`) must strictly be preserved in segmented outputs.

---

## 3. Modification Approval Procedure

If a task requires modifying a protected module:
1. State the exact reason for the change.
2. Identify the target file and lines.
3. Run pre-change baseline benchmark using `backend/tests/phase12_cuda_benchmark.py` or `backend/tests/run_cpu_production_benchmark.py`.
4. Apply minimal isolated change.
5. Execute regression test suite to verify 100% mask parity and zero degradation.
6. Present findings to user before committing.
