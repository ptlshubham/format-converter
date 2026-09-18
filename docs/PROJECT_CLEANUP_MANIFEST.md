# Project Cleanup Manifest

This document categorizes all files, folders, and temporary build directories in the project workspace to ensure safe maintenance and deployment.

---

## 1. Production Required (KEEP AT ALL TIMES)

- `src/`: Angular 21 frontend application source code.
- `backend/`: FastAPI backend application source code.
  - `backend/models/`: Essential ONNX model files:
    - `backend/models/segmentation/birefnet-general/model_fp16.onnx`
    - `backend/models/super_resolution/RealESRGAN_x4plus_fp16.onnx`
    - `backend/models/super_resolution/RealESRGAN_x2plus_fp16.onnx`
    - `backend/models/face_detection/face_detection_yunet_2023mar.onnx`
- `docs/`: Technical documentation system.
- `package.json`, `package-lock.json`, `angular.json`, `tsconfig.json`: Frontend build configurations.
- `backend/requirements.txt`: Python backend dependency list.

---

## 2. Development Only (DO NOT INCLUDE IN DEPLOYMENT)

- `node_modules/`: Local node package dependencies.
- `.angular/`: Angular CLI local build cache.
- `.vscode/`: IDE configuration settings.
- `dist/`: Local build output folder (generated during `npm run build`).
- `birefnet-cuda-phase12/` (if present): Temporary standalone CUDA test folder created during Phase 12 investigation. Can be safely deleted without affecting production backend code.

---

## 3. Test Data & Artifacts (TEST ONLY)

- `test_data/`: Sample test images and benchmark inputs (`test_images/`, `test_human_images/`, `test_portrait_images/`).
- `backend/tests/results/`: Generated JSON and Markdown benchmark reports (`cpu_production_benchmark.json`, `cpu_production_benchmark_report.md`).
- `test_err.log`, `run.txt`: Local execution logs.
