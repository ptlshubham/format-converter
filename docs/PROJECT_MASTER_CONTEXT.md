# Project Master Context

## 1. Project Name
**Image Converter & AI Image Processing Suite** (Angular 21 + FastAPI + BiRefNet General FP16 ONNX)

## 2. Project Purpose
This project is a high-performance web application designed for:
- **Client-side instant image format conversion & batch processing** (PNG, JPEG, WEBP, multi-resolution ICO, AVIF, BMP, TIFF, SVG, etc.).
- **High-precision AI-powered Background Removal** driven by **BiRefNet General FP16 ONNX** (mitigating halo/fringe artifacts with classical CV alpha matte refinement, defringing, specular transparency detection, and single-inference VRAM/RAM protection).
- **Super-Resolution Upscaling (2x / 4x HD Export)** powered by **Real-ESRGAN FP16 ONNX** combined with **YuNet ONNX face detection** for 70/30 facial preservation blending.

---

## 3. Main Features

### Client-Side Features
- **Instant Client-Side Format Conversion**: Processed via HTML5 Canvas API in browser without backend roundtrips for standard formats.
- **Multi-Resolution ICO Converter**: Generates valid ICO files containing 8 independent high-quality embedded sizes (`16x16`, `24x24`, `32x32`, `48x48`, `64x64`, `128x128`, `256x256`, `512x512`), each resampled directly from the original source image.
- **Interactive Workspace & Crop**: Image resizing, aspect ratio locks, quality sliders, and visual cropping.
- **Batch Export**: Zip creation using `JSZip` for multi-file exports.

### Backend Features
- **FastAPI Asynchronous Backend**: Runs on Uvicorn, serving endpoint routes for heavy machine learning inference.
- **Single-Inference Lock (`HEAVY_AI_LOCK`)**: Enforces strictly 1 AI inference at a time to prevent RAM/VRAM exhaustion.
- **Lazy Loading & 90-Second Idle Unload**: AI models load lazily on first request and automatically release VRAM/RAM after 90 seconds of inactivity.
- **RAM/VRAM Safety Limits**: Hard limit cap of 33.17M pixels (~5760×5760 max resolution) to prevent Out-Of-Memory (OOM) crashes.

### AI Features
- **BiRefNet General FP16 ONNX Background Removal**: Universal dichotomic salient object & portrait segmentation at 1024×1024 model resolution, mapped back to native image dimensions.
- **Alpha Matte & Fringe Refinement**: Trimap generation, transparency analysis, transition-band edge softness, and de-fringing decontamination.
- **Real-ESRGAN Super Resolution**: 2x (`x2plus`) and 4x (`x4plus`) FP16 upscaling with dynamic overlap tiling.
- **YuNet Face Preservation**: Detects human faces and blends 70% original facial details with 30% upscaled textures to avoid AI facial distortion artifacts.

### Export Features
- **Custom Background Compositing**: Export segmented cutouts with transparent, solid color, or custom gradient backgrounds.
- **HD Export (2x / 4x)**: High-definition export option using backend super-resolution engine.

### Diagnostic Features
- **Phase 10 Terminal Profiler**: Live ASCII execution graphs, memory telemetry (RSS, VRAM), stage-by-stage timing breakdowns, and mask statistics.
- **Diagnostic JSON Endpoints**: Detailed timing breakdown returned in API responses when `diagnostics=true`.

---

## 4. Technology Stack

| Layer | Technology | Version | Purpose |
| :--- | :--- | :--- | :--- |
| **Frontend Framework** | Angular | `^21.2.0` | Modern SPA with Standalone Components & Signals |
| **Frontend Language** | TypeScript | `~5.9.2` | Type-safe client logic |
| **Reactive State** | RxJS & Signals | `~7.8.0` | State management & image conversion queue |
| **UI & Styling** | Vanilla CSS | Custom 3D Glassmorphism | Dark glass UI system without Tailwind |
| **Client Compression** | JSZip | `^3.10.2` | Batch download zip archiving |
| **Backend Framework** | FastAPI | `>=0.115.0` | Asynchronous REST API framework |
| **ASGI Server** | Uvicorn | `>=0.30.0` | High-performance ASGI server |
| **Backend Runtime** | Python | `3.13` (64-bit) | Backend execution environment |
| **AI Machine Learning** | ONNX Runtime / GPU | `1.24.4` / `onnxruntime-gpu` | Model execution (`CPUExecutionProvider` / `CUDAExecutionProvider`) |
| **Computer Vision** | OpenCV | `>=4.9.0` (`opencv-python-headless`) | Matrix operations, face detection, resizing |
| **Image Processing** | Pillow (PIL) | `>=10.0.0` | Decoding, EXIF rotation handling, RGBA encoding |
| **Numerical Math** | NumPy & SciPy | `^2.3.5` / `^1.17.0` | Array manipulation & spatial distance transforms |

---

## 5. Supported Image Formats

| Format | Extension | Processing Location | Technology | Transparency Support | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PNG** | `.png` | Client / Backend | Canvas API / Pillow | FULL (Alpha) | Native lossless support |
| **WEBP** | `.webp` | Client / Backend | Canvas API / Pillow | FULL (Alpha) | Modern compressed format |
| **JPEG** | `.jpg`, `.jpeg` | Client / Backend | Canvas API / Pillow | NO (Solid BG) | Standard photo format |
| **ICO** | `.ico` | Client | HTML5 Canvas / Custom Encoder | FULL (Alpha) | Multi-res (16x16 to 512x512 PNG-encoded) |
| **AVIF** | `.avif` | Client (Browser native) | Canvas API | FULL (Alpha) | Supported in modern browsers |
| **BMP** | `.bmp` | Client / Backend | Canvas API / Pillow | Partial | Uncompressed bitmap |
| **TIFF** | `.tiff`, `.tif` | Backend / Client | Pillow / Canvas API | FULL | Professional print format |
| **SVG** | `.svg` | Client | DOM / Canvas API | FULL | Vector format conversion to raster |
| **GIF** | `.gif` | Client / Backend | Canvas API / Pillow | Indexed Alpha | Single frame export support |

---

## 6. AI Models

| Model Name | File Path | Purpose | Input Spec | Output Spec | Precision | Provider | Current Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **BiRefNet General** | `backend/models/segmentation/birefnet-general/model_fp16.onnx` | Salient Object & Portrait Segmentation | `[1, 3, 1024, 1024]` RGB float32 | `[1, 1, 1024, 1024]` Logits float32 | FP16 ONNX | CUDA / CPU | **VERIFIED / ACTIVE** |
| **Real-ESRGAN x4plus** | `backend/models/super_resolution/RealESRGAN_x4plus_fp16.onnx` | 4x Super Resolution Upscaling | `[1, 3, H, W]` RGB float32 | `[1, 3, 4H, 4W]` RGB float32 | FP16 ONNX | CUDA / CPU | **VERIFIED / ACTIVE** |
| **Real-ESRGAN x2plus** | `backend/models/super_resolution/RealESRGAN_x2plus_fp16.onnx` | 2x Super Resolution Upscaling | `[1, 3, H, W]` RGB float32 | `[1, 3, 2H, 2W]` RGB float32 | FP16 ONNX | CUDA / CPU | **VERIFIED / ACTIVE** |
| **YuNet Face Detector** | `backend/models/face_detection/face_detection_yunet_2023mar.onnx` | Human Face Detection for ROI preservation | `[1, 3, H, W]` RGB uint8 | Face Bounding Boxes & Landmarks | FP32 ONNX | OpenCV DNN / CPU | **VERIFIED / ACTIVE** |

---

## 7. Development & Benchmark Hardware (DEVELOPMENT ENVIRONMENT)
- **CPU**: AMD Ryzen 5 7535HS (6 Physical Cores / 12 Logical Threads)
- **RAM**: 8 GB DDR5
- **GPU**: NVIDIA GeForce RTX 2050 Laptop GPU (4 GB VRAM)
- **NVIDIA Driver**: `572.70`
- **CUDA Version**: CUDA 12.4 / cuDNN 9.2

---

## 8. Current Project Status

- **VERIFIED / WORKING**:
  - Client-side format conversion engine (PNG, WEBP, JPEG, multi-res ICO).
  - BiRefNet AI Background Removal engine (ONNX execution, alpha matte refinement, de-fringing).
  - Real-ESRGAN 2x/4x Super Resolution engine with YuNet face detail preservation.
  - Phase 12 NVIDIA RTX 2050 CUDA GPU benchmark (**35.19x speedup**: 2.04s CUDA warm avg vs 71.64s CPU warm avg).
  - Phase 13 Production CPU benchmark across 4, 6, and 8 threads.
  - Custom compositing (solid color, gradient, transparent backgrounds).

- **KNOWN LIMITATIONS**:
  - CPU inference execution takes ~60 seconds per image; GPU (NVIDIA CUDA) is strongly recommended for sub-3s response times in production.
  - Render Free Tier (0.5 vCPU, 512 MB RAM) is insufficient due to 30s gateway timeouts and 3.5 GB peak RSS during CPU inference.

---

## 9. Important Architecture Constraints
1. **Single-Inference Lock (`HEAVY_AI_LOCK`)**: Strictly 1 AI model inference runs at a time. Concurrent requests receive HTTP `429 Too Many Requests`.
2. **Lazy Loading**: Models load on the first incoming API request, avoiding startup overhead.
3. **90-Second Idle Unload**: Idle timer automatically releases model sessions from memory after 90 seconds.
4. **Safety Pixel Limit**: Images exceeding 33,177,600 total pixels (~5760×5760) are rejected to prevent RAM exhaustion.

---

## 10. Directory Overview

```
project-root/
├── src/                               # Angular 21 Frontend Source Code
│   ├── app/
│   │   ├── components/                # UI Components (workspace, bg-remover, result-screen, etc.)
│   │   ├── services/                  # Angular Services (conversion-state, bg-remover, encoders)
│   │   └── models/                    # Data interfaces & conversion options
├── backend/                           # FastAPI Python Backend Source Code
│   ├── background_remover/            # FastAPI app & route handlers
│   ├── segmentation/                  # BiRefNet model manager & adapter
│   ├── processing/                    # Classical CV pipeline (alpha refine, defringe, profiler)
│   ├── models/                        # Saved ONNX model files (BiRefNet, Real-ESRGAN, YuNet)
│   └── tests/                         # Benchmark scripts & automated test suite
└── docs/                              # Technical Documentation System
```
