# UNIVERSAL IMAGE CONVERTER & AI IMAGE PROCESSING SUITE
## Complete Technical Architecture, Data Flows, AI Models & System Documentation

---

## 1. PROJECT OVERVIEW (POINT-BY-POINT)

- **Universal Multi-Format Image Converter**:
  - Converts images across 12+ standard and professional formats directly in the browser using pure TypeScript and HTML5 2D Canvas.
  - Supports formats: PNG, JPG, WEBP, ICO, GIF, TIFF, PSD, SVG, BMP, PDF, RAW/DNG, HEIC.
  - Preserves native image resolution, color space, and alpha transparency where supported.

- **High-Quality Multi-Resolution ICO Engine**:
  - Standard-compliant Windows ICO generator embedding 8 independent standard resolutions: `16×16`, `24×24`, `32×32`, `48×48`, `64×64`, `128×128`, `256×256`, `512×512`.
  - Every resolution is downsampled independently directly from the original source canvas.
  - Correct binary ICO header packing (`ICONDIR` + `ICONDIRENTRY`), with genuine 512×512 embedded PNG streams.

- **AI Background Removal System**:
  - High-precision subject & portrait segmentation using **BiRefNet General FP16 ONNX**.
  - Includes a full classical computer vision refinement pipeline: mask quality analysis, trimap generation, edge/hair refinement, specular/translucency analysis, and de-fringing halo removal.
  - Accelerated via **NVIDIA CUDA 12.4 / cuDNN 9.2 (2.04s warm inference on RTX 2050 GPU)** with automatic CPU fallback.

- **AI Super Resolution & Face Preservation System**:
  - Upscales images 2× and 4× using **Real-ESRGAN x2plus / x4plus FP16 ONNX**.
  - Features **YuNet Face Detection** with a **70/30 Face Preservation Blend** (70% AI detail + 30% original face blend) to prevent facial distortion.

- **RAM-Safe Infrastructure & Live Diagnostics**:
  - Thread-safe single-inference lock (`HEAVY_AI_LOCK`) to prevent RAM/VRAM exhaustion.
  - Lazy model loading on first request + warm session reuse in RAM.
  - 90-second non-blocking idle timer to unload sessions when inactive.
  - Microsecond-level Live Terminal Performance Monitor logging request timings and memory usage.

---

## 2. END-TO-END EXECUTION FLOW (FRONTEND & BACKEND WORKFLOWS)

### 2.1 Universal Client-Side Conversion Flow (Frontend Only)

```
[USER UPLOAD]
      │  Selects image files (PNG, JPG, WEBP, RAW, HEIC, etc.)
      ▼
[ANGULAR WORKSPACE]
      │  Renders 3D Glassmorphic preview canvas
      │  User selects target format (e.g. ICO, WEBP, PNG) & options
      ▼
[CONVERSION-STATE SERVICE]
      │  Extracts options: quality (0-100%), transparency (true/false), ICO dimensions (Multi-Res 16-512px)
      ▼
[IMAGE-ENCODERS SERVICE]
      ├── Standard Formats (PNG, JPG, WEBP) ──► Native HTML5 Canvas toBlob()
      ├── Multi-Res ICO Format ────────────────► Renders 8 independent canvases (16px..512px) + Binary ICONDIR Packing
      ├── Specialty Formats (TIFF, PSD, BMP) ──► Custom ArrayBuffer / DataView Binary Encoders
      └── PDF Format ──────────────────────────► Custom PDF/XObject Binary Stream Generator
      │
      ▼
[BLOB GENERATION & DOWNLOAD]
      │  Creates browser ObjectURL -> Triggers instant client-side download
```

- **Detailed Steps**:
  1. **Upload & Inspection**: The user drags/drops or uploads images into `WorkspaceComponent`. Images are wrapped into `ImageItem` signals.
  2. **Preview & Dimensions**: Frontend calculates original file size, native pixel dimensions, and estimates target file size.
  3. **Encoding Dispatch**: `ImageConverterService.convertSingleItem()` selects the target encoder from `ImageFormatEncoders`.
  4. **Multi-Res ICO Execution**: If target format is ICO:
     - 8 square canvases are created: `16×16`, `24×24`, `32×32`, `48×48`, `64×64`, `128×128`, `256×256`, `512×512`.
     - `ctx.imageSmoothingQuality = 'high'` downsamples each size independently from the source.
     - Proportional scaling centers the image with transparent padding.
     - Encodes each canvas to PNG bytes and constructs binary `ICONDIR` (6 bytes) + `ICONDIRENTRY` (16 bytes × 8).
  5. **Completion**: Generated Blob is saved, progress signal reaches 100%, and download starts automatically.

---

### 2.2 AI Background Removal Flow (Frontend ↔ Backend Integration)

```
[ANGULAR FRONTEND: BG-REMOVER COMPONENT]
      │  User uploads image & adjusts parameters (Sensitivity, Edge Softness, De-Fringe)
      │  Clicks "Remove Background" or "Proceed"
      ▼
[BG-REMOVAL-API SERVICE]
      │  Sends HTTP POST to `http://127.0.0.1:8000/api/remove-background`
      │  Payload: FormData(image_file, sensitivity, edge_softness, defringe_strength)
      ▼
[FASTAPI BACKEND: MAIN ROUTER]
      │  Validates image format, dimensions, and generates unique Request ID (e.g. BG-20260918-001)
      ▼
[MODEL MANAGER (SINGLETON)]
      │  Checks single-inference lock (`HEAVY_AI_LOCK.acquire(blocking=False)`)
      │  Checks if BiRefNet FP16 ONNX is loaded in RAM:
      │    - IF IDLE/COLD: Loads model into session via CUDAExecutionProvider / CPU fallback (~11.7s load)
      │    - IF READY/WARM: Reuses warm session instantly (0.00ms load time)
      ▼
[BIREFNET ADAPTER (INFERENCE)]
      │  Preprocesses image: Letterbox scale to 1024x1024 + ImageNet normalization
      │  Runs ONNX inference on `[1, 3, 1024, 1024]` tensor
      │  Applies Sigmoid activation -> Un-letterboxes probability map back to native resolution
      ▼
[CLASSICAL CV REFINEMENT PIPELINE]
      │  1. Mask Quality Analysis: Computes edge sharpness & contrast metrics
      │  2. Trimap Generation: Creates FG (255), Unknown (128), BG (0) bands
      │  3. Transition Band Filtering: Refines hair/fur details via bilateral alpha matte refinement
      │  4. Specular/Translucency Analysis: Handles glassware & semi-transparent objects
      │  5. De-fringing: Removes color halos & edge contamination from foreground boundary
      │  6. RGBA Assembly: Combines original RGB with final uint8 alpha matte
      ▼
[FASTAPI RESPONSE & LIVE MONITOR]
      │  Prints Live Terminal Performance Log (Load ms, Preprocess ms, ONNX ms, Postprocess ms, Total ms, Memory RSS/VRAM)
      │  Releases inference lock (`HEAVY_AI_LOCK.release()`)
      │  Schedules 90-second idle unload timer
      │  Returns PNG image bytes to Angular UI
      ▼
[ANGULAR UI DISPLAY]
      │  Displays transparent background cutout + side-by-side comparison slider
```

- **Detailed Steps**:
  1. **User Action**: User uploads an image in the Background Removal tab.
  2. **API Request**: `BgRemovalApiService` sends a multi-part POST request to FastAPI.
  3. **Inference Locking**: `ModelManager.try_acquire_inference()` acquires `HEAVY_AI_LOCK`. If another request is active, rejects concurrent execution safely.
  4. **ONNX Execution**: BiRefNet FP16 ONNX runs on `CUDAExecutionProvider` (NVIDIA RTX 2050 GPU, taking **2.04s**) or falls back to multi-threaded `CPUExecutionProvider`.
  5. **Matte Refinement**: OpenCV/NumPy refines raw mask probabilities into fractional alpha for fine hair, glass, and crisp edges while removing color halos.
  6. **Response & Telemetry**: FastAPI prints execution stats in backend terminal and sends the transparent RGBA image back to the Angular UI.

---

### 2.3 AI Super Resolution Flow (HD 2× & 4×)

```
[ANGULAR FRONTEND]
      │  User selects "Export HD 2x" or "Export HD 4x"
      ▼
[FASTAPI BACKEND: SUPER RESOLUTION ROUTER]
      │  Receives request to `/api/super-resolution` with factor (2 or 4)
      ▼
[REAL-ESRGAN ENGINE]
      │  1. Face Detection (YuNet ONNX): Scans image for human face bounding boxes
      │  2. Tiled Upscaling (Real-ESRGAN x2plus / x4plus ONNX):
      │     - Splits large image into overlapping tiles (with tile padding)
      │     - Runs Real-ESRGAN ONNX inference per tile on GPU/CPU
      │     - Merges tiles seamlessly to prevent boundary seams
      │  3. 70/30 Face Preservation Blend:
      │     - Blends 70% Real-ESRGAN enhanced face + 30% original face to eliminate AI hallucinations
      ▼
[RESPONSE & DOWNLOAD]
      │  Returns HD upscaled RGBA image back to Frontend
```

---

### 2.4 RAM Safety & Lifecycle Management Flow

```
[FIRST REQUEST] ─────► Load BiRefNet Model into RAM/VRAM (~11.7s) ───► State: READY
                             │
                             ▼
[WARM REQUESTS] ────► Reuse Warm Session (0.00ms load time) ────────► Reset Idle Timer
                             │
                             ▼
[INACTIVITY (90s)] ──► Idle Timer Triggers Unload ──────────────────► Release ONNX Session
                             │
                             ▼
[GARBAGE COLLECTION] ► gc.collect() frees RAM/VRAM ─────────────────► State: IDLE
```

---

## 3. TECHNOLOGY STACK (POINT-BY-POINT)

### 3.1 Frontend Framework & Logic
- **Framework**: **Angular 21** (`@angular/core`, `@angular/common`, `@angular/forms`, `@angular/router`).
- **Language**: **TypeScript 5.9** (Strict type checking enabled).
- **Reactive State**: Angular Signals (`signal`, `computed`, `effect`) for zero-lag UI updates.
- **Async Operations**: **RxJS 7.8** (`Observable`, `Subject`, `firstValueFrom`).
- **Image Processing**: HTML5 2D Canvas API (`HTMLCanvasElement`, `CanvasRenderingContext2D`, `ImageData`).

### 3.2 Frontend UI & Styling
- **Architecture**: **Vanilla 3D Glassmorphic CSS** (Zero external CSS frameworks or Tailwind dependencies).
- **Visual Design**: Dark glassmorphism, HSL color tokens, backdrop blur filters, 3D transform cards, dynamic format pills, smooth micro-animations.
- **Typography**: Google Fonts (*Outfit* for headings, *Inter* for body text).

### 3.3 Backend API & Infrastructure
- **Web Framework**: **FastAPI** (`0.128.1`) + **Starlette**.
- **ASGI Server**: **Uvicorn** (`0.40.0`).
- **Environment**: **Python 3.13 (64-bit)**.

### 3.4 AI Inference & Acceleration
- **ONNX Engine**: **ONNX Runtime** (`1.24.4` / `1.30.0`).
- **Execution Providers**:
  - `CUDAExecutionProvider` (NVIDIA CUDA 12.4 + cuDNN 9.2 GPU Acceleration).
  - `DmlExecutionProvider` (DirectML DirectX 12 GPU Acceleration).
  - `CPUExecutionProvider` (Multi-threaded CPU with `ORT_ENABLE_BASIC`).
- **Deep Learning Suite**: **PyTorch 2.6+cu124** (`torch`, `torchvision`).

### 3.5 Image Processing & System Libraries
- **Computer Vision**: **OpenCV** (`opencv-python` 5.0 / `cv2`).
- **Image Manipulation**: **Pillow / PIL** (`12.0.0`).
- **Numerical Operations**: **NumPy** (`2.3.5` / `2.5.3`), **SciPy** (`1.17.0`).
- **System Telemetry**: **psutil** (`7.2.1`), **WMI**, **PyWin32**.

---

## 4. AI MODELS & SPECIFICATIONS (POINT-BY-POINT)

### 4.1 BiRefNet General FP16 ONNX (Background Removal)
- **Model Name**: BiRefNet General (Half Precision FP16 ONNX).
- **Architecture**: Bilateral Reference Network for Dichotomous Image Segmentation.
- **File Location**: `backend/models/segmentation/birefnet-general/model_fp16.onnx`.
- **License**: MIT License (Commercial Friendly).
- **Input Tensor**: `[1, 3, 1024, 1024]` float32.
- **Output Tensor**: `[1, 1, 1024, 1024]` float32 raw logits map.
- **Preprocessing**: ImageNet Mean `[0.485, 0.456, 0.406]` & Std `[0.229, 0.224, 0.225]` with aspect-ratio letterboxing padding `(124, 116, 104)`.
- **Inference Speed**: **2.04s on RTX 2050 CUDA GPU** | **52.94s on 8-thread CPU**.

### 4.2 Real-ESRGAN x4plus FP16 ONNX (Super Resolution 4×)
- **Model Name**: Real-ESRGAN x4plus (FP16 ONNX).
- **File Location**: `backend/models/super_resolution/realesrgan-x4plus.onnx`.
- **Function**: 4× deep learning image upscaling and noise reduction.
- **Tiling**: Overlap tile rendering to prevent GPU memory allocation limits on large photos.

### 4.3 Real-ESRGAN x2plus FP16 ONNX (Super Resolution 2×)
- **Model Name**: Real-ESRGAN x2plus (FP16 ONNX).
- **File Location**: `backend/models/super_resolution/realesrgan-x2plus.onnx`.
- **Function**: Fast 2× image upscaling.

### 4.4 YuNet ONNX (Face Detection & Blending)
- **Model Name**: YuNet Face Detector (ONNX).
- **File Location**: `backend/models/face_detection/face_detection_yunet.onnx`.
- **Function**: Detects human facial bounding boxes.
- **Blending Strategy**: **70/30 Face Preservation Blend** — 70% Real-ESRGAN enhanced facial features + 30% original face blend to guarantee 0% facial distortion.

---

## 5. MULTI-RESOLUTION ICO ENGINE (POINT-BY-POINT)

- **Pure TypeScript Implementation**: Located in `src/app/services/image-encoders.ts` (`canvasToIcoBlob`).
- **Standard Sizes Embedded**: `16×16`, `24×24`, `32×32`, `48×48`, `64×64`, `128×128`, `256×256`, `512×512`.
- **Independent Downsampling**: Each entry is downsampled **independently directly from the high-resolution original source**.
- **Aspect Ratio & Centering**: Proportional scaling fits the image on a square canvas with transparent padding.
- **Smoothing Quality**: `ctx.imageSmoothingQuality = 'high'` (Bicubic quality resampling).
- **Binary Packing**: Constructs binary `ICONDIR` (6 bytes) + `ICONDIRENTRY` (16 bytes × 8 entries).
- **512×512 Standard Compliance**: Directory dimension bytes set `bWidth = 0` and `bHeight = 0` (per 1-byte ICO spec for ≥256px), while the embedded PNG stream contains genuine `512×512` pixels verified via PNG `IHDR` metadata.

---

## 6. SUPPORTED FILE FORMATS TABLE

| Format | Extension | Tech Stack | Transparency | Primary Use Case |
| ------ | --------- | ---------- | ------------ | ---------------- |
| **PNG** | `.png` | Native Canvas API | Yes | Lossless web graphics & cutouts |
| **JPEG** | `.jpg` / `.jpeg` | Native Canvas API | No | Compressed photography |
| **WebP** | `.webp` | Native Canvas API | Yes | Modern high-efficiency web standard |
| **ICO** | `.ico` | Pure TS Binary Encoder | Yes | Windows Icons (Multi-Res 16px to 512px) |
| **GIF** | `.gif` | Custom LZW TS Encoder | Yes (Indexed) | Animated & 256-color web images |
| **TIFF** | `.tif` | Custom IFD TS Encoder | Yes | Professional uncompressed print |
| **PSD** | `.psd` | Custom Raw Channel TS | Yes | Adobe Photoshop compatibility |
| **SVG** | `.svg` | Custom XML Generator | Yes | Vector container with embedded canvas data |
| **BMP** | `.bmp` | Custom BITMAPINFO TS | No | Standard Windows bitmap format |
| **PDF** | `.pdf` | Custom PDF/XObject TS | No | High-DPI Adobe PDF document generation |
| **RAW / DNG** | `.raw` / `.dng` | Custom Tiff Container | Yes | Uncompressed camera digital negative |
| **HEIC** | `.heic` | Custom Container | Yes | High-efficiency iOS image container |

---

## 7. HARDWARE ACCELERATION & BENCHMARK PERFORMANCE

### Verified Benchmark Summary (NVIDIA RTX 2050 4 GB GPU / AMD Ryzen 5 7535HS)

- **BiRefNet CUDA Acceleration**: **2.04s warm inference** (**35.19x faster** than 71.64s CPU warm average).
- **BiRefNet CPU Scaling**: 1 Thread (**109.05s**), 2 Threads (**75.91s**), 4 Threads (**55.15s**), 8 Threads (**52.94s**).
- **BiRefNet ONNX Graph Optimization**: `ORT_ENABLE_BASIC` is optimal (10.11s load vs 43.25s load on `ORT_DISABLE_ALL`).
- **DirectML Status**: Failed with `8007000E` OOM at 1024×1024 fusion node on 4 GB VRAM (CUDA is the preferred GPU provider).
- **Multi-Resolution ICO Encoding**: **0.05 ms encoding time** for all 8 standard icon entries.

---

## 8. PROJECT DIRECTORY STRUCTURE (POINT-BY-POINT)

- `src/app/components/workspace/`: Interactive 3D glassmorphic converter UI.
- `src/app/components/bg-remover/`: AI Background Removal preview component.
- `src/app/services/image-encoders.ts`: Pure TS multi-format & multi-resolution ICO encoders.
- `src/app/services/image-converter.service.ts`: Batch conversion pipeline service.
- `src/app/services/conversion-state.service.ts`: Global state & signals management.
- `src/app/services/bg-removal-api.service.ts`: FastAPI backend HTTP bridge.
- `backend/background_removal_engine.py`: BiRefNet AI engine + classical CV alpha refinement pipeline.
- `backend/segmentation/model_manager.py`: Singleton model lifecycle manager (lazy load, lock, 90s idle timer).
- `backend/segmentation/model_adapter.py`: BiRefNet pre/postprocessing adapter.
- `backend/processing/super_resolution.py`: Real-ESRGAN + YuNet 70/30 face preservation engine.
- `backend/processing/phase10_profiler.py`: Monotonic microsecond profiler.
- `backend/tests/test_ico_conversion.py`: Automated ICO multi-resolution test validator.
- `backend/tests/phase12_cuda_benchmark.py`: Controlled CUDA RTX 2050 benchmark tool.

---

## 9. HOW TO RUN THE APPLICATION

- **Start Angular Frontend**:
  ```bash
  npm start
  # App available at http://localhost:4200
  ```

- **Start FastAPI Backend**:
  ```bash
  npm run backend
  # API available at http://127.0.0.1:8000
  ```

- **Build Production Frontend**:
  ```bash
  npm run build
  ```

- **Run Automated ICO Test Suite**:
  ```bash
  py -3.13 backend/tests/test_ico_conversion.py
  ```

- **Run Phase 12 CUDA RTX 2050 Benchmark**:
  ```bash
  py -3.13 backend/tests/phase12_cuda_benchmark.py
  ```
