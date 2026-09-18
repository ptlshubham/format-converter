# Image Converter & AI Image Processing Suite

An enterprise-grade, high-performance web application featuring client-side image format conversion, multi-resolution icon generation, and AI-powered background removal and super-resolution upscaling.

Built with an **Angular 21** frontend (standalone components, signals, vanilla 3D glassmorphic styling) and a **FastAPI** Python backend powered by **BiRefNet General FP16 ONNX**, **Real-ESRGAN x4plus/x2plus FP16 ONNX**, and **YuNet Face Preservation**.

---

## Key Features

- **Privacy-First Client-Side Format Conversion**: Instant format conversion directly in the browser via HTML5 Canvas 2D API with zero server upload for standard conversions.
  - **Supported Formats**: PNG, WEBP, JPEG, ICO, AVIF, BMP, TIFF, SVG, GIF.
  - **Multi-Resolution ICO Encoder**: Generates multi-frame `.ico` files containing 16x16, 32x32, 48x48, 64x64, 128x128, 256x256, and 512x512 PNG-compressed sub-images.
- **AI Background Removal**: State-of-the-art salient object and portrait background removal using **BiRefNet General FP16 ONNX** (`model_fp16.onnx`).
  - **Edge Refinement & De-fringing**: Advanced trimap generation, alpha matte smoothing, and edge color decontamination.
- **AI Super-Resolution Upscaling**: 2x (`x2plus`) and 4x (`x4plus`) FP16 super-resolution using **Real-ESRGAN** with dynamic tile overlapping.
  - **YuNet Face Detail Preservation**: Detects human facial landmarks and intelligently blends 70% original facial details with 30% upscaled textures to avoid facial distortion artifacts.
- **Custom Compositing Canvas**: Instant client-side background compositing with transparent, solid color, or custom gradient backgrounds.
- **Batch Processing & ZIP Export**: Convert multiple images in parallel and download all results in a single structured ZIP archive powered by JSZip.
- **3D Glassmorphism UI**: High-contrast, dark-mode visual interface with glowing glass panels, dynamic progress indicators, and responsive workspace layout.
- **Enterprise Safety & Memory Management**:
  - **Single-Inference Concurrency Lock (`HEAVY_AI_LOCK`)**: Prevents RAM/VRAM exhaustion under concurrent load.
  - **90-Second Idle Unload & Lazy Loading**: ONNX model sessions load on-demand and auto-unload after 90 seconds of inactivity.
  - **Resolution Cap**: Maximum image size restricted to 33.17M pixels (~5760x5760) for hardware protection.
- **Phase 10 Terminal Profiler**: ASCII execution timing graphs, RSS/VRAM telemetry, and stage-by-stage diagnostics.

---

## Technology Stack

### Frontend
- **Framework**: Angular `^21.2.0` (Standalone Components, Signals, Reactive Forms)
- **Language**: TypeScript `~5.9.2`
- **State & Streams**: RxJS `~7.8.0`
- **Styling**: Vanilla CSS with custom 3D Glassmorphism theme (No Tailwind dependencies)
- **Archiving**: JSZip `^3.10.2`

### Backend
- **Framework**: FastAPI `>=0.115.0`
- **ASGI Server**: Uvicorn `>=0.30.0`
- **Runtime**: Python `3.13` (64-bit)
- **AI / ML Runtime**: ONNX Runtime `1.24.4` (`CPUExecutionProvider` / `CUDAExecutionProvider`)
- **Computer Vision & Math**: OpenCV (`opencv-python-headless`), Pillow (PIL), NumPy, SciPy, `psutil`

---

## Supported Image Formats

| Format | Extension | Conversion Location | Engine / Library | Transparency Support | Highlights |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **PNG** | `.png` | Client / Backend | Canvas API / Pillow | Full Alpha | Lossless format, primary export format for AI cutouts |
| **WEBP** | `.webp` | Client / Backend | Canvas API / Pillow | Full Alpha | Highly efficient web compression |
| **JPEG** | `.jpg`, `.jpeg` | Client / Backend | Canvas API / Pillow | Solid BG | Standard photo format |
| **ICO** | `.ico` | Client | Custom Canvas Encoder | Full Alpha | Multi-resolution icon stack (16x16 to 512x512) |
| **AVIF** | `.avif` | Client | HTML5 Canvas | Full Alpha | Modern high-efficiency image format |
| **BMP** | `.bmp` | Client / Backend | Canvas API / Pillow | Partial | Standard bitmap format |
| **TIFF** | `.tiff`, `.tif` | Backend / Client | Pillow / Canvas | Full Alpha | Professional print format |
| **SVG** | `.svg` | Client | DOM / Canvas | Full Alpha | Vector to raster conversion |
| **GIF** | `.gif` | Client / Backend | Canvas API / Pillow | Indexed Alpha | Frame export support |

---

## Project Architecture & Directory Structure

```
format-converter/
├── src/                                  # Angular 21 Frontend Source Code
│   ├── app/
│   │   ├── components/                   # UI Components (workspace, bg-remover, result-screen)
│   │   ├── services/                     # State management, HTTP client, ICO/Canvas encoders
│   │   └── models/                       # Data interfaces & conversion parameters
│   └── environments/                     # Environment configuration (dev vs prod backend URLs)
├── backend/                              # FastAPI Backend Source Code
│   ├── background_remover/               # FastAPI application entrypoint & API endpoints
│   ├── segmentation/                     # BiRefNet ONNX model manager & session loader
│   ├── processing/                       # Matting refinement, de-fringing, face preservation & profiler
│   ├── models/                           # ONNX model weights (BiRefNet, Real-ESRGAN, YuNet)
│   ├── requirements.txt                  # Python dependencies
│   └── tests/                            # Diagnostic & CPU/GPU benchmark scripts
├── docs/                                 # Project Technical Documentation Suite
│   ├── PROJECT_MASTER_CONTEXT.md         # Master technical specification
│   ├── ARCHITECTURE.md                   # System architecture & data flow diagrams
│   ├── AI_ENGINE_SPECIFICATION.md        # Detailed ML model pipelines & post-processing math
│   ├── API_CONTRACT.md                   # OpenAPI endpoints & payload structures
│   ├── DEPLOYMENT_GUIDE.md               # GPU Docker & production deployment instructions
│   ├── PERFORMANCE_BENCHMARKS.md         # Hardware benchmark findings (CUDA vs CPU)
│   ├── ENGINE_FREEZE_RULES.md            # Engine lock & maintenance governance
│   └── TESTING_AND_VALIDATION.md         # Testing procedures & verification rules
├── Dockerfile                            # Production GPU Docker configuration (CUDA 12.4)
├── start-app.bat                         # One-click Windows launcher (Frontend + Backend)
├── start-backend.bat                     # Windows backend launcher
└── README.md                             # Project README documentation
```

---

## Getting Started

### Prerequisites

- **Node.js**: v18.0.0 or higher (v20+ recommended)
- **npm**: v9.0.0 or higher
- **Python**: 3.10+ (64-bit, Python 3.13 recommended)

---

### Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/image-converter.git
   cd image-converter
   ```

2. **Install Frontend Dependencies**:
   ```bash
   npm install
   ```

3. **Install Backend Dependencies**:
   ```bash
   cd backend
   pip install -r requirements.txt
   cd ..
   ```

---

### Running the Application Locally

#### Option 1: One-Click Launcher (Windows)
Double-click `start-app.bat` or run:
```cmd
start-app.bat
```
This automatically starts both the Python FastAPI backend on port 8000 and the Angular frontend on port 4200, then opens `http://localhost:4200` in your default browser.

#### Option 2: Manual Terminal Startup

1. **Start the FastAPI Backend**:
   ```bash
   npm run backend
   ```
   *Backend will run at `http://127.0.0.1:8000` with interactive API docs at `http://127.0.0.1:8000/docs`.*

2. **Start the Angular Frontend**:
   ```bash
   npm start
   ```
   *Frontend will run at `http://localhost:4200`.*

---

## API Endpoints Summary

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Backend status, hardware acceleration info, and memory state |
| `POST` | `/api/background-remover` | Processes image for background removal and optional HD super-resolution upscaling |

### Request Payload (`multipart/form-data`)
- `file`: Image file input (PNG, JPG, WEBP, etc.)
- `format`: Output image format (`png`, `webp`, `jpeg`)
- `scale`: Super resolution factor (`1` = off, `2` = 2x, `4` = 4x)
- `bg_color`: Custom background color hex or `transparent`
- `diagnostics`: Return detailed timing profiler JSON (`true`/`false`)

---

## Production Deployment

The project includes a production-ready container build configured for GPU acceleration:

```bash
docker build -t image-converter-backend .
docker run --gpus all -p 8000:8000 image-converter-backend
```

For detailed deployment strategies, see [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md).

---

## Technical Documentation Index

For in-depth architectural and engineering details, refer to the documentation suite in `docs/`:

- [PROJECT_MASTER_CONTEXT.md](docs/PROJECT_MASTER_CONTEXT.md) - Executive project summary & system specifications.
- [ARCHITECTURE.md](docs/ARCHITECTURE.md) - End-to-end data flow & component interactions.
- [AI_ENGINE_SPECIFICATION.md](docs/AI_ENGINE_SPECIFICATION.md) - BiRefNet, Real-ESRGAN, and YuNet pipeline specs.
- [API_CONTRACT.md](docs/API_CONTRACT.md) - Detailed REST API specification.
- [DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) - Containerization & GPU hosting guide.
- [PERFORMANCE_BENCHMARKS.md](docs/PERFORMANCE_BENCHMARKS.md) - CPU vs NVIDIA RTX CUDA benchmarks.
- [ENGINE_FREEZE_RULES.md](docs/ENGINE_FREEZE_RULES.md) - AI engine protection & freeze policies.
- [TESTING_AND_VALIDATION.md](docs/TESTING_AND_VALIDATION.md) - Verification procedures.

---

## License

This project is licensed under the MIT License.
