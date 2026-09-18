# Deployment Guide

## 1. Target Deployment Architecture Options

### Option A: GPU Host (RECOMMENDED FOR PRODUCTION)
- **Frontend**: Hosted on Vercel, Netlify, or Static Web Host (`ng build`).
- **Backend**: Hosted on a GPU-enabled Cloud Instance (AWS EC2 g4dn/g5, RunPod, Modal, or Azure N-series with NVIDIA GPU).
- **Inference Runtime**: `onnxruntime-gpu` configured with `CUDAExecutionProvider` (CUDA 12.4, cuDNN 9.2).
- **Performance**: Sub-3s response times (~2.04s per 1408×768 image).

### Option B: High-Spec CPU Host
- **Frontend**: Vercel / Netlify (`ng build`).
- **Backend**: Dedicated CPU instance (minimum 4 to 8 physical CPU cores, 4 GB to 8 GB RAM).
- **Inference Runtime**: `onnxruntime` with `CPUExecutionProvider` (6 intra-op threads).
- **Performance**: ~59s processing time per image.

> [!WARNING]
> **Render Free Tier is INSUFFICIENT**: Render Free provides 0.5 vCPU and 512 MB RAM. BiRefNet model inference on CPU takes ~60s and peak RSS reaches ~3.5 GB RAM. Deploying on Render Free will cause instant Out-Of-Memory (OOM) crashes and 30s HTTP gateway timeouts.

---

## 2. Frontend Production Deployment (Angular 21)

### Build Command
```bash
npm run build
```
Generates production build artifacts inside `dist/image-converter/browser/`.

### Environment Configuration
Update API Base URL in `src/environments/environment.prod.ts` to point to your production FastAPI server URL:
```typescript
export const environment = {
  production: true,
  apiUrl: 'https://api.yourdomain.com'
};
```

---

## 3. Backend Production Deployment (FastAPI)

### Prerequisites
- **Python**: 3.13 (64-bit)
- **NVIDIA Driver**: Version 550+ (if deploying with GPU)
- **CUDA/cuDNN**: CUDA 12.4 and cuDNN 9.2 DLLs in PATH.

### Required Model Files Checklist
Ensure the following ONNX model files are placed in `backend/models/`:
- `backend/models/segmentation/birefnet-general/model_fp16.onnx`
- `backend/models/super_resolution/RealESRGAN_x4plus_fp16.onnx`
- `backend/models/super_resolution/RealESRGAN_x2plus_fp16.onnx`
- `backend/models/face_detection/face_detection_yunet_2023mar.onnx`

### Installation & Production Startup
```bash
# 1. Install Backend Dependencies
pip install -r backend/requirements.txt

# 2. Run Uvicorn Production Server
python -m uvicorn backend.background_remover.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> [!IMPORTANT]
> **Do NOT increase uvicorn `--workers` beyond 1** when running on a single GPU instance. Each worker process loads a separate 200MB+ model session into GPU VRAM. Use single-worker mode with FastAPI `asyncio.to_thread` and `HEAVY_AI_LOCK`.

---

## 4. Docker Deployment Configuration

### `Dockerfile`
```dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install Python 3.13 and system dependencies
RUN apt-get update && apt-get install -y \
    python3.13 \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements and install
COPY backend/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy backend code and models
COPY backend/ ./backend/

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "backend.background_remover.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

---

## 5. Deployment Verification Checklist

- [ ] `POST /api/background-remover/health` returns status `online` and provider `CUDAExecutionProvider`.
- [ ] Uploading test image to `/api/background-remover/remove` returns transparent PNG and valid base64 string.
- [ ] Single-inference lock rejects simultaneous duplicate request with HTTP `429`.
- [ ] Idle timer auto-unloads model after 90 seconds of inactivity.
- [ ] Memory RSS usage remains under 4 GB RAM.
