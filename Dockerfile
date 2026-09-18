FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3 and system libraries required for OpenCV & Pillow
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend dependencies definition
COPY backend/requirements.txt ./backend/requirements.txt

# Install backend Python dependencies and GPU-enabled ONNX Runtime
RUN pip3 install --no-cache-dir -r backend/requirements.txt && \
    pip3 install --no-cache-dir onnxruntime-gpu==1.20.1

# Copy backend source code and ONNX models into container
COPY backend/ ./backend/

# Expose FastAPI server port
EXPOSE 8000

# Production startup command: 1 Uvicorn worker bound to 0.0.0.0:8000
CMD ["python3", "-m", "uvicorn", "backend.background_remover.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
