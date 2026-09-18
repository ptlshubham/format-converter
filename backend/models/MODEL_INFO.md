# Model Information & Architecture

## 1. Primary Segmentation Model: BiRefNet General FP16 ONNX
- **Model Name**: BiRefNet (General Dichotomous Image Segmentation)
- **Architecture**: Bilateral Reference with High-Resolution Localization and Guidance
- **File**: `backend/models/segmentation/birefnet-general/model_fp16.onnx`
- **File Size**: ~467.0 MB
- **Input Specification**: RGB image normalized `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`, shape `[1, 3, 1024, 1024]` with aspect-ratio preserving letterbox
- **Output Specification**: Continuous foreground logits `[1, 1, 1024, 1024]`, converted to probability via `sigmoid(logits)` (values in range `[0.0, 1.0]`), with un-letterboxing to native resolution
- **License**: **MIT License**
- **Repository**: https://github.com/ZhengPeng7/BiRefNet / https://huggingface.co/onnx-community/BiRefNet-ONNX

---

## 2. Super Resolution Model: Real-ESRGAN x4plus ONNX
- **Model Name**: Real-ESRGAN x4plus
- **Architecture**: Real-ESRGAN Deep Convolutional Neural Network
- **File**: `backend/models/super_resolution/realesrgan_x4plus.onnx`
- **File Size**: ~64.0 MB
- **Input Specification**: RGB image normalized float32 tensor `[1, 3, H, W]` processed in 128x128 overlapping tiles
- **Output Specification**: 4x upscaled RGB float32 tensor `[1, 3, 4H, 4W]`
- **License**: **BSD 3-Clause License**
- **Repository**: https://github.com/xinntao/Real-ESRGAN

---

## 3. Face Detection Model: YuNet ONNX
- **Model Name**: YuNet Face Detector
- **Architecture**: Lightweight NMS Face Detector
- **File**: `backend/models/face_detector/face_detection_yunet_2023mar.onnx`
- **File Size**: ~232.5 KB
- **Input Specification**: RGB uint8 image `[1, 3, H, W]`
- **Output Specification**: Face bounding boxes, confidence scores, and facial landmark coordinates used for ROI detail preservation
- **License**: **Apache License 2.0**
- **Repository**: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
