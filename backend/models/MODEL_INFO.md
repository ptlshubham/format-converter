# Model Information & Architecture

## Primary Segmentation Model: BiRefNet General FP16 ONNX
- **Model Name**: BiRefNet (General Dichotomous Image Segmentation)
- **Architecture**: Bilateral Reference with High-Resolution Localization and Guidance
- **File**: `backend/models/segmentation/birefnet-general/model_fp16.onnx`
- **File Size**: ~467.5 MB
- **Input Specification**: RGB image normalized `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`, shape `[1, 3, 1024, 1024]` with aspect-ratio preserving letterbox
- **Output Specification**: Continuous foreground logits `[1, 1, 1024, 1024]`, converted to probability via `sigmoid(logits)` (values in range `[0.0, 1.0]`), with un-letterboxing to native resolution
- **License**: **MIT License**
- **Repository**: https://github.com/ZhengPeng7/BiRefNet / https://huggingface.co/onnx-community/BiRefNet-ONNX

---

## Secondary Rollback Model: U²-Net
- **Model Name**: U²-Net (Nested U-Structure for Salient Object Detection)
- **File**: `backend/models/u2net.onnx`
- **File Size**: 175,997,641 bytes (~167.8 MB)
- **Input Specification**: RGB image normalized `(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])`, shape `[1, 3, 320, 320]`
- **Output Specification**: Continuous probability map `[1, 1, 320, 320]`
- **License**: **Apache License 2.0**
- **Purpose**: Engaged as automatic rollback protection when host hardware exhibits memory constraints (e.g. BFCArena allocation limits on limited RAM systems).
