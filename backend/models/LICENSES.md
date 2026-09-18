# Model Licenses & Attribution

This document details the official licenses, sources, and attributions for the AI models used in the Image Processing Suite.

---

## 1. BiRefNet General FP16 (Bilateral Reference for Dichotomous Image Segmentation)
- **Model Name**: BiRefNet (General Dichotomous Image Segmentation)
- **Model Architecture**: Bilateral Reference with High-Resolution Localization and Guidance
- **Model Source**: [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) / [onnx-community/BiRefNet-ONNX](https://huggingface.co/onnx-community/BiRefNet-ONNX)
- **Model License**: **MIT License**
- **Model File**: `backend/models/segmentation/birefnet-general/model_fp16.onnx`
- **Required Attribution**:
  ```
  Copyright (c) 2024 Peng Zheng, Dehong Gao, et al.
  Licensed under the MIT License.
  Reference: Zheng et al., "Bilateral Reference for High-Resolution Dichotomous Image Segmentation", CAAI AIR 2024.
  ```

---

## 2. Real-ESRGAN x4plus (Super Resolution Upscaling)
- **Model Name**: Real-ESRGAN x4plus
- **Model Architecture**: Real-ESRGAN Deep Convolutional Neural Network
- **Model Source**: [xinntao/Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN)
- **Model License**: **BSD 3-Clause License**
- **Model File**: `backend/models/super_resolution/realesrgan_x4plus.onnx`
- **Required Attribution**:
  ```
  Copyright (c) 2021, Xintao Wang
  Licensed under the BSD 3-Clause License.
  Reference: Wang et al., "Real-ESRGAN: Training Real-World Blind Image Restoration with Pure Synthetic Data", ICCVW 2021.
  ```

---

## 3. YuNet Face Detector (Facial Landmark Detection for ROI Preservation)
- **Model Name**: YuNet Face Detector
- **Model Architecture**: YuNet Light-Weight Face Detector
- **Model Source**: [opencv/opencv_zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- **Model License**: **Apache License 2.0**
- **Model File**: `backend/models/face_detector/face_detection_yunet_2023mar.onnx`
- **Required Attribution**:
  ```
  Copyright (c) OpenCV Zoo Contributors
  Licensed under the Apache License 2.0.
  ```
