# Model Licenses & Attribution

This document details the official licenses, sources, and attributions for the AI segmentation models used in the Background Removal Engine.

---

## 1. BiRefNet General FP16 (Bilateral Reference for Dichotomous Image Segmentation)
- **Model Name**: BiRefNet (General Dichotomous Image Segmentation)
- **Model Architecture**: Bilateral Reference with High-Resolution Localization and Guidance
- **Model Source**: [ZhengPeng7/BiRefNet](https://github.com/ZhengPeng7/BiRefNet) / [onnx-community/BiRefNet-ONNX](https://huggingface.co/onnx-community/BiRefNet-ONNX)
- **Model License**: **MIT License**
- **Model File**: `backend/models/segmentation/birefnet-general/model_fp16.onnx`
- **Original Model URL**: https://github.com/ZhengPeng7/BiRefNet
- **Required Attribution**:
  ```
  Copyright (c) 2024 Peng Zheng, Dehong Gao, et al.
  Licensed under the MIT License.
  Reference: Zheng et al., "Bilateral Reference for High-Resolution Dichotomous Image Segmentation", CAAI AIR 2024.
  ```

---

## 2. U²-Net (Rollback & Hardware Protection Model)
- **Model Name**: U²-Net (Nested U-Structure for Salient Object Detection)
- **Model License**: **Apache License 2.0**
- **Model File**: `backend/models/u2net.onnx`
- **Purpose**: Hardware memory constraint fallback protection for systems with limited RAM/VRAM.
- **Original Model URL**: https://github.com/xuebinqin/U-2-Net
- **Required Attribution**:
  ```
  Copyright (c) 2020 Xuebin Qin
  Licensed under the Apache License 2.0.
  Reference: Qin et al., "U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection", Pattern Recognition 2020.
  ```
