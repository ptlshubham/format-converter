# AI Engine Specification

## 1. BiRefNet General FP16 ONNX Background Removal

### Model Overview
- **Model Name**: BiRefNet General (Bilateral Reference Network for High-Resolution Dichotomous Image Segmentation)
- **Model Variant**: `general` (FP16 ONNX)
- **Model File Path**: `backend/models/segmentation/birefnet-general/model_fp16.onnx`
- **File Size**: ~180 MB
- **License**: MIT License (Commercial Friendly)
- **Precision**: FP16 ONNX (Float16 graph exported format)

### Tensor Specification
- **Input Tensor**: `[1, 3, 1024, 1024]` Float32 (RGB normalized)
- **Output Tensor**: `[1, 1, 1024, 1024]` Float32 (Raw Logits)

### Preprocessing Pipeline
1. **Letterbox Padding**: Aspect-ratio preserving resize to maximum 1024 dimension. Centered with constant background color padding `(124, 116, 104)`.
2. **ImageNet Normalization**:
   - `mean = [0.485, 0.456, 0.406]`
   - `std = [0.229, 0.224, 0.225]`
   - Single-pass formula: `norm_img = padded_img * inv_std_255 - mean_over_std`
3. **NCHW Transpose**: Transposed from `[1024, 1024, 3]` (HWC) to `[1, 3, 1024, 1024]` (NCHW) contiguous float32 array.

### Post-Processing & Refinement Pipeline
1. **Sigmoid Activation**: In-place logistic activation `1 / (1 + exp(-clip(logits, -50, 50)))`.
2. **Letterbox Crop & Un-resize**: Crops letterbox padding and resizes float probability map directly back to exact native image resolution using `cv2.INTER_LINEAR`.
3. **Mask Quality Evaluation**: Computes foreground pixel ratio, transparent pixel ratio, semi-transparent ratio, edge sharpness index, and mask entropy.
4. **Trimap Generation**:
   - Definite Foreground (`255`): Probability > `0.85`
   - Transition/Unknown (`128`): `0.15 <= Probability <= 0.85`
   - Definite Background (`0`): Probability < `0.15`
5. **Specular Transparency Analysis**: Detects specular highlights and glass transparent regions based on color gradient variance.
6. **Alpha Matte Refinement**: Fractional alpha preservation for glassware/hair edges; solid core enforcement for opaque foreground bodies.
7. **De-Fringing / Halo Removal**: Color decontamination on boundary pixels using background color sampling and edge color bleeding cancellation.
8. **RGBA Output Assembly**: Assembles clean RGB channels with refined uint8 `[0, 255]` alpha channel into final RGBA array.

---

## 2. Real-ESRGAN Super-Resolution Engine

### Model Overview
- **Models**:
  - `RealESRGAN_x4plus_fp16.onnx` (`backend/models/super_resolution/RealESRGAN_x4plus_fp16.onnx`)
  - `RealESRGAN_x2plus_fp16.onnx` (`backend/models/super_resolution/RealESRGAN_x2plus_fp16.onnx`)
- **Purpose**: High-definition 2x and 4x upscaling.

### Tiling & Overlap Architecture
- **Tile Size**: `512×512`
- **Tile Padding / Overlap**: `32px` border overlap
- **Seamless Blending**: Linear feathering across tile seams to prevent grid boundary lines.

---

## 3. YuNet Face Detection & Preservation

### Model Overview
- **Model File**: `backend/models/face_detection/face_detection_yunet_2023mar.onnx`
- **Purpose**: Facial detail detection during upscaling.

### Preservation Blending
- When human faces are detected inside the upscaled image:
  - Generates a smooth facial ROI mask with feathering.
  - Blends **70% original facial details** with **30% upscaled textures** inside the facial region to prevent AI facial warping or artificial skin distortion.

---

## 4. Execution Providers & Fallback Lifecycle

```
       ┌────────────────────────────────────────────────┐
       │             Incoming Inference Request          │
       └───────────────────────┬────────────────────────┘
                               │
                Is CUDA Execution Provider Available?
                               │
               ┌───────────────┴───────────────┐
               ▼                               ▼
            [ YES ]                         [ NO ]
               │                               │
┌──────────────┴───────────────┐   ┌───────────┴───────────────┐
│ Attempt CUDAExecutionProvider│   │   Use CPUExecutionProvider│
│ (NVIDIA RTX / CUDA 12.4)     │   │   (Configured CPU Threads)│
└──────────────┬───────────────┘   └───────────┬───────────────┘
               │                               │
     Did CUDA Init Succeed?                    │
               │                               │
       ┌───────┴───────┐                       │
       ▼               ▼                       │
    [ YES ]         [ NO ] (Fallback)          │
       │               └───────────────────────┤
       ▼                                       ▼
[ Execute on GPU ]                     [ Execute on CPU ]
```

---

## 5. Frozen AI Engine Components (DO NOT MODIFY)
The following components are protected under `docs/ENGINE_FREEZE_RULES.md`:
- `backend/models/segmentation/birefnet-general/model_fp16.onnx`
- `backend/models/super_resolution/RealESRGAN_x4plus_fp16.onnx`
- `backend/models/super_resolution/RealESRGAN_x2plus_fp16.onnx`
- `backend/models/face_detection/face_detection_yunet_2023mar.onnx`
- Preprocessing normalization formulas (`inv_std_255`, `mean_over_std`).
- Post-processing trimap thresholds and de-fringing formulas in `backend/processing/`.
